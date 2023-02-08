/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "node_compile_cache_module.h"
#include <securec.h>
#include "external/ge/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/compile_cache_policy/policy_register.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "common/math/math_util.h"
#include "graph/compute_graph.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/plugin/ge_util.h"

namespace {
constexpr ge::char_t const *kAttrSupportDynamicShape = "support_dynamicshape";
template<typename T>
size_t AttrValueSizeByType(const ge::AnyValue &attr_value) {
  (void)attr_value;
  return sizeof(T);
}

template<>
size_t AttrValueSizeByType<std::string>(const ge::AnyValue &attr_value) {
  std::string val;
  (void)attr_value.GetValue<std::string>(val);
  return val.length();
}

template<typename T>
size_t ListAttrValueSizeByType(const ge::AnyValue &attr_value)
{
    std::vector<T> values;
    (void)attr_value.GetValue<std::vector<T>>(values);
    return (sizeof(T) * values.size());
}

template<>
size_t ListAttrValueSizeByType<std::string>(const ge::AnyValue &attr_value)
{
    std::vector<std::string> values;
    (void)attr_value.GetValue<std::vector<std::string>>(values);
    size_t str_size = 0U;
    for (const auto &val : values) {
      str_size += val.length();
    }
    return str_size;
}

template<typename T>
size_t ListListAttrValueSizeByType(const ge::AnyValue &attr_value)
{
    std::vector<std::vector<T>> values;
    (void)attr_value.GetValue<std::vector<std::vector<T>>>(values);
    size_t cnt = 0U;
    for (const auto &vals : values) {
      cnt += vals.size();
    }
    return (sizeof(T) * cnt);
}

template<typename T>
ge::Status CopyAttrValueSizeByType(const ge::AnyValue &attr_value,
    uint8_t *base, const size_t max_size, size_t &offset) {
  T val;
  (void)attr_value.GetValue<T>(val);
  const auto mem_ret = memcpy_s((base + offset), (max_size - offset), &val, sizeof(T));
  if (mem_ret != EOK) {
    GELOGE(ge::FAILED, "memcpy failed.");
    return ge::FAILED;
  }
  offset += sizeof(T);
  return ge::SUCCESS;
}

template<>
ge::Status CopyAttrValueSizeByType<std::string>(const ge::AnyValue &attr_value,
    uint8_t *base, const size_t max_size, size_t &offset) {
  std::string val;
  (void)attr_value.GetValue<std::string>(val);
  if (val.empty()) {
    return ge::SUCCESS;
  }
  const auto mem_ret = memcpy_s((base + offset), (max_size - offset), val.data(), val.length());
  if (mem_ret != EOK) {
    GELOGE(ge::FAILED, "memcpy failed.");
    return ge::FAILED;
  }
  offset += val.length();
  return ge::SUCCESS;
}

template<typename T>
ge::Status CopyListAttrValueSizeByType(const ge::AnyValue &attr_value,
    uint8_t *base, const size_t max_size, size_t &offset)
{
  std::vector<T> values;
  (void)attr_value.GetValue<std::vector<T>>(values);
  for (const auto &val : values) {
    // is not use tmp value, &val compile failed in android
    T tmp_val = val;
    const auto mem_ret = memcpy_s((base + offset), (max_size - offset), &tmp_val, sizeof(T));
    if (mem_ret != EOK) {
      GELOGE(ge::FAILED, "memcpy failed.");
      return ge::FAILED;
    }
    offset += sizeof(T);
  }
  return ge::SUCCESS;
}

template<>
ge::Status CopyListAttrValueSizeByType<std::string>(const ge::AnyValue &attr_value,
    uint8_t *base, const size_t max_size, size_t &offset)
{
  std::vector<std::string> values;
  (void)attr_value.GetValue<std::vector<std::string>>(values);
  for (const auto &val : values) {
    if (val.empty()) {
      continue;
    }
    const auto mem_ret = memcpy_s((base + offset), (max_size - offset), val.data(), val.length());
    if (mem_ret != EOK) {
      GELOGE(ge::FAILED, "memcpy failed.");
      return ge::FAILED;
    }
    offset += val.length();
  }
  return ge::SUCCESS;
}

template<typename T>
ge::Status CopyListListAttrValueSizeByType(const ge::AnyValue &attr_value,
    uint8_t *base, const size_t max_size, size_t &offset)
{
  std::vector<std::vector<T>> values;
  (void)attr_value.GetValue<std::vector<std::vector<T>>>(values);
  for (const auto &vals : values) {
    for (const auto &val : vals) {
      const auto mem_ret = memcpy_s((base + offset), (max_size - offset), &val, sizeof(T));
      if (mem_ret != EOK) {
        GELOGE(ge::FAILED, "memcpy failed.");
        return ge::FAILED;
      }
      offset += sizeof(T);
    }
  }
  return ge::SUCCESS;
}
}  // namespace

namespace ge {
Status NodeCompileCacheItem::Build(const KernelLaunchBinType bin_type, const NodePtr &node, void *handle,
                                   NodeCompileCacheItem &item) {
  item.bin_type_ = bin_type;
  item.handle_ = handle;
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if ((!ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_KEY, item.op_compile_info_.key)) ||
      item.op_compile_info_.key.empty()) {
    GELOGW("Op[%s] does not have attr[%s].", op_desc->GetName().c_str(), COMPILE_INFO_KEY.c_str());
  }
  if ((!ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_JSON, item.op_compile_info_.str)) ||
      item.op_compile_info_.str.empty()) {
    GELOGW("Op[%s] does not have attr[%s].", op_desc->GetName().c_str(), COMPILE_INFO_JSON.c_str());
  }
  if ((!ge::AttrUtils::GetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, item.atomic_op_compile_info_.key)) ||
      item.atomic_op_compile_info_.key.empty()) {
    GELOGW("Op[%s] does not have attr[%s].", op_desc->GetName().c_str(), ATOMIC_COMPILE_INFO_KEY.c_str());
  }
  if ((!ge::AttrUtils::GetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, item.atomic_op_compile_info_.str)) ||
      item.atomic_op_compile_info_.str.empty()) {
    GELOGW("Op[%s] does not have attr[%s].", op_desc->GetName().c_str(), ATOMIC_COMPILE_INFO_JSON.c_str());
  }
  if (!ge::AttrUtils::GetInt(op_desc, kAttrOpParamSize, item.max_tiling_size_)) {
    GELOGW("Op[%s] does not have attr[%s].", op_desc->GetName().c_str(), kAttrOpParamSize);
  }
  if (!ge::AttrUtils::GetInt(op_desc, kAttrAtomicOpParamSize, item.atomic_max_tiling_size_)) {
    GELOGW("Op[%s] does not have attr[%s].", op_desc->GetName().c_str(), kAttrAtomicOpParamSize);
  }
  if (!ge::AttrUtils::GetBool(op_desc, kAttrSupportDynamicShape, item.is_dynamic_)) {
    GELOGW("Op[%s] does not have attr[%s].", op_desc->GetName().c_str(), kAttrSupportDynamicShape);
  }
  return SUCCESS;
}

uint64_t NodeCompileCacheItem::GetCacheItemId() const {
  return cache_item_id_;
}

void NodeCompileCacheItem::SetCacheItemId(const uint64_t cache_item_id) {
  cache_item_id_ = cache_item_id;
}

void *NodeCompileCacheItem::GetBinHandle() const {
  return handle_;
}

KernelLaunchBinType NodeCompileCacheItem::GetBinType() const {
  return bin_type_;
}

const optiling::OpCompileInfo *NodeCompileCacheItem::GetCompileInfo() const {
  return &op_compile_info_;
}

const optiling::OpCompileInfo *NodeCompileCacheItem::GetAtomicCompileInfo() const {
  return &atomic_op_compile_info_;
}

int64_t NodeCompileCacheItem::GetMaxTilingSize() const {
  return max_tiling_size_;
}
int64_t NodeCompileCacheItem::GetAtomicMaxTilingSize() const {
  return atomic_max_tiling_size_;
}

bool NodeCompileCacheItem::IsSupportDynamic() const {
  return is_dynamic_;
}

static Status GetTensorInfos(const OpDesc &op_desc, CompileCacheDesc &cache_desc, const bool need_range) {
  for (size_t i = 0U; i < op_desc.GetInputsSize(); ++i) {
    auto input_desc = op_desc.MutableInputDesc(static_cast<uint32_t>(i));
    if (input_desc == nullptr) {
      continue;
    }
    TensorInfoArgs tensor_info_args(input_desc->GetFormat(), input_desc->GetOriginFormat(), input_desc->GetDataType());
    // shape
    const auto &dims = input_desc->MutableShape().GetMutableDims();
    tensor_info_args.SetShape(dims);
    // origin shape
    const auto &origin_dims = input_desc->GetOriginShape().GetMutableDims();
    tensor_info_args.SetOriginShape(origin_dims);
    // shape range
    if (need_range) {
      std::vector<std::pair<int64_t, int64_t>> ranges;
      (void)input_desc->GetShapeRange(ranges);
      tensor_info_args.SetShapeRange(ranges);
    }
    cache_desc.AddTensorInfo(tensor_info_args);
  }
  return SUCCESS;
}

static Status GetOriginAttr(const std::string &op_type, std::set<string> &ordered_origin_attr) {
  auto node_op = ge::OperatorFactory::CreateOperator("node_op", op_type.c_str());
  if (node_op.IsEmpty()) {
    GELOGE(FAILED, "get op from OperatorFactory fail. opType: %s", op_type.c_str());
    return FAILED;
  }

  GELOGD("get op from OperatorFactory success. opType is %s", op_type.c_str());
  auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
  node_op.BreakConnect();
  if (temp_op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "GetOpDescFromOperator failed, return nullptr.");
    GELOGE(FAILED, "[Get][OpDesc] temp op desc is null");
    return FAILED;
  }
  const auto &ir_origin_attr_names = temp_op_desc->GetIrAttrNames();
  ordered_origin_attr.insert(ir_origin_attr_names.cbegin(), ir_origin_attr_names.cend());
  GELOGD("get origin attr name success, size is %zu", ordered_origin_attr.size());
  return SUCCESS;
}

size_t NodeCompileCacheModule::GetAttrSize(const AnyValue &attr_value) const {
  switch (attr_value.GetValueType()) {
    case ge::GeAttrValue::VT_STRING:
      return AttrValueSizeByType<std::string>(attr_value);
    case ge::GeAttrValue::VT_BOOL:
      return AttrValueSizeByType<bool>(attr_value);
    case ge::GeAttrValue::VT_INT:
        return AttrValueSizeByType<int64_t>(attr_value);
    case ge::GeAttrValue::VT_FLOAT:
      return AttrValueSizeByType<float32_t>(attr_value);
    case ge::GeAttrValue::VT_DATA_TYPE:
      return AttrValueSizeByType<ge::GeAttrValue::DATA_TYPE>(attr_value);
    case ge::GeAttrValue::VT_LIST_STRING:
      return ListAttrValueSizeByType<std::string>(attr_value);
    case ge::GeAttrValue::VT_LIST_BOOL:
      return ListAttrValueSizeByType<bool>(attr_value);
    case ge::GeAttrValue::VT_LIST_INT:
      return ListAttrValueSizeByType<int64_t>(attr_value);
    case ge::GeAttrValue::VT_LIST_FLOAT:
      return ListAttrValueSizeByType<float32_t>(attr_value);
    case ge::GeAttrValue::VT_LIST_DATA_TYPE:
      return ListAttrValueSizeByType<ge::GeAttrValue::DATA_TYPE>(attr_value);
    case ge::GeAttrValue::VT_LIST_LIST_INT:
      return ListListAttrValueSizeByType<int64_t>(attr_value);
    default:
      GELOGD("unsupported type %d", attr_value.GetValueType());
      return 0U;
  }
  return 0U;
}

Status NodeCompileCacheModule::CopyAttrValues(const AnyValue &attr_value,
                                              uint8_t *base,
                                              const size_t max_size,
                                              size_t &offset) const {
  switch (attr_value.GetValueType()) {
    case ge::GeAttrValue::VT_STRING:
      return CopyAttrValueSizeByType<std::string>(attr_value, base, max_size, offset);
    case ge::GeAttrValue::VT_BOOL:
      return CopyAttrValueSizeByType<bool>(attr_value, base, max_size, offset);
    case ge::GeAttrValue::VT_INT:
        return CopyAttrValueSizeByType<int64_t>(attr_value, base, max_size, offset);
    case ge::GeAttrValue::VT_FLOAT:
      return CopyAttrValueSizeByType<float32_t>(attr_value, base, max_size, offset);
    case ge::GeAttrValue::VT_DATA_TYPE:
      return CopyAttrValueSizeByType<ge::GeAttrValue::DATA_TYPE>(attr_value, base, max_size, offset);
    case ge::GeAttrValue::VT_LIST_STRING:
      return CopyListAttrValueSizeByType<std::string>(attr_value, base, max_size, offset);
    case ge::GeAttrValue::VT_LIST_BOOL:
      return CopyListAttrValueSizeByType<bool>(attr_value, base, max_size, offset);
    case ge::GeAttrValue::VT_LIST_INT:
      return CopyListAttrValueSizeByType<int64_t>(attr_value, base, max_size, offset);
    case ge::GeAttrValue::VT_LIST_FLOAT:
      return CopyListAttrValueSizeByType<float32_t>(attr_value, base, max_size, offset);
    case ge::GeAttrValue::VT_LIST_DATA_TYPE:
      return CopyListAttrValueSizeByType<ge::GeAttrValue::DATA_TYPE>(attr_value, base, max_size, offset);
    case ge::GeAttrValue::VT_LIST_LIST_INT:
      return CopyListListAttrValueSizeByType<int64_t>(attr_value, base, max_size, offset);
    default:
      GELOGD("unsupported type %d, no need to copy", attr_value.GetValueType());
      return SUCCESS;
  }
  return SUCCESS;
}

Status NodeCompileCacheModule::GetAttrTotalSize(const std::map<std::string, AnyValue> &all_attributes,
    const std::set<string> &ordered_origin_attr_name, size_t &attr_size) const {
  for (const auto &name : ordered_origin_attr_name) {
    GELOGD("current origin attr name is %s", name.c_str());
    auto it = all_attributes.find(name);
    if (it != all_attributes.end()) {
      const AnyValue &attr_value = it->second;
      FMK_SIZET_ADDCHECK(attr_size, it->first.length());
      attr_size += it->first.length();
      const size_t current_attr_size = GetAttrSize(attr_value);
      GELOGD("find attr name %s, size is %zu", it->first.c_str(), current_attr_size);
      FMK_SIZET_ADDCHECK(attr_size, current_attr_size);
      attr_size += current_attr_size;
    } else {
      GELOGD("can not get attr name %s", name.c_str());
    }
  }
  return SUCCESS;
}

Status NodeCompileCacheModule::CopyAttrToMem(const std::map<std::string, AnyValue> &all_attributes,
    std::unique_ptr<uint8_t[]> &attr_mem, const std::set<string> &ordered_origin_attr_name,
    const size_t attr_size) const {
  size_t offset = 0U;
  for (const auto &name : ordered_origin_attr_name) {
    auto it = all_attributes.find(name);
    if (it != all_attributes.end()) {
      const AnyValue &attr_value = it->second;
      FMK_SIZET_SUBCHECK(attr_size, offset);
      const auto mem_ret = memcpy_s((attr_mem.get() + offset), (attr_size - offset),
          it->first.data(), it->first.length());
      if (mem_ret != EOK) {
        GELOGE(FAILED, "memcpy failed.");
        return FAILED;
      }
      FMK_SIZET_ADDCHECK(offset, it->first.length());
      offset += it->first.length();
      if (CopyAttrValues(attr_value, attr_mem.get(), attr_size, offset) != SUCCESS) {
        GELOGE(FAILED, "copy attr mem failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status NodeCompileCacheModule::GetOpAttrMem(OpDesc &op_desc, CompileCacheDesc &cache_desc) const {
  if (op_desc.HasAttr("_origin_attr_value_bytes")) {
    Buffer attr_mem;
    if (AttrUtils::GetZeroCopyBytes(op_desc, "_origin_attr_value_bytes", attr_mem)) {
      GELOGD("this op %s has attr mem", op_desc.GetName().c_str());
      BinaryHolder binary_holder(attr_mem.GetData(), attr_mem.GetSize());
      cache_desc.AddBinary(binary_holder);
      return SUCCESS;
    }
  }
  std::set<string> ordered_origin_attr_name;
  GE_CHK_STATUS_RET_NOLOG(GetOriginAttr(op_desc.GetType(), ordered_origin_attr_name));
  GELOGD("get current origin attr size is %zu", ordered_origin_attr_name.size());
  if (ordered_origin_attr_name.empty()) {
    return SUCCESS;
  }
  size_t attr_size = 0U;
  const auto &all_attributes = op_desc.GetAllAttrs();
  GELOGD("get current attr size is %zu", all_attributes.size());
  GE_CHK_STATUS_RET_NOLOG(GetAttrTotalSize(all_attributes, ordered_origin_attr_name, attr_size));
  if (attr_size == 0U) {
    return SUCCESS;
  }
  GELOGD("get total size is %zu", attr_size);
  auto attr_mem = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[attr_size]);
  GE_CHECK_NOTNULL(attr_mem);
  GE_CHK_STATUS_RET_NOLOG(CopyAttrToMem(all_attributes, attr_mem, ordered_origin_attr_name, attr_size));

  Buffer attr_buf = Buffer::CopyFrom(attr_mem.get(), attr_size);
  (void)AttrUtils::SetZeroCopyBytes(op_desc, "_origin_attr_value_bytes", std::move(attr_buf));
  auto binary_holder = BinaryHolder::createFrom(std::move(attr_mem), attr_size);
  cache_desc.AddBinary(std::move(*binary_holder));
  return SUCCESS;
}

NodeCompileCacheModule::NodeCompileCacheModule()
    : ccp_(CompileCachePolicy::Create(MatchPolicyType::MATCH_POLICY_EXACT_ONLY, AgingPolicyType::AGING_POLICY_LRU)) {}

void NodeCompileCacheModule::Initialize() {
  if (ccp_ == nullptr) {
    ccp_ = CompileCachePolicy::Create(MatchPolicyType::MATCH_POLICY_EXACT_ONLY, AgingPolicyType::AGING_POLICY_LRU);
    GELOGD("Initialize ccm.");
  }
}

void NodeCompileCacheModule::Finalize() {
  ccp_.reset(nullptr);
  ids_to_cci_.clear();
  GELOGD("Finalize ccm.");
}

Status NodeCompileCacheModule::GetFusionOpCacheDesc(const NodePtr &node, CompileCacheDesc &cache_desc) const {
  // fusion op need name to identify
  cache_desc.SetOpType(node->GetName());
  auto compute_graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(compute_graph);
  const uint32_t graph_id = compute_graph->GetGraphID();
  const uint64_t session_id = compute_graph->GetSessionID();
  cache_desc.SetScopeId({session_id, graph_id});
  return SUCCESS;
}

Status NodeCompileCacheModule::GetInputConstTensor(const NodePtr &node, CompileCacheDesc &cache_desc) const {
  for (size_t index = 0U; index < node->GetAllInDataAnchorsSize(); ++index) {
    NodePtr input_node = nullptr;
    (void)NodeUtils::GetInNodeCrossPartionedCallNode(node, static_cast<uint32_t>(index), input_node);
    if ((input_node != nullptr) && (input_node->GetOpDesc() != nullptr)) {
      GeTensorPtr const_tensor = nullptr;
      (void)AttrUtils::MutableTensor(input_node->GetOpDesc(), ATTR_NAME_WEIGHTS, const_tensor);
      if ((const_tensor != nullptr) && (const_tensor->GetData().data() != nullptr) &&
          (const_tensor->GetData().size() > 0U)) {
        GELOGD("find node %s input node %s which has weight", node->GetName().c_str(), input_node->GetName().c_str());
        BinaryHolder holder_const(const_tensor->GetData().data(), const_tensor->GetData().size());
        GE_CHECK_NOTNULL(holder_const.GetDataPtr());
        cache_desc.AddBinary(std::move(holder_const));
      }
    }
  }
  return SUCCESS;
}

Status NodeCompileCacheModule::GetCompileCacheDescFromOp(const NodePtr &node,
    std::unique_ptr<CompileCacheDesc> &cache_desc, const bool need_range) const {
  auto cache_desc_tmp = MakeUnique<CompileCacheDesc>();
  GE_CHECK_NOTNULL(cache_desc_tmp);
  OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  ge::ComputeGraphPtr graph_ptr = nullptr;
  GE_CHK_STATUS_RET_NOLOG(GetTensorInfos(*op_desc, *cache_desc_tmp, need_range));
  GE_CHK_STATUS_RET_NOLOG(GetInputConstTensor(node, *cache_desc_tmp));
  if (ge::AttrUtils::GetGraph(op_desc, "_original_fusion_graph", graph_ptr)) {
    GELOGD("This is fusion op %s", node->GetName().c_str());
    GE_CHK_STATUS_RET_NOLOG(GetFusionOpCacheDesc(node, *cache_desc_tmp));
    cache_desc = std::move(cache_desc_tmp);
    return SUCCESS;
  }
  cache_desc_tmp->SetOpType(op_desc->GetType());
  GE_CHK_STATUS_RET_NOLOG(GetOpAttrMem(*op_desc, *cache_desc_tmp));
  cache_desc = std::move(cache_desc_tmp);
  return SUCCESS;
}

void NodeCompileCacheModule::UpdateTensorInfos(const NodePtr &node, CompileCacheDesc &cache_desc) const {
  const auto &op_desc = node->GetOpDesc();
  size_t index = 0U;
  for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
    auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (input_desc == nullptr) {
      continue;
    }
    // shape
    const auto &dims = input_desc->MutableShape().GetMutableDims();
    // origin shape
    const auto &origin_dims = input_desc->GetOriginShape().GetMutableDims();
    auto tensor_info = cache_desc.MutableTensorInfo(index);
    ++index;
    if (tensor_info == nullptr) {
      continue;
    }
    tensor_info->SetShape(dims);
    tensor_info->SetOriginShape(origin_dims);
  }
  return;
}

NodeCompileCacheItem *NodeCompileCacheModule::FindCompileCache(const NodePtr &node) {
  if ((ccp_ == nullptr) || (node == nullptr)) {
    return nullptr;
  }
  std::unique_ptr<CompileCacheDesc> cache_desc = nullptr;
  CompileCacheDesc *added_cache_desc = GetCompileCacheDesc(node);
  if (added_cache_desc != nullptr) {
    GELOGD("get compile cache desc from map successful");
    UpdateTensorInfos(node, *added_cache_desc);
  } else {
    GELOGW("get compile cache desc from op in find process");
    if (GetCompileCacheDescFromOp(node, cache_desc, false) != SUCCESS) {
      GELOGW("get cache desc failed in find process");
      return nullptr;
    }
    added_cache_desc = cache_desc.get();
  }
  CacheItemId id = ccp_->FindCache(*added_cache_desc);
  GELOGD("find cache item id is %lu, node name is %s", id, node->GetName().c_str());
  const std::lock_guard<std::mutex> lk(ids_to_cci_mu_);
  const auto it = ids_to_cci_.find(id);
  if (it == ids_to_cci_.end()) {
    GELOGD("can not find id %lu", id);
    return nullptr;
  } else {
    return &it->second;
  }
}

CompileCacheDesc *NodeCompileCacheModule::GetCompileCacheDesc(const NodePtr &node) {
  const uintptr_t node_id = PtrToValue(node.get());
  const std::lock_guard<std::mutex> lk(node_to_cache_desc_map_mu_);
  const auto it = node_to_cache_desc_map_.find(node_id);
  if (it == node_to_cache_desc_map_.end()) {
    GELOGW("can not get cache desc from map, node_id is %lu", node_id);
    return nullptr;
  }
  return it->second.get();
}

void NodeCompileCacheModule::InsertCompileCacheDesc(const NodePtr &node,
                                                    std::unique_ptr<CompileCacheDesc> &cache_desc) {
  const uintptr_t node_id = PtrToValue(node.get());
  const std::lock_guard<std::mutex> lk(node_to_cache_desc_map_mu_);
  node_to_cache_desc_map_[node_id] = std::move(cache_desc);
}

NodeCompileCacheItem *NodeCompileCacheModule::AddCompileCache(const NodePtr &node, NodeCompileCacheItem &item) {
  if ((ccp_ == nullptr) || (node == nullptr)) {
    return nullptr;
  }
  std::unique_ptr<CompileCacheDesc> cache_desc = nullptr;
  if (GetCompileCacheDescFromOp(node, cache_desc, true) != SUCCESS) {
    GELOGW("get cache desc failed in add process");
    return nullptr;
  }
  CacheItemId id = ccp_->AddCache(*cache_desc);
  GELOGD("add cache item id is %lu, node name is %s", id, node->GetName().c_str());
  InsertCompileCacheDesc(node, cache_desc);
  const std::lock_guard<std::mutex> lk(ids_to_cci_mu_);
  const auto it = ids_to_cci_.find(id);
  if (it == ids_to_cci_.end()) {
    item.SetCacheItemId(id);
    ids_to_cci_[id] = item;
    return &ids_to_cci_[id];
  } else {
    return &it->second;
  }
}
} // namespace ge
