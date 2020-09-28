/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "external/graph/graph.h"
#include "utils/attr_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/model_serialize.h"
#include "proto/ge_ir.pb.h"
#include "detail/model_serialize_imp.h"
#include "debug/ge_attr_define.h"
#include "debug/ge_log.h"
#include "debug/ge_util.h"

using std::map;
using std::string;
using std::vector;

namespace ge {
NamedAttrs::NamedAttrs() { named_attrs_.InitDefault(); }

NamedAttrs::NamedAttrs(const ProtoMsgOwner &owner, proto::NamedAttrs *proto_msg)
    : named_attrs_(owner, proto_msg) {}  // lint !e1744

void NamedAttrs::SetName(const std::string &name) {
  auto proto_msg = named_attrs_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_name(name);
  }
}

string NamedAttrs::GetName() const {
  auto proto_msg = named_attrs_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return proto_msg->name();
  }
  return string();
}

GeAttrValue NamedAttrs::GetItem(const string &key) const {
  GeAttrValue value;
  (void)GetAttr(key, value);
  return value;
}

ProtoAttrMapHelper NamedAttrs::MutableAttrMap() {
  auto proto_msg = named_attrs_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return ProtoAttrMapHelper(named_attrs_.GetProtoOwner(), proto_msg->mutable_attr());
  }
  return ProtoAttrMapHelper(named_attrs_.GetProtoOwner(), nullptr);
}

ConstProtoAttrMapHelper NamedAttrs::GetAttrMap() const {
  auto proto_msg = named_attrs_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return ConstProtoAttrMapHelper(named_attrs_.GetProtoOwner(), &proto_msg->attr());
  }
  return ConstProtoAttrMapHelper(named_attrs_.GetProtoOwner(), nullptr);
}

class GeAttrValueImp {
 public:
  static map<proto::AttrDef::ValueCase, GeAttrValue::ValueType> attr_val_one_type_map_;
  static map<proto::AttrDef_ListValue_ListValueType, GeAttrValue::ValueType> attr_val_list_type_map_;

  static bool SetValue(proto::AttrDef &attr_def, GeAttrValue::INT val);
  static bool SetValue(proto::AttrDef &attr_def, GeAttrValue::FLOAT val);
  static bool SetValue(proto::AttrDef &attr_def, GeAttrValue::BOOL val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::STR &val);
  static bool SetValue(proto::AttrDef &attr_def, const ConstGeTensorPtr &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeTensor &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::TENSOR_DESC &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::BYTES &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::NAMED_ATTRS &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::GRAPH &val);
  static bool SetValue(proto::AttrDef &attr_def, const vector<int64_t> &val);
  static bool SetValue(proto::AttrDef &attr_def, const vector<int32_t> &val);
  static bool SetValue(proto::AttrDef &attr_def, const vector<uint32_t> &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::LIST_FLOAT &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::LIST_BOOL &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::LIST_STR &val);
  static bool SetValue(proto::AttrDef &proto_attr_val, const vector<GeTensorPtr> &value);
  static bool SetValue(proto::AttrDef &proto_attr_val, const vector<ConstGeTensorPtr> &value);
  static bool SetValue(proto::AttrDef &attr_def, const vector<GeTensor> &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::LIST_TENSOR_DESC &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::LIST_BYTES &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::LIST_NAMED_ATTRS &val);
  static bool SetValue(proto::AttrDef &attr_def, const GeAttrValue::LIST_GRAPH &val);

  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, GeAttrValue::INT &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, GeAttrValue::FLOAT &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, GeAttrValue::BOOL &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, GeAttrValue::STR &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, GeAttrValue::TENSOR &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, GeTensor &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::TENSOR_DESC &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, GeAttrValue::BYTES &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::NAMED_ATTRS &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, GeAttrValue::GRAPH &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::LIST_INT &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::LIST_FLOAT &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::LIST_BOOL &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::LIST_STR &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::LIST_TENSOR &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, vector<GeTensor> &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::LIST_TENSOR_DESC &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::LIST_BYTES &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::LIST_NAMED_ATTRS &val);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       GeAttrValue::LIST_GRAPH &val);
  // Value will be moved
  static bool SetZeroCopyBytes(proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, Buffer &&buffer);
  static bool GetZeroCopyBytes(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, Buffer &buffer);
  // Value will be moved
  static bool SetZeroCopyListBytes(proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                                   vector<Buffer> &list_buffer);
  static bool GetZeroCopyListBytes(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                                   vector<Buffer> &list_buffer);

  static bool SetValue(proto::AttrDef &attr_def, const vector<vector<int64_t>> &value);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       vector<vector<int64_t>> &value);
  static bool SetValue(proto::AttrDef &attr_def, const vector<ge::DataType> &value);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner,
                       vector<ge::DataType> &value);

  static bool SetValue(proto::AttrDef &attr_def, const ge::DataType &value);
  static bool GetValue(const proto::AttrDef &attr_def, const ProtoMsgOwner &proto_msg_owner, ge::DataType &value);
};

map<proto::AttrDef::ValueCase, GeAttrValue::ValueType> GeAttrValueImp::attr_val_one_type_map_ = {
  {proto::AttrDef::kI, GeAttrValue::VT_INT},
  {proto::AttrDef::kF, GeAttrValue::VT_FLOAT},
  {proto::AttrDef::kB, GeAttrValue::VT_BOOL},
  {proto::AttrDef::kS, GeAttrValue::VT_STRING},
  {proto::AttrDef::kT, GeAttrValue::VT_TENSOR},
  {proto::AttrDef::kTd, GeAttrValue::VT_TENSOR_DESC},
  {proto::AttrDef::kG, GeAttrValue::VT_GRAPH},
  {proto::AttrDef::kBt, GeAttrValue::VT_BYTES},
  {proto::AttrDef::kFunc, GeAttrValue::VT_NAMED_ATTRS},
  {proto::AttrDef::kListListInt, GeAttrValue::VT_LIST_LIST_INT},
  {proto::AttrDef::kDt, GeAttrValue::VT_DATA_TYPE},
};
map<proto::AttrDef_ListValue_ListValueType, GeAttrValue::ValueType> GeAttrValueImp::attr_val_list_type_map_ = {
  {proto::AttrDef_ListValue_ListValueType_VT_LIST_INT, GeAttrValue::VT_LIST_INT},
  {proto::AttrDef_ListValue_ListValueType_VT_LIST_FLOAT, GeAttrValue::VT_LIST_FLOAT},
  {proto::AttrDef_ListValue_ListValueType_VT_LIST_BOOL, GeAttrValue::VT_LIST_BOOL},
  {proto::AttrDef_ListValue_ListValueType_VT_LIST_STRING, GeAttrValue::VT_LIST_STRING},
  {proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR, GeAttrValue::VT_LIST_TENSOR},
  {proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR_DESC, GeAttrValue::VT_LIST_TENSOR_DESC},
  {proto::AttrDef_ListValue_ListValueType_VT_LIST_GRAPH, GeAttrValue::VT_LIST_GRAPH},
  {proto::AttrDef_ListValue_ListValueType_VT_LIST_BYTES, GeAttrValue::VT_LIST_BYTES},
  {proto::AttrDef_ListValue_ListValueType_VT_LIST_NAMED_ATTRS, GeAttrValue::VT_LIST_NAMED_ATTRS},
  {proto::AttrDef_ListValue_ListValueType_VT_LIST_DATA_TYPE, GeAttrValue::VT_LIST_DATA_TYPE},
};

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeAttrValue::GeAttrValue() { value_.InitDefault(); }

GeAttrValue::GeAttrValue(const ProtoMsgOwner &proto_owner, ge::proto::AttrDef *val) : value_(proto_owner, val) {}

GeAttrValue::ValueType GeAttrValue::GetValueType() const {
  auto proto_msg = value_.GetProtoMsg();
  if (proto_msg != nullptr) {
    auto val_case = proto_msg->value_case();
    if (val_case != proto::AttrDef::kList) {
      auto it = GeAttrValueImp::attr_val_one_type_map_.find(val_case);
      if (it != GeAttrValueImp::attr_val_one_type_map_.end()) {
        return it->second;
      }
    } else {
      auto it = GeAttrValueImp::attr_val_list_type_map_.find(proto_msg->list().val_type());
      if (it != GeAttrValueImp::attr_val_list_type_map_.end()) {
        return it->second;
      }
    }
  }
  return GeAttrValue::VT_NONE;
}

bool GeAttrValue::IsEmpty() const { return GetValueType() == VT_NONE; }

GeAttrValue GeAttrValue::Copy() const {
  GeAttrValue valueRet;
  auto proto_msg = value_.GetProtoMsg();
  auto proto_msg_ret = valueRet.value_.GetProtoMsg();
  if (proto_msg != nullptr && proto_msg_ret != nullptr) {
    *proto_msg_ret = *proto_msg;
  }
  return valueRet;
}

#define ATTR_VALUE_SET_GET_IMP(type)                                           \
  graphStatus GeAttrValue::SetValue(const type &val) {                         \
    auto proto_msg = value_.GetProtoMsg();                                     \
    if (proto_msg) {                                                           \
      if (GeAttrValueImp::SetValue(*proto_msg, val)) {                         \
        return GRAPH_SUCCESS;                                                  \
      }                                                                        \
    }                                                                          \
    return GRAPH_FAILED;                                                       \
  }                                                                            \
                                                                               \
  graphStatus GeAttrValue::GetValue(type &val) const {                         \
    auto proto_msg = value_.GetProtoMsg();                                     \
    if (proto_msg) {                                                           \
      if (GeAttrValueImp::GetValue(*proto_msg, value_.GetProtoOwner(), val)) { \
        return GRAPH_SUCCESS;                                                  \
      }                                                                        \
    }                                                                          \
    return GRAPH_FAILED;                                                       \
  }

ATTR_VALUE_SET_GET_IMP(GeAttrValue::STR)
ATTR_VALUE_SET_GET_IMP(vector<GeAttrValue::STR>)
ATTR_VALUE_SET_GET_IMP(GeAttrValue::INT)
ATTR_VALUE_SET_GET_IMP(vector<GeAttrValue::INT>)
ATTR_VALUE_SET_GET_IMP(GeAttrValue::FLOAT)  // lint !e524
ATTR_VALUE_SET_GET_IMP(vector<GeAttrValue::FLOAT>)
ATTR_VALUE_SET_GET_IMP(GeAttrValue::BOOL)
ATTR_VALUE_SET_GET_IMP(vector<GeAttrValue::BOOL>)
ATTR_VALUE_SET_GET_IMP(GeAttrValue::TENSOR_DESC)
ATTR_VALUE_SET_GET_IMP(vector<GeAttrValue::TENSOR_DESC>)
ATTR_VALUE_SET_GET_IMP(GeAttrValue::TENSOR)
ATTR_VALUE_SET_GET_IMP(vector<GeAttrValue::TENSOR>)
ATTR_VALUE_SET_GET_IMP(GeAttrValue::GRAPH)
ATTR_VALUE_SET_GET_IMP(vector<GeAttrValue::GRAPH>)
ATTR_VALUE_SET_GET_IMP(GeAttrValue::BYTES)
ATTR_VALUE_SET_GET_IMP(vector<GeAttrValue::BYTES>)
ATTR_VALUE_SET_GET_IMP(GeAttrValue::NAMED_ATTRS)
ATTR_VALUE_SET_GET_IMP(vector<GeAttrValue::NAMED_ATTRS>)
/*lint -e665*/
ATTR_VALUE_SET_GET_IMP(vector<vector<int64_t>>)
/*lint +e665*/
ATTR_VALUE_SET_GET_IMP(vector<DataType>)        // lint !e665
ATTR_VALUE_SET_GET_IMP(GeAttrValue::DATA_TYPE)  // lint !e665

#undef ATTR_VALUE_SET_GET_IMP

graphStatus GeAttrValue::MutableTensor(GeTensorPtr &tensor) { return GetValue(tensor); }

graphStatus GeAttrValue::MutableListTensor(vector<GeTensorPtr> &list_tensor) { return GetValue(list_tensor); }

class AttrUtilsHelper {
 public:
  inline static bool GetValueCheckType(const proto::AttrDef &attr_def, proto::AttrDef::ValueCase proto_case) {
    if (attr_def.value_case() != proto_case) {
      GELOGW("Check Type Failed, proto case type %u, expected %u", attr_def.value_case(), proto_case);
      return false;
    }
    return true;
  }

  inline static bool GetValueCheckListType(
    const proto::AttrDef &attr_def, proto::AttrDef_ListValue_ListValueType proto_list_case,
    const std::function<bool(const proto::AttrDef &proto_attr_val)> item_check_fun) {
    if (attr_def.value_case() != proto::AttrDef::kList) {
      GELOGW("Check ListType Failed, value_case %u", attr_def.value_case());
      return false;
    }
    auto &list = attr_def.list();
    if (list.val_type() == proto::AttrDef_ListValue_ListValueType_VT_LIST_NONE) {
      return item_check_fun(attr_def);
    }
    if (list.val_type() != proto_list_case) {
      GELOGW("Check ListType Failed, val_type %u, expected %u", list.val_type(), proto_list_case);
      return false;
    }
    return true;
  }

  inline static bool SetValueCheckType(proto::AttrDef &attr_def, proto::AttrDef::ValueCase proto_case) {
    if (attr_def.value_case() != proto::AttrDef::VALUE_NOT_SET && attr_def.value_case() != proto_case) {
      GELOGW("Check Type Failed, proto case type %u, expected %u", attr_def.value_case(), proto_case);
      return false;
    }
    return true;
  }

  inline static bool SetValueCheckAndSetListType(proto::AttrDef &attr_def,
                                                 proto::AttrDef_ListValue_ListValueType proto_list_case) {
    if (attr_def.value_case() != proto::AttrDef::VALUE_NOT_SET && attr_def.value_case() != proto::AttrDef::kList) {
      GELOGW("AttrUtils::Check Type Failed, value_case %u", attr_def.value_case());
      return false;
    }
    auto list = attr_def.mutable_list();
    if (list == nullptr) {
      GELOGE(GRAPH_FAILED, "list is nullptr");
      return false;
    }
    if (list->val_type() != proto::AttrDef_ListValue_ListValueType_VT_LIST_NONE &&
        list->val_type() != proto_list_case) {
      GELOGW("AttrUtils::Check ListType Type Failed, val_type %d, expected %d", static_cast<int>(list->val_type()),
             static_cast<int>(proto_list_case));
      return false;
    }
    list->set_val_type(proto_list_case);
    return true;
  }

  static bool GetAttrMapItem(const AttrHolder *obj, const string &name, const proto::AttrDef *&attr_def) {
    if (obj == nullptr) {
      GELOGE(FAILED, "%s obj is nullptr", name.c_str());
      return false;
    }
    auto attr_map = obj->GetAttrMap().GetProtoMsg();
    if (attr_map == nullptr) {
      GELOGE(FAILED, "%s attr map is nullptr", name.c_str());
      return false;
    }
    auto it = attr_map->find(name);
    if (it == attr_map->end()) {
      return false;
    }
    attr_def = &it->second;
    return true;
  }

  inline static bool MutableAttrMapItem(AttrHolder *obj, const string &name, proto::AttrDef *&attr_def) {
    if (obj == nullptr) {
      GELOGE(FAILED, " %s obj is nullptr", name.c_str());
      return false;
    }
    auto attr_map = obj->MutableAttrMap().GetProtoMsg();
    if (attr_map == nullptr) {
      GELOGE(FAILED, "%s attr map is nullptr", name.c_str());
      return false;
    }
    // Get or add
    attr_def = &((*attr_map)[name]);
    return true;
  }
};

#define ATTR_VALUE_IMP_SET_ONE(ValType, proto_case, protoItem)                             \
  bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, ValType value) {           \
    if (!AttrUtilsHelper::SetValueCheckType(proto_attr_val, proto::AttrDef::proto_case)) { \
      return false;                                                                        \
    }                                                                                      \
    proto_attr_val.set_##protoItem(value);                                                 \
    return true;                                                                           \
  }

#define ATTR_VALUE_IMP_SET_LIST(ValType, proto_list_case, protoItem)                                               \
  bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, ValType value) {                                   \
    if (!AttrUtilsHelper::SetValueCheckAndSetListType(proto_attr_val,                                              \
                                                      proto::AttrDef_ListValue_ListValueType_##proto_list_case)) { \
      return false;                                                                                                \
    }                                                                                                              \
    auto list = proto_attr_val.mutable_list();                                                                     \
    list->clear_##protoItem();                                                                                     \
    for (const auto &item : value) {                                                                               \
      list->add_##protoItem(item);                                                                                 \
    }                                                                                                              \
    return true;                                                                                                   \
  }

ATTR_VALUE_IMP_SET_ONE(int64_t, kI, i)
ATTR_VALUE_IMP_SET_ONE(float, kF, f)
ATTR_VALUE_IMP_SET_ONE(const string &, kS, s)
ATTR_VALUE_IMP_SET_ONE(bool, kB, b)

ATTR_VALUE_IMP_SET_LIST(const vector<int64_t> &, VT_LIST_INT, i)
ATTR_VALUE_IMP_SET_LIST(const vector<int32_t> &, VT_LIST_INT, i)
ATTR_VALUE_IMP_SET_LIST(const vector<uint32_t> &, VT_LIST_INT, i)
ATTR_VALUE_IMP_SET_LIST(const vector<float> &, VT_LIST_FLOAT, f)
ATTR_VALUE_IMP_SET_LIST(const vector<string> &, VT_LIST_STRING, s)
ATTR_VALUE_IMP_SET_LIST(const vector<bool> &, VT_LIST_BOOL, b)

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const GeTensorDesc &value) {
  if (!AttrUtilsHelper::SetValueCheckType(proto_attr_val, proto::AttrDef::kTd)) {
    return false;
  }
  auto proto_msg = value.tensor_descriptor_.GetProtoMsg();
  if (proto_msg == nullptr) {
    return false;
  }
  *proto_attr_val.mutable_td() = *proto_msg;
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const vector<GeTensorDesc> &value) {
  if (!AttrUtilsHelper::SetValueCheckAndSetListType(proto_attr_val,
                                                    proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR_DESC)) {
    return false;
  }
  auto list = proto_attr_val.mutable_list();
  GE_CHECK_NOTNULL_EXEC(list, return false);
  list->clear_td();
  for (const auto &item : value) {
    auto proto_msg = item.tensor_descriptor_.GetProtoMsg();
    if (proto_msg == nullptr) {
      proto_attr_val.clear_list();
      return false;
    }
    *list->add_td() = *proto_msg;
  }
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const ConstGeTensorPtr &value) {
  if (value) {
    return SetValue(proto_attr_val, *value);
  } else {
    return SetValue(proto_attr_val, GeTensor());
  }
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const GeTensor &val) {
  if (!AttrUtilsHelper::SetValueCheckType(proto_attr_val, proto::AttrDef::kT)) {
    return false;
  }
  auto proto_msg = val.tensor_def_.GetProtoMsg();
  if (proto_msg == nullptr) {
    GELOGE(FAILED, "Proto msg is nullptr");
    return false;
  }
  *proto_attr_val.mutable_t() = *proto_msg;
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const vector<GeTensorPtr> &value) {
  vector<ConstGeTensorPtr> constList(value.size());
  std::copy(value.begin(), value.end(), constList.begin());
  return SetValue(proto_attr_val, constList);
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const vector<ConstGeTensorPtr> &value) {
  if (!AttrUtilsHelper::SetValueCheckAndSetListType(proto_attr_val,
                                                    proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR)) {
    return false;
  }
  auto list = proto_attr_val.mutable_list();
  GE_CHECK_NOTNULL_EXEC(list, return false);
  list->clear_t();
  for (const auto &item : value) {
    if (item == nullptr) {
      GELOGE(GRAPH_FAILED, "AttrUtils::SetListTensor item is nullptr");
      proto_attr_val.clear_list();
      return false;
    }
    auto proto_msg = item->tensor_def_.GetProtoMsg();
    if (proto_msg == nullptr) {
      GELOGE(FAILED, "Proto msg is nullptr");
      proto_attr_val.clear_list();
      return false;
    }
    *list->add_t() = *proto_msg;
  }
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const vector<GeTensor> &value) {
  if (!AttrUtilsHelper::SetValueCheckAndSetListType(proto_attr_val,
                                                    proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR)) {
    return false;
  }
  auto list = proto_attr_val.mutable_list();
  GE_CHECK_NOTNULL_EXEC(list, return false);
  list->clear_t();
  for (const auto &item : value) {
    auto proto_msg = item.tensor_def_.GetProtoMsg();
    if (proto_msg == nullptr) {
      GELOGE(FAILED, "Proto msg is nullptr");
      proto_attr_val.clear_list();
      return false;
    }
    *list->add_t() = *proto_msg;
  }
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const GeAttrValue::BYTES &value) {
  if (!AttrUtilsHelper::SetValueCheckType(proto_attr_val, proto::AttrDef::kBt)) {
    return false;
  }
  size_t val_size = value.GetSize();
  proto_attr_val.set_bt(value.GetData(), val_size);
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const vector<GeAttrValue::BYTES> &value) {
  if (!AttrUtilsHelper::SetValueCheckAndSetListType(proto_attr_val,
                                                    proto::AttrDef_ListValue_ListValueType_VT_LIST_BYTES)) {
    return false;
  }
  auto list = proto_attr_val.mutable_list();
  GE_CHECK_NOTNULL_EXEC(list, return false);
  list->clear_bt();
  for (const auto &item : value) {
    list->add_bt(item.GetData(), item.GetSize());
  }
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const GeAttrValue::NAMED_ATTRS &value) {
  if (!AttrUtilsHelper::SetValueCheckType(proto_attr_val, proto::AttrDef::kFunc)) {
    return false;
  }
  auto proto_msg = value.named_attrs_.GetProtoMsg();
  if (proto_msg == nullptr) {
    GELOGE(FAILED, "Proto msg is nullptr");
    return false;
  }
  *proto_attr_val.mutable_func() = *proto_msg;
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const vector<GeAttrValue::NAMED_ATTRS> &value) {
  if (!AttrUtilsHelper::SetValueCheckAndSetListType(proto_attr_val,
                                                    proto::AttrDef_ListValue_ListValueType_VT_LIST_NAMED_ATTRS)) {
    return false;
  }
  auto list = proto_attr_val.mutable_list();
  GE_CHECK_NOTNULL_EXEC(list, return false);
  list->clear_na();
  for (const auto &item : value) {
    auto proto_msg = item.named_attrs_.GetProtoMsg();
    if (proto_msg == nullptr) {
      proto_attr_val.clear_list();
      return false;
    }
    *list->add_na() = *proto_msg;
  }
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const ge::ComputeGraphPtr &value) {
  if (!AttrUtilsHelper::SetValueCheckType(proto_attr_val, proto::AttrDef::kG)) {
    return false;
  }
  ModelSerializeImp imp;
  if (!imp.SerializeGraph(value, proto_attr_val.mutable_g())) {
    GELOGE(GRAPH_FAILED, "AttrUtils::SetGraph SerializeGraph Failed");
    proto_attr_val.clear_g();
    return false;
  }
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const vector<ge::ComputeGraphPtr> &value) {
  if (!AttrUtilsHelper::SetValueCheckAndSetListType(proto_attr_val,
                                                    proto::AttrDef_ListValue_ListValueType_VT_LIST_GRAPH)) {
    return false;
  }
  auto list = proto_attr_val.mutable_list();
  GE_CHECK_NOTNULL_EXEC(list, return false);
  list->clear_g();

  ModelSerializeImp imp;
  for (const auto &item : value) {
    if (!imp.SerializeGraph(item, list->add_g())) {
      GELOGE(GRAPH_FAILED, "AttrUtils::SetListGraph SerializeGraph");
      proto_attr_val.clear_list();
      return false;
    }
  }
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const vector<vector<int64_t>> &value) {
  if (!AttrUtilsHelper::SetValueCheckType(proto_attr_val, proto::AttrDef::kListListInt)) {
    return false;
  }
  proto_attr_val.clear_list_list_int();
  auto list_list_int = proto_attr_val.mutable_list_list_int();
  GE_CHECK_NOTNULL_EXEC(list_list_int, return false);
  for (auto &list_int : value) {
    auto list_item = list_list_int->add_list_list_i();
    GE_CHECK_NOTNULL_EXEC(list_item, return false);
    for (auto &int_item : list_int) {
      list_item->add_list_i(int_item);
    }
  }
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const vector<ge::DataType> &value) {
  if (!AttrUtilsHelper::SetValueCheckAndSetListType(proto_attr_val,
                                                    proto::AttrDef_ListValue_ListValueType_VT_LIST_DATA_TYPE)) {
    return false;
  }
  auto list = proto_attr_val.mutable_list();
  GE_CHECK_NOTNULL_EXEC(list, return false);
  list->clear_dt();
  for (const auto &item : value) {
    list->add_dt(static_cast<int64_t>(item));
  }
  return true;
}

bool GeAttrValueImp::SetValue(proto::AttrDef &proto_attr_val, const ge::DataType &value) {
  if (!AttrUtilsHelper::SetValueCheckType(proto_attr_val, proto::AttrDef::kDt)) {
    return false;
  }
  proto_attr_val.set_dt(static_cast<int64_t>(value));

  return true;
}

#define ATTR_VALUE_IMP_GET_ONE(ValType, proto_case, protoItem)                                                \
  bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &, ValType value) { \
    if (!AttrUtilsHelper::GetValueCheckType(proto_attr_val, proto::AttrDef::proto_case)) {                    \
      return false;                                                                                           \
    }                                                                                                         \
    value = proto_attr_val.protoItem();                                                                       \
    return true;                                                                                              \
  }

#define ListValueItemCheck(protoItem) \
  [](const proto::AttrDef &proto_attr_val) { return proto_attr_val.list().protoItem##_size() > 0; }

#define ATTR_VALUE_IMP_GET_LIST(ValType, proto_list_case, protoItem)                                                   \
  bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &, vector<ValType> &value) { \
    value.clear();                                                                                                     \
    if (!AttrUtilsHelper::GetValueCheckListType(                                                                       \
          proto_attr_val, proto::AttrDef_ListValue_ListValueType_##proto_list_case, ListValueItemCheck(protoItem))) {  \
      return false;                                                                                                    \
    }                                                                                                                  \
    auto &list = proto_attr_val.list();                                                                                \
    for (const auto &item : list.protoItem()) {                                                                        \
      value.push_back(item);                                                                                           \
    }                                                                                                                  \
    return true;                                                                                                       \
  }

ATTR_VALUE_IMP_GET_ONE(int64_t &, kI, i)
ATTR_VALUE_IMP_GET_ONE(float &, kF, f)
ATTR_VALUE_IMP_GET_ONE(string &, kS, s)
ATTR_VALUE_IMP_GET_ONE(bool &, kB, b)

ATTR_VALUE_IMP_GET_LIST(int64_t, VT_LIST_INT, i)
ATTR_VALUE_IMP_GET_LIST(float, VT_LIST_FLOAT, f)
ATTR_VALUE_IMP_GET_LIST(string, VT_LIST_STRING, s)
ATTR_VALUE_IMP_GET_LIST(bool, VT_LIST_BOOL, b)

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &, GeTensorDesc &value) {
  if (!AttrUtilsHelper::GetValueCheckType(proto_attr_val, proto::AttrDef::kTd)) {
    return false;
  }
  auto proto_msg = value.tensor_descriptor_.GetProtoMsg();
  if (proto_msg == nullptr) {
    return false;
  }
  *proto_msg = proto_attr_val.td();
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &,
                              vector<GeTensorDesc> &value) {
  if (!AttrUtilsHelper::GetValueCheckListType(
        proto_attr_val, proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR_DESC, ListValueItemCheck(td))) {
    return false;
  }
  auto &list = proto_attr_val.list();
  for (const auto &item : list.td()) {
    value.emplace_back(GeTensorDesc());
    auto proto_msg = value.back().tensor_descriptor_.GetProtoMsg();
    if (proto_msg == nullptr) {
      return false;
    }
    *proto_msg = item;
  }
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &proto_owner,
                              GeTensorPtr &value) {
  if (!AttrUtilsHelper::GetValueCheckType(proto_attr_val, proto::AttrDef::kT)) {
    return false;
  }
  value = std::shared_ptr<GeTensor>(new (std::nothrow)
                                      GeTensor(proto_owner, const_cast<proto::AttrDef &>(proto_attr_val).mutable_t()));
  GE_CHK_BOOL_RET_STATUS(value != nullptr, false, "value is nullptr");
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &proto_owner,
                              vector<GeTensorPtr> &value) {
  value.clear();
  if (!AttrUtilsHelper::GetValueCheckListType(proto_attr_val, proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR,
                                              ListValueItemCheck(t))) {
    return false;
  }
  auto list = const_cast<proto::AttrDef &>(proto_attr_val).mutable_list();
  GE_CHECK_NOTNULL_EXEC(list, return false);
  for (auto &item : *(list->mutable_t())) {
    std::shared_ptr<GeTensor> temp_value = std::shared_ptr<GeTensor>(new (std::nothrow) GeTensor(proto_owner, &item));
    GE_CHK_BOOL_RET_STATUS(temp_value != nullptr, false, "temp_value is nullptr");
    value.push_back(temp_value);
  }
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &, GeAttrValue::BYTES &value) {
  if (!AttrUtilsHelper::GetValueCheckType(proto_attr_val, proto::AttrDef::kBt)) {
    return false;
  }
  auto &proto_val = proto_attr_val.bt();
  GE_LOGI_IF(proto_val.size() == 0, "size res is 0.");
  value = Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(proto_val.data()), proto_val.size());
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &,
                              vector<GeAttrValue::BYTES> &value) {
  value.clear();
  if (!AttrUtilsHelper::GetValueCheckListType(proto_attr_val, proto::AttrDef_ListValue_ListValueType_VT_LIST_BYTES,
                                              ListValueItemCheck(bt))) {
    return false;
  }
  auto &list = proto_attr_val.list();
  for (const auto &item : list.bt()) {
    value.push_back(Buffer::CopyFrom((const uint8_t *)item.data(), item.size()));
  }
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &,
                              GeAttrValue::NAMED_ATTRS &value) {
  if (!AttrUtilsHelper::GetValueCheckType(proto_attr_val, proto::AttrDef::kFunc)) {
    return false;
  }
  auto proto_msg = value.named_attrs_.GetProtoMsg();
  if (proto_msg == nullptr) {
    return false;
  }
  *proto_msg = proto_attr_val.func();
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &,
                              vector<GeAttrValue::NAMED_ATTRS> &value) {
  value.clear();
  if (!AttrUtilsHelper::GetValueCheckListType(
        proto_attr_val, proto::AttrDef_ListValue_ListValueType_VT_LIST_NAMED_ATTRS, ListValueItemCheck(na))) {
    return false;
  }
  auto &list = proto_attr_val.list();
  for (const auto &item : list.na()) {
    value.emplace_back(GeAttrValue::NAMED_ATTRS());
    if (value.empty()) {
      return false;
    }
    auto proto_msg = value.back().named_attrs_.GetProtoMsg();
    if (proto_msg == nullptr) {
      return false;
    }
    *proto_msg = item;
  }
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &, ComputeGraphPtr &value) {
  if (!AttrUtilsHelper::GetValueCheckType(proto_attr_val, proto::AttrDef::kG)) {
    return false;
  }
  ComputeGraphPtr graph = nullptr;
  std::shared_ptr<proto::GraphDef> graph_def;
  graph_def = ComGraphMakeShared<proto::GraphDef>(proto_attr_val.g());
  if (graph_def == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::GraphDef make shared failed");
    graph_def = nullptr;
    return false;  // lint !e665
  } else {
    ModelSerializeImp imp;
    imp.SetProtobufOwner(graph_def);
    if (!imp.UnserializeGraph(graph, *graph_def)) {
      GELOGE(GRAPH_FAILED, "UnserializeGraph Failed");
      return false;
    }  // lint !e514
    value = graph;
  }
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &,
                              vector<ComputeGraphPtr> &value) {
  value.clear();
  if (!AttrUtilsHelper::GetValueCheckListType(proto_attr_val, proto::AttrDef_ListValue_ListValueType_VT_LIST_GRAPH,
                                              ListValueItemCheck(g))) {
    return false;
  }
  auto &list = proto_attr_val.list();
  for (const auto &item : list.g()) {
    std::shared_ptr<proto::GraphDef> graph_def;
    graph_def = ComGraphMakeShared<proto::GraphDef>(item);
    if (graph_def == nullptr) {
      GELOGE(GRAPH_FAILED, "proto::GraphDef make shared failed");
      graph_def = nullptr;
      return false;  // lint !e665
    } else {
      ComputeGraphPtr graph = nullptr;
      ModelSerializeImp imp;
      imp.SetProtobufOwner(graph_def);
      if (!imp.UnserializeGraph(graph, *graph_def)) {
        GELOGE(GRAPH_FAILED, "UnserializeGraph Failed");
        return false;
      }  // lint !e514
      value.push_back(graph);
    }
  }
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &,
                              vector<vector<int64_t>> &value) {
  value.clear();
  if (!AttrUtilsHelper::GetValueCheckType(proto_attr_val, proto::AttrDef::kListListInt)) {
    return false;
  }

  auto &list_listint = proto_attr_val.list_list_int().list_list_i();
  for (auto &list_int : list_listint) {
    vector<int64_t> list_item(list_int.list_i().size());
    if (!list_int.list_i().empty()) {
      (void)std::copy(list_int.list_i().begin(), list_int.list_i().end(), list_item.begin());
    }
    value.push_back(list_item);
  }
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &,
                              vector<ge::DataType> &value) {
  if (!AttrUtilsHelper::GetValueCheckListType(proto_attr_val, proto::AttrDef_ListValue_ListValueType_VT_LIST_DATA_TYPE,
                                              ListValueItemCheck(dt))) {
    return false;
  }
  auto &list = proto_attr_val.list();
  for (const auto &item : list.dt()) {
    value.emplace_back(static_cast<ge::DataType>(item));
  }
  return true;
}

bool GeAttrValueImp::GetValue(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &, ge::DataType &value) {
  if (!AttrUtilsHelper::GetValueCheckType(proto_attr_val, proto::AttrDef::kDt)) {
    return false;
  }
  value = static_cast<ge::DataType>(proto_attr_val.dt());
  return true;
}

GE_FUNC_HOST_VISIBILITY bool GeAttrValueImp::SetZeroCopyBytes(proto::AttrDef &proto_attr_val, const ProtoMsgOwner &,
                                                              Buffer &&buffer) {
  if (!AttrUtilsHelper::SetValueCheckType(proto_attr_val, proto::AttrDef::kBt)) {
    return false;
  }
  auto proto_msg = buffer.data_.GetProtoMsg();
  if (proto_msg == nullptr) {
    return false;
  }
  proto_attr_val.set_bt(std::move(*proto_msg->mutable_bt()));
  return true;
}

bool GeAttrValueImp::GetZeroCopyBytes(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &proto_owner,
                                      Buffer &buffer) {
  if (!AttrUtilsHelper::GetValueCheckType(proto_attr_val, proto::AttrDef::kBt)) {
    return false;
  }
  buffer = Buffer(proto_owner, &const_cast<proto::AttrDef &>(proto_attr_val));
  return true;
}

bool GeAttrValueImp::SetZeroCopyListBytes(proto::AttrDef &proto_attr_val, const ProtoMsgOwner &,
                                          vector<Buffer> &list_buffer) {
  if (!AttrUtilsHelper::SetValueCheckAndSetListType(proto_attr_val,
                                                    proto::AttrDef_ListValue_ListValueType_VT_LIST_BYTES)) {
    return false;
  }
  auto list = proto_attr_val.mutable_list();
  GE_CHECK_NOTNULL_EXEC(list, return false);
  list->clear_bt();
  for (auto &item : list_buffer) {
    auto proto_msg = item.data_.GetProtoMsg();
    if (proto_msg == nullptr) {
      return false;
    }
    list->add_bt(std::move(*proto_msg->mutable_bt()));
  }
  return true;
}

bool GeAttrValueImp::GetZeroCopyListBytes(const proto::AttrDef &proto_attr_val, const ProtoMsgOwner &proto_owner,
                                          vector<Buffer> &list_buffer) {
  list_buffer.clear();
  if (!AttrUtilsHelper::GetValueCheckListType(proto_attr_val, proto::AttrDef_ListValue_ListValueType_VT_LIST_BYTES,
                                              ListValueItemCheck(bt))) {
    return false;
  }
  auto list = const_cast<proto::AttrDef &>(proto_attr_val).mutable_list();
  GE_CHECK_NOTNULL_EXEC(list, return false);
  for (auto &item : *(list->mutable_bt())) {
    list_buffer.emplace_back(Buffer(proto_owner, &item));
  }
  return true;
}

bool AttrUtils::HasAttr(ConstAttrHolderAdapter &&obj, const string &name) {
  if (!obj) {
    return false;
  }
  return obj->HasAttr(name);
}

#define ATTR_UTILS_SET_IMP(FuncName, Type)                                                                    \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::Set##FuncName(                               \
    AttrHolderAdapter &&obj, const string &name, const Type &value) {                                         \
    proto::AttrDef *proto_attr_val = nullptr;                                                                 \
    if (!AttrUtilsHelper::MutableAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) { \
      return false;                                                                                           \
    }                                                                                                         \
    if (!GeAttrValueImp::SetValue(*proto_attr_val, value)) {                                                  \
      GELOGW("Set" #FuncName " failed key %s", name.c_str());                                                 \
      return false;                                                                                           \
    }                                                                                                         \
    return true;                                                                                              \
  }

#define ATTR_UTILS_GET_IMP(FuncName, Type)                                                                        \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::Get##FuncName(ConstAttrHolderAdapter &&obj,      \
                                                                               const string &name, Type &value) { \
    const proto::AttrDef *proto_attr_val = nullptr;                                                               \
    if (!AttrUtilsHelper::GetAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) {         \
      return false;                                                                                               \
    }                                                                                                             \
    if (!GeAttrValueImp::GetValue(*proto_attr_val, obj->GetAttrMap().GetProtoOwner(), value)) {                   \
      GELOGW("Get" #FuncName " failed key %s", name.c_str());                                                     \
      return false;                                                                                               \
    }                                                                                                             \
    return true;                                                                                                  \
  }

#define ATTR_UTILS_SET_GET_IMP(FuncName, Type) \
  ATTR_UTILS_SET_IMP(FuncName, Type)           \
  ATTR_UTILS_GET_IMP(FuncName, Type)

ATTR_UTILS_SET_GET_IMP(Int, int64_t)
ATTR_UTILS_SET_GET_IMP(Float, float)
ATTR_UTILS_SET_GET_IMP(Bool, bool)
ATTR_UTILS_SET_GET_IMP(Str, string)
ATTR_UTILS_SET_GET_IMP(TensorDesc, GeTensorDesc)
ATTR_UTILS_SET_IMP(Tensor, GeTensorPtr)
ATTR_UTILS_SET_IMP(Tensor, ConstGeTensorPtr)
ATTR_UTILS_SET_IMP(Tensor, GeTensor)
ATTR_UTILS_SET_GET_IMP(NamedAttrs, GeAttrValue::NAMED_ATTRS)
ATTR_UTILS_SET_GET_IMP(Bytes, Buffer)
ATTR_UTILS_SET_GET_IMP(Graph, ComputeGraphPtr)
/*lint -e665*/
ATTR_UTILS_SET_GET_IMP(ListListInt, vector<vector<int64_t>>)
/*lint +e665*/

ATTR_UTILS_SET_GET_IMP(ListInt, vector<int64_t>)
ATTR_UTILS_SET_IMP(ListInt, vector<int32_t>)
ATTR_UTILS_SET_IMP(ListInt, vector<uint32_t>)
ATTR_UTILS_SET_GET_IMP(ListFloat, vector<float>)
ATTR_UTILS_SET_GET_IMP(ListBool, vector<bool>)
ATTR_UTILS_SET_GET_IMP(ListStr, vector<string>)
ATTR_UTILS_SET_GET_IMP(ListTensorDesc, vector<GeTensorDesc>)
ATTR_UTILS_SET_IMP(ListTensor, vector<GeTensorPtr>)
ATTR_UTILS_SET_IMP(ListTensor, vector<ConstGeTensorPtr>)
ATTR_UTILS_SET_IMP(ListTensor, vector<GeTensor>)
ATTR_UTILS_SET_GET_IMP(ListNamedAttrs, vector<GeAttrValue::NAMED_ATTRS>)
ATTR_UTILS_SET_GET_IMP(ListBytes, vector<Buffer>)
ATTR_UTILS_SET_GET_IMP(ListGraph, vector<ComputeGraphPtr>)
ATTR_UTILS_SET_GET_IMP(ListDataType, vector<ge::DataType>)  // lint !e665
ATTR_UTILS_SET_GET_IMP(DataType, ge::DataType)              // lint !e665

bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj, const string &name,
                              std::initializer_list<ConstGeTensorPtr> &&value) {
  return SetListTensor(std::move(obj), name, vector<ConstGeTensorPtr>(value));
}

bool AttrUtils::GetTensor(ConstAttrHolderAdapter &&obj, const string &name, ConstGeTensorPtr &value) {
  const proto::AttrDef *proto_attr_val = nullptr;
  if (!AttrUtilsHelper::GetAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) {
    return false;
  }
  GeTensorPtr tensor;
  if (!GeAttrValueImp::GetValue(*proto_attr_val, obj->GetAttrMap().GetProtoOwner(), tensor)) {
    return false;
  }
  value = tensor;
  return true;
}

bool AttrUtils::GetListTensor(ConstAttrHolderAdapter &&obj, const string &name, vector<ConstGeTensorPtr> &value) {
  value.clear();
  const proto::AttrDef *proto_attr_val = nullptr;
  if (!AttrUtilsHelper::GetAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) {
    return false;
  }
  vector<GeTensorPtr> tensor;
  if (!GeAttrValueImp::GetValue(*proto_attr_val, obj->GetAttrMap().GetProtoOwner(), tensor)) {
    return false;
  }
  value.insert(value.begin(), tensor.begin(), tensor.end());
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::MutableTensor(AttrHolderAdapter &&obj,
                                                                             const string &name, GeTensorPtr &value) {
  const proto::AttrDef *proto_attr_val = nullptr;
  if (!AttrUtilsHelper::GetAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) {
    return false;
  }
  return GeAttrValueImp::GetValue(*proto_attr_val, obj->GetAttrMap().GetProtoOwner(), value);
}

bool AttrUtils::MutableListTensor(AttrHolderAdapter &&obj, const string &name, vector<GeTensorPtr> &value) {
  value.clear();
  const proto::AttrDef *proto_attr_val = nullptr;
  if (!AttrUtilsHelper::GetAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) {
    return false;
  }
  return GeAttrValueImp::GetValue(*proto_attr_val, obj->GetAttrMap().GetProtoOwner(), value);
}

bool AttrUtils::SetListInt(AttrHolderAdapter &&obj, const string &name, std::initializer_list<int64_t> &&value) {
  proto::AttrDef *proto_attr_val = nullptr;
  if (!AttrUtilsHelper::MutableAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) {
    return false;
  }
  return GeAttrValueImp::SetValue(*proto_attr_val, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj, const string &name,
                                                                      int32_t &value) {
  int64_t int64_val = 0;
  if (!AttrUtils::GetInt(std::move(obj), name, int64_val)) {
    return false;
  }
  if (int64_val > INT32_MAX) {
    GELOGE(GRAPH_FAILED, "%ld int64_t value cannot cast to int32_t", int64_val);
    return false;
  }
  value = static_cast<int32_t>(int64_val);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj, const string &name,
                                                                      uint32_t &value) {
  int64_t int64_val = 0;
  if (!AttrUtils::GetInt(std::move(obj), name, int64_val)) {
    return false;
  }
  if (int64_val > UINT32_MAX) {
    GELOGE(GRAPH_FAILED, "%ld int64_t value cannot cast to uint32_t", int64_val);
    return false;
  }
  value = static_cast<uint32_t>(int64_val);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj,
                                                                          const string &name, vector<int32_t> &value) {
  value.clear();
  vector<int64_t> int64_list;
  if (!GetListInt(std::move(obj), name, int64_list)) {
    return false;
  }

  for (size_t i = 0; i < int64_list.size(); ++i) {
    if (int64_list[i] > INT32_MAX) {
      GELOGE(GRAPH_FAILED, "index %zu %ld int64_t value cannot cast to int32_t", i, int64_list[i]);
      return false;
    }
  }
  value.insert(value.begin(), int64_list.begin(), int64_list.end());
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj,
                                                                          const string &name, vector<uint32_t> &value) {
  value.clear();
  vector<int64_t> int64_list;
  if (!GetListInt(std::move(obj), name, int64_list)) {
    return false;
  }

  for (size_t i = 0; i < int64_list.size(); ++i) {
    if (int64_list[i] > UINT32_MAX) {
      GELOGE(GRAPH_FAILED, "index %zu %ld int64_t value cannot cast to uint32_t", i, int64_list[i]);
      return false;
    }
  }
  value.insert(value.begin(), int64_list.begin(), int64_list.end());
  return true;
}

bool AttrUtils::SetListOpDesc(AttrHolderAdapter &&obj, const string &name, const vector<ConstOpDescPtr> &value) {
  if (obj) {
    vector<Buffer> bytes_vals;
    for (auto &item : value) {
      ModelSerialize serialize;
      auto buffer = serialize.SerializeOpDesc(item);
      if (buffer.GetSize() == 0) {
        return false;
      }
      bytes_vals.push_back(buffer);
    }
    return SetZeroCopyListBytes(std::move(obj), name, bytes_vals);
  }
  return false;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListOpDesc(AttrHolderAdapter &&obj,
                                                                             const string &name,
                                                                             const vector<OpDescPtr> &value) {
  if (obj) {
    vector<Buffer> bytes_vals;
    for (auto &item : value) {
      ModelSerialize serialize;
      auto buffer = serialize.SerializeOpDesc(item);
      if (buffer.GetSize() == 0) {
        return false;
      }
      bytes_vals.push_back(buffer);
    }
    return SetZeroCopyListBytes(std::move(obj), name, bytes_vals);
  }
  return false;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListOpDesc(ConstAttrHolderAdapter &&obj,
                                                                             const string &name,
                                                                             vector<OpDescPtr> &value) {
  value.clear();

  vector<Buffer> bytes_vals;
  if (!GetZeroCopyListBytes(std::move(obj), name, bytes_vals)) {
    return false;
  }
  for (const auto &item : bytes_vals) {
    ModelSerialize serialize;
    auto op_desc = serialize.UnserializeOpDesc(item.GetData(), item.GetSize());  // lint !e732
    value.push_back(op_desc);
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetZeroCopyBytes(AttrHolderAdapter &&obj,
                                                                                const string &name, Buffer &&buffer) {
  // Value will be moved
  proto::AttrDef *proto_attr_val = nullptr;
  if (!AttrUtilsHelper::MutableAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) {
    return false;
  }
  return GeAttrValueImp::SetZeroCopyBytes(*proto_attr_val, obj->GetAttrMap().GetProtoOwner(), std::move(buffer));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetZeroCopyBytes(ConstAttrHolderAdapter &&obj,
                                                                                const string &name, Buffer &buffer) {
  const proto::AttrDef *proto_attr_val = nullptr;
  if (!AttrUtilsHelper::GetAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) {
    return false;
  }
  return GeAttrValueImp::GetZeroCopyBytes(*proto_attr_val, obj->GetAttrMap().GetProtoOwner(), buffer);
}

bool AttrUtils::SetZeroCopyListBytes(AttrHolderAdapter &&obj, const string &name, vector<Buffer> &list_buffer) {
  // Value will be moved
  proto::AttrDef *proto_attr_val = nullptr;
  if (!AttrUtilsHelper::MutableAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) {
    return false;
  }
  return GeAttrValueImp::SetZeroCopyListBytes(*proto_attr_val, obj->GetAttrMap().GetProtoOwner(), list_buffer);
}

bool AttrUtils::GetZeroCopyListBytes(ConstAttrHolderAdapter &&obj, const string &name, vector<Buffer> &list_buffer) {
  list_buffer.clear();
  const proto::AttrDef *proto_attr_val = nullptr;
  if (!AttrUtilsHelper::GetAttrMapItem(obj.get(), name, proto_attr_val) || proto_attr_val == nullptr) {
    return false;
  }
  return GeAttrValueImp::GetZeroCopyListBytes(*proto_attr_val, obj->GetAttrMap().GetProtoOwner(), list_buffer);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr AttrUtils::CloneOpDesc(const ConstOpDescPtr &org_op_desc) {
  if (org_op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "org_op_desc is null");
    return nullptr;
  }
  std::shared_ptr<proto::OpDef> op_def;
  op_def = ComGraphMakeShared<proto::OpDef>();
  if (op_def == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::OpDef make shared failed");
    return nullptr;  // lint !e665
  }
  ModelSerializeImp imp;
  (void)imp.SerializeOpDesc(org_op_desc, op_def.get());

  imp.SetProtobufOwner(op_def);
  OpDescPtr op_desc = nullptr;
  GE_CHK_BOOL_EXEC(imp.UnserializeOpDesc(op_desc, *op_def), return op_desc, "op_desc unserialize failed");
  op_desc->extAttrs_ = org_op_desc->extAttrs_;

  // This function may be called by some passes of fusion engine, in this condition, do not need these attribute
  if (!op_desc->input_name_idx_.empty()) {
    op_desc->input_name_idx_.clear();
  }
  if (!op_desc->output_name_idx_.empty()) {
    op_desc->output_name_idx_.clear();
  }
  if (!op_desc->optional_input_names_.empty()) {
    op_desc->optional_input_names_.clear();
  }

  return op_desc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr AttrUtils::CopyOpDesc(const ConstOpDescPtr &org_op_desc) {
  if (org_op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "org_op_desc is null");
    return nullptr;
  }
  std::shared_ptr<proto::OpDef> op_def = ComGraphMakeShared<proto::OpDef>();
  if (op_def == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::OpDef make shared failed");
    return nullptr;
  }
  ModelSerializeImp imp;
  (void)imp.SerializeOpDesc(org_op_desc, op_def.get());

  imp.SetProtobufOwner(op_def);
  OpDescPtr op_desc = nullptr;
  GE_CHK_BOOL_EXEC(imp.UnserializeOpDesc(op_desc, *op_def), return op_desc, "op_desc unserialize failed");

  op_desc->extAttrs_ = org_op_desc->extAttrs_;

  op_desc->input_name_idx_.insert(org_op_desc->input_name_idx_.begin(), org_op_desc->input_name_idx_.end());
  op_desc->optional_input_names_.insert(org_op_desc->optional_input_names_.begin(),
                                        org_op_desc->optional_input_names_.end());
  op_desc->output_name_idx_.insert(org_op_desc->output_name_idx_.begin(), org_op_desc->output_name_idx_.end());

  op_desc->infer_func_ = org_op_desc->infer_func_;
  op_desc->infer_format_func_ = org_op_desc->infer_format_func_;
  op_desc->verifier_func_ = org_op_desc->verifier_func_;

  return op_desc;
}
std::string AttrUtils::GetAllAttrsStr(AttrUtils::ConstAttrHolderAdapter &&obj) {
  auto holder = obj.get();
  if (holder == nullptr) {
    return "";
  }
  auto attrs_map = holder->GetAttrMap();
  if (attrs_map.GetProtoMsg() == nullptr) {
    return "";
  }

  std::map<std::string, std::string> ordered_attrs;
  for (auto &attr : *(attrs_map.GetProtoMsg())) {
    ordered_attrs[attr.first] = attr.second.SerializeAsString();
  }

  std::stringstream ss;
  for (auto &attr : ordered_attrs) {
    ss << attr.first << ":" << attr.second << ";";
  }
  return ss.str();
}
}  // namespace ge
