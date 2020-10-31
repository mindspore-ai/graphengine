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

#include "graph/op_desc.h"
#include "debug/ge_attr_define.h"
#include "debug/ge_util.h"
#include "external/graph/operator.h"
#include "framework/common/debug/ge_log.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/ge_ir.pb.h"

using std::make_pair;
using std::shared_ptr;
using std::string;
using std::vector;

/*lint -save -e521 -e681 -e732 -e737*/
namespace ge {
const std::string ATTR_NAME_ID = "id";

const std::string ATTR_NAME_STREAM_ID = "stream_id";

const std::string ATTR_NAME_INPUT_NAME = "input_name";

const std::string ATTR_NAME_SRC_NAME = "src_name";

const std::string ATTR_NAME_SRC_INDEX = "src_index";

const std::string ATTR_NAME_INPUT = "input";

const std::string ATTR_NAME_OUTPUT = "output";

const std::string ATTR_NAME_INPUT_DESC = "input_desc";

const std::string ATTR_NAME_OUTPUT_DESC = "output_desc";

const std::string ATTR_NAME_DST_NAME = "dst_name";

const std::string ATTR_NAME_DST_INDEX = "dst_index";

const std::string ATTR_NAME_WORKSPACE = "workspace";

const std::string ATTR_NAME_WORKSPACE_BYTES = "workspace_bytes";

const std::string ATTR_NAME_IS_INPUT_CONST = "is_input_const";

const std::string ATTR_NAME_OP_INFER_DEPENDS = "_op_infer_depends";

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc() {
  op_def_.InitDefault();
  if (op_def_.GetProtoMsg() != nullptr) {
    op_def_.GetProtoMsg()->set_has_out_attr(true);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::~OpDesc() {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(const std::string &name, const std::string &type) {
  op_def_.InitDefault();
  if (op_def_.GetProtoMsg() != nullptr) {
    op_def_.GetProtoMsg()->set_has_out_attr(true);
  }
  SetName(name);
  SetType(type);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(const ProtoMsgOwner &proto_msg_owner,
                                                              ge::proto::OpDef *op_def)
    : op_def_(proto_msg_owner, op_def) {
  if (op_def != nullptr && !op_def->has_out_attr()) {
    op_def->set_has_out_attr(true);

    int64_t id = 0;
    (void)AttrUtils::GetInt(this, ATTR_NAME_ID, id);
    op_def->set_id(id);

    int64_t stream_id = 0;
    (void)AttrUtils::GetInt(this, ATTR_NAME_STREAM_ID, stream_id);
    op_def->set_stream_id(stream_id);

    vector<string> input_name;
    (void)AttrUtils::GetListStr(this, ATTR_NAME_INPUT_NAME, input_name);
    for (auto &item : input_name) {
      op_def->add_input_name(item);
    }
    vector<string> src_name;
    (void)AttrUtils::GetListStr(this, ATTR_NAME_SRC_NAME, src_name);
    for (auto &item : src_name) {
      op_def->add_src_name(item);
    }
    vector<int64_t> src_index;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_SRC_INDEX, src_index);
    for (auto &item : src_index) {
      op_def->add_src_index(item);
    }
    vector<int64_t> input;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_INPUT, input);
    for (auto &item : input) {
      op_def->add_input_i(item);
    }
    vector<int64_t> output;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_OUTPUT, output);
    for (auto &item : output) {
      op_def->add_output_i(item);
    }
    vector<string> dst_name;
    (void)AttrUtils::GetListStr(this, ATTR_NAME_DST_NAME, dst_name);
    for (auto &item : dst_name) {
      op_def->add_dst_name(item);
    }
    vector<int64_t> dst_index;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_DST_INDEX, dst_index);
    for (auto &item : dst_index) {
      op_def->add_dst_index(item);
    }
    vector<int64_t> workspace;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_WORKSPACE, workspace);
    for (auto &item : workspace) {
      op_def->add_workspace(item);
    }
    vector<int64_t> workspace_bytes;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_WORKSPACE_BYTES, workspace_bytes);
    for (auto &item : workspace_bytes) {
      op_def->add_workspace_bytes(item);
    }
    vector<bool> is_input_const;
    (void)AttrUtils::GetListBool(this, ATTR_NAME_IS_INPUT_CONST, is_input_const);
    for (auto item : is_input_const) {
      op_def->add_is_input_const(item);
    }
    auto input_desc_mutable_list = (*op_def->mutable_attr())[ATTR_NAME_INPUT_DESC].mutable_list();
    if (input_desc_mutable_list != nullptr) {
      *op_def->mutable_input_desc() = *(input_desc_mutable_list->mutable_td());
    }
    auto output_desc_mutable_list = (*op_def->mutable_attr())[ATTR_NAME_OUTPUT_DESC].mutable_list();
    if (output_desc_mutable_list != nullptr) {
      *op_def->mutable_output_desc() = *(output_desc_mutable_list->mutable_td());
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY string OpDesc::GetName() const {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return proto_msg->name();
  }
  return "";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetName(const std::string &name) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_name(name);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY string OpDesc::GetType() const {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return proto_msg->type();
  }
  return "";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetType(const string &type) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_type(type);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::AddInputDesc(const ge::GeTensorDesc &input_desc) {
  int index = static_cast<int>(inputs_desc_.size());
  return AddInputDesc("__input" + std::to_string(index), input_desc);
}

graphStatus OpDesc::AddInputDesc(uint32_t index, const ge::GeTensorDesc &input_desc) {
  graphStatus ret = GRAPH_SUCCESS;
  if (index < inputs_desc_.size()) {
    //  InputsDesc[index] is exist, then update it
    ret = UpdateInputDesc(index, input_desc);
  } else {
    //  InputDesc[index] is not exist, then add it
    ret = AddInputDesc(input_desc);
  }
  return ret;
}

graphStatus OpDesc::AddInputDesc(const string &name, const ge::GeTensorDesc &input_desc) {
  if (input_name_idx_.find(name) != input_name_idx_.end()) {
    GELOGI("input %s is exist,  update it", name.c_str());
    graphStatus ret = UpdateInputDesc(name, input_desc);
    return ret;
  } else {
    int index = static_cast<int>(inputs_desc_.size());
    std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(input_desc);
    if (in_desc == nullptr) {
      GELOGE(GRAPH_FAILED, "AddInputDesc failed, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    inputs_desc_.push_back(in_desc);
    (void)input_name_idx_.insert(make_pair(name, index));
    if (find(register_input_name_.begin(), register_input_name_.end(), name) == register_input_name_.end()) {
      register_input_name_.push_back(name);
    }

    return GRAPH_SUCCESS;
  }
}

graphStatus OpDesc::AddInputDescMiddle(const string &name, const unsigned int num, size_t index) {
  for (unsigned int i = 0; i < num; i++) {
    string input_name = name + std::to_string(i);
    GE_CHK_BOOL_RET_STATUS((input_name_idx_.find(input_name) == input_name_idx_.end()), GRAPH_FAILED,
                           "Add input tensor_desc is existed. name[%s]", input_name.c_str());

    std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (in_desc == nullptr) {
      GELOGE(GRAPH_FAILED, "AddInputDescMiddle failed, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }

    if (index > inputs_desc_.size()) {
      GELOGE(GRAPH_FAILED, "AddInputDescMiddle failed, insert index should not more than inputs size.");
      return GRAPH_FAILED;
    }

    (void)inputs_desc_.insert(inputs_desc_.begin() + index + i, in_desc);

    // Update index in input_name_idx
    for (auto it = input_name_idx_.begin(); it != input_name_idx_.end(); ++it) {
      if (it->second >= (index + i)) {
        it->second += 1;
      }
    }

    (void)input_name_idx_.insert(make_pair(input_name, i + index));
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDesc::AddOutputDescMiddle(const string &name, const unsigned int num, size_t index) {
  for (unsigned int i = 0; i < num; i++) {
    string output_name = name + std::to_string(i);
    GE_CHK_BOOL_RET_STATUS((output_name_idx_.find(output_name) == output_name_idx_.end()), GRAPH_FAILED,
                           "Add input tensor_desc is existed. name[%s]", output_name.c_str());

    std::shared_ptr<GeTensorDesc> out_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (out_desc == nullptr) {
      GELOGE(GRAPH_FAILED, "AddInputDescMiddle failed, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }

    if (index > outputs_desc_.size()) {
      GELOGE(GRAPH_FAILED, "AddInputDescMiddle failed, insert index should not more than inputs size.");
      return GRAPH_FAILED;
    }

    (void)outputs_desc_.insert(outputs_desc_.begin() + index + i, out_desc);

    // Update index in input_name_idx
    for (auto it = output_name_idx_.begin(); it != output_name_idx_.end(); ++it) {
      if (it->second >= (index + i)) {
        it->second += 1;
      }
    }

    (void)output_name_idx_.insert(make_pair(output_name, i + index));
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDesc::AddInputDescForward(const string &name, const unsigned int num) {
  for (unsigned int i = 0; i < num; i++) {
    string input_name = name + std::to_string(i);
    GE_CHK_BOOL_RET_STATUS((input_name_idx_.find(input_name) == input_name_idx_.end()), GRAPH_FAILED,
                           "Add input tensor_desc is existed. name[%s]", input_name.c_str());

    std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (in_desc == nullptr) {
      GELOGE(GRAPH_FAILED, "AddInputDescForward failed, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    (void)inputs_desc_.insert(inputs_desc_.begin(), in_desc);

    // Update index in input_name_idx
    for (auto it = input_name_idx_.begin(); it != input_name_idx_.end(); ++it) {
      it->second += 1;
    }

    (void)input_name_idx_.insert(make_pair(input_name, 0));
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDesc::AddOutputDescForward(const string &name, const unsigned int num) {
  for (unsigned int i = 0; i < num; i++) {
    string output_name = name + std::to_string(i);
    GE_CHK_BOOL_RET_STATUS((output_name_idx_.find(output_name) == output_name_idx_.end()), GRAPH_FAILED,
                           "Add output tensor_desc is existed. name[%s]", output_name.c_str());

    std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (in_desc == nullptr) {
      GELOGE(GRAPH_FAILED, "AddOutputDescForward failed, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }

    (void)outputs_desc_.insert(outputs_desc_.begin(), in_desc);

    // Update index in output_name_idx
    for (auto it = output_name_idx_.begin(); it != output_name_idx_.end(); ++it) {
      it->second += 1;
    }
    (void)output_name_idx_.insert(make_pair(output_name, 0));
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDesc::AddOptionalInputDesc(const string &name, const ge::GeTensorDesc &input_desc) {
  if (OpDesc::AddInputDesc(name, input_desc) == GRAPH_FAILED) return GRAPH_FAILED;
  (void)optional_input_names_.insert(name);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDesc::UpdateInputDesc(uint32_t index, const ge::GeTensorDesc &tensor_Desc) {
  if (index >= inputs_desc_.size()) {
    GELOGW("The index is invalid. index[%u]", index);
    return GRAPH_FAILED;
  }

  inputs_desc_[index] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (inputs_desc_[index] == nullptr) {
    GELOGE(GRAPH_FAILED, "UpdateInputDesc failed, malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::OpDescMembersAreEqual(const OpDesc &r_op_desc) const {
  return (IsEqual(this->input_name_idx_, r_op_desc.input_name_idx_, "OpDesc.input_name_idx_") &&
          IsEqual(this->output_name_idx_, r_op_desc.output_name_idx_, "OpDesc.output_name_idx_") &&
          IsEqual(this->optional_input_names_, r_op_desc.optional_input_names_, "OpDesc.optional_input_names_") &&
          IsEqual(this->engine_name_, r_op_desc.engine_name_, "OpDesc.engine_name_") &&
          IsEqual(this->op_kernel_lib_name_, r_op_desc.op_kernel_lib_name_, "OpDesc.op_kernel_lib_name_"));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::OpDescAttrsAreEqual(const OpDesc &r_op_desc) const {
  const auto &op_def = this->op_def_.GetProtoMsg();
  const auto &r_op_def = r_op_desc.op_def_.GetProtoMsg();
  if ((op_def != nullptr) && (r_op_def != nullptr)) {
    // Message OpDef in ge_ir.proto
    return (
      IsEqual(op_def->name(), r_op_def->name(), "OpDef_.name()") &&
      IsEqual(op_def->type(), r_op_def->type(), "OpDef_.type()") &&
      IsEqual(ToString(op_def->input()), ToString(r_op_def->input()), "OpDef_.input()") &&
      IsEqual(op_def->has_out_attr(), r_op_def->has_out_attr(), "OpDef_.has_out_attr()") &&
      IsEqual(op_def->stream_id(), r_op_def->stream_id(), "OpDef_.stream_id()") &&
      IsEqual(ToString(op_def->input_name()), ToString(r_op_def->input_name()), "OpDef_.input_name()") &&
      IsEqual(ToString(op_def->src_name()), ToString(r_op_def->src_name()), "OpDef_.src_name()") &&
      IsEqual(ToString(op_def->dst_name()), ToString(r_op_def->dst_name()), "OpDef_.dst_name()") &&
      IsEqual(ToString(op_def->src_index()), ToString(r_op_def->src_index()), "OpDef_.src_index()") &&
      IsEqual(ToString(op_def->dst_index()), ToString(r_op_def->dst_index()), "OpDef_.dst_index()") &&
      IsEqual(ToString(op_def->input_i()), ToString(r_op_def->input_i()), "OpDef_.input_i()") &&
      IsEqual(ToString(op_def->output_i()), ToString(r_op_def->output_i()), "OpDef_.output_i()") &&
      IsEqual(ToString(op_def->workspace()), ToString(r_op_def->workspace()), "OpDef_.workspace()") &&
      IsEqual(ToString(op_def->workspace_bytes()), ToString(r_op_def->workspace_bytes()), "OpDef_.workspace_bytes()") &&
      IsEqual(ToString(op_def->is_input_const()), ToString(r_op_def->is_input_const()), "OpDef_.is_input_const()"));
  } else {
    return ((op_def == nullptr) && (r_op_def == nullptr));
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::OpDescGenTensorDescsAreEqual(
  const OpDesc &r_op_desc) const {
  // 1.Verify inputs and outputs desc size
  const auto inputs_desc_size = this->inputs_desc_.size();
  const auto r_inputs_desc_size = r_op_desc.inputs_desc_.size();
  if (inputs_desc_size != r_inputs_desc_size) {
    GELOGE(GRAPH_FAILED, "Size of OpDesc's inputs desc verify failed, node name: %s.", this->GetName().c_str());
    return false;
  }
  const auto outputs_desc_size = this->outputs_desc_.size();
  const auto r_outputs_desc_size = r_op_desc.outputs_desc_.size();
  if (outputs_desc_size != r_outputs_desc_size) {
    GELOGE(GRAPH_FAILED, "Size of OpDesc's outputs desc verify failed, node name: %s.", this->GetName().c_str());
    return false;
  }
  // 2.Verify all inputs desc equal
  for (uint32_t i = 0; i < inputs_desc_size; i++) {
    const auto &in_ge_tensor_desc = this->GetInputDesc(i);
    const auto &r_in_ge_tensor_desc = r_op_desc.GetInputDesc(i);
    // Determine the connection relationship by GeTensorDesc
    if (!(in_ge_tensor_desc == r_in_ge_tensor_desc)) {
      GELOGE(GRAPH_FAILED, "Link info of OpDesc's inputs desc verify failed, OpDesc name: %s.",
             this->GetName().c_str());
      return false;
    }
  }
  // 3.Verify all outputs desc equal
  for (uint32_t i = 0; i < outputs_desc_size; i++) {
    const auto &out_ge_tensor_desc = this->GetOutputDesc(i);
    const auto &r_out_ge_tensor_desc = r_op_desc.GetOutputDesc(i);
    if (!(out_ge_tensor_desc == r_out_ge_tensor_desc)) {
      GELOGE(GRAPH_FAILED, "Link info of OpDesc's outputs desc verify failed, OpDesc name: %s.",
             this->GetName().c_str());
      return false;
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::operator==(const OpDesc &r_op_desc) const {
  return (OpDescAttrsAreEqual(r_op_desc) && OpDescMembersAreEqual(r_op_desc) &&
          OpDescGenTensorDescsAreEqual(r_op_desc));
}

graphStatus OpDesc::UpdateInputDesc(const string &name, const ge::GeTensorDesc &tensor_Desc) {
  auto it = input_name_idx_.find(name);
  if (it == input_name_idx_.end()) {
    GELOGW("Cann't find the input desc. name[%s]", name.c_str());
    return GRAPH_FAILED;
  }
  if (it->second >= inputs_desc_.size()) {
    GELOGE(GRAPH_FAILED, "[%d] more than size of inputs_desc_", it->second);
    return GRAPH_FAILED;
  }
  GE_IF_BOOL_EXEC(it->second >= inputs_desc_.size(), GELOGE(GRAPH_FAILED, "it->second is invalid.");
                  return GRAPH_FAILED);
  inputs_desc_[it->second] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (inputs_desc_[it->second] == nullptr) {
    GELOGE(GRAPH_FAILED, "UpdateInputDesc failed, malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

bool OpDesc::InputIsSet(const string &name) const {
  auto it = input_name_idx_.find(name);
  if (it != input_name_idx_.end()) {
    GE_IF_BOOL_EXEC(it->second >= inputs_desc_.size(), GELOGE(GRAPH_FAILED, "it->second is invalid."); return false);
    auto tensor_desc = inputs_desc_[it->second];
    GE_IF_BOOL_EXEC(tensor_desc == nullptr, GELOGE(GRAPH_FAILED, "tensor_desc is null."); return false);
    auto dims = tensor_desc->GetShape().GetDims();
    if (dims.size() > 0) {
      return true;
    }
  }
  return false;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDesc OpDesc::GetInputDesc(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG(index < inputs_desc_.size(), GeTensorDesc());
  return *(inputs_desc_[index].get());
}

GeTensorDesc OpDesc::GetInputDesc(const string &name) const {
  auto it = input_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != input_name_idx_.end(), GeTensorDesc());
  GE_CHK_BOOL_RET_STATUS_NOLOG(it->second < inputs_desc_.size(), GeTensorDesc());
  return *(inputs_desc_[it->second].get());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr OpDesc::MutableInputDesc(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS(index < inputs_desc_.size(), nullptr, "Can't find the input desc %u", index);
  if (inputs_desc_[index] == nullptr) {
    return nullptr;
  }
  if (inputs_desc_[index]->IsValid() != GRAPH_SUCCESS) {
    GELOGW("input desc is invalid");
    return nullptr;
  }
  return inputs_desc_[index];
}

GeTensorDescPtr OpDesc::MutableInputDesc(const string &name) const {
  auto input_name_idx = GetAllInputName();
  auto it = input_name_idx.find(name);
  if (it == input_name_idx.end()) {
    GELOGW("Failed to get [%s] input desc", name.c_str());
    return nullptr;
  }
  return MutableInputDesc(it->second);
}

GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<string> OpDesc::GetAllInputNames() const {
  vector<string> names;
  if (input_name_idx_.empty()) {
    return OpDesc::Vistor<string>(shared_from_this(), names);
  }
  for (std::pair<string, uint32_t> input : input_name_idx_) {
    names.push_back(input.first);
  }
  return OpDesc::Vistor<string>(shared_from_this(), names);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOpKernelLibName(const std::string &name) {
  op_kernel_lib_name_ = name;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetOpKernelLibName() const {
  return op_kernel_lib_name_;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOpEngineName(const std::string &name) {
  engine_name_ = name;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetOpEngineName() const { return engine_name_; }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDesc> OpDesc::GetAllInputsDesc() const {
  vector<GeTensorDesc> temp{};
  for (const auto &it : inputs_desc_) {
    if (it->IsValid() == GRAPH_SUCCESS) {
      temp.push_back(*it);
    } else {
      GELOGW("this inputDesc is InValid, it won't be return");
      continue;
    }
  }
  return OpDesc::Vistor<GeTensorDesc>(shared_from_this(), temp);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDescPtr> OpDesc::GetAllInputsDescPtr() const {
  vector<GeTensorDescPtr> temp{};
  for (const auto &it : inputs_desc_) {
    if (it->IsValid() == GRAPH_SUCCESS) {
      temp.push_back(it);
    } else {
      GELOGW("this inputDesc is InValid, it won't be return");
      continue;
    }
  }
  return OpDesc::Vistor<GeTensorDescPtr>(shared_from_this(), temp);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetInputsSize() const {
  //  Just return valid inputs size.InValid desc is set in default OPTION_INPUT register.
  size_t size = 0;
  for (const auto &it : inputs_desc_) {
    if (it->IsValid() == GRAPH_SUCCESS) {
      size++;
    }
  }
  return size;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetAllInputsSize() const { return inputs_desc_.size(); }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::AddOutputDesc(const ge::GeTensorDesc &output_desc) {
  int index = static_cast<int>(outputs_desc_.size());
  return AddOutputDesc("__output" + std::to_string(index), output_desc);
}

graphStatus OpDesc::AddOutputDesc(const string &name, const ge::GeTensorDesc &output_desc) {
  GE_CHK_BOOL_RET_STATUS((output_name_idx_.find(name) == output_name_idx_.end()), GRAPH_FAILED,
                         "Add output tensor_Desc is existed. name[%s]", name.c_str());
  int index = static_cast<int>(outputs_desc_.size());

  std::shared_ptr<GeTensorDesc> tensor = ComGraphMakeShared<GeTensorDesc>(output_desc);
  if (tensor == nullptr) {
    GELOGE(GRAPH_FAILED, "AddOutputDesc failed, malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }
  outputs_desc_.push_back(tensor);
  (void)output_name_idx_.insert(make_pair(name, index));
  if (find(register_output_name_.begin(), register_output_name_.end(), name) == register_output_name_.end()) {
    register_output_name_.push_back(name);
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDesc::UpdateOutputDesc(uint32_t index, const ge::GeTensorDesc &tensor_Desc) {
  GE_CHK_BOOL_RET_STATUS((index < outputs_desc_.size()), GRAPH_FAILED, "The index is invalid. index[%u]", index);

  outputs_desc_[index] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (outputs_desc_[index] == nullptr) {
    GELOGE(GRAPH_FAILED, "UpdateOutputDesc failed, malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDesc::UpdateOutputDesc(const string &name, const ge::GeTensorDesc &tensor_Desc) {
  auto it = output_name_idx_.find(name);
  if (it == output_name_idx_.end()) {
    GELOGW("Cann't find the output desc. name[%s]", name.c_str());
    return GRAPH_FAILED;
  }
  GE_IF_BOOL_EXEC(it->second >= outputs_desc_.size(), GELOGE(GRAPH_FAILED, "it->second is invalid.");
                  return GRAPH_FAILED);
  outputs_desc_[it->second] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (outputs_desc_[it->second] == nullptr) {
    GELOGE(GRAPH_FAILED, "UpdateOutputDesc failed, malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDesc OpDesc::GetOutputDesc(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG(index < outputs_desc_.size(), GeTensorDesc());
  return *(outputs_desc_[index].get());
}

GeTensorDesc OpDesc::GetOutputDesc(const string &name) const {
  auto it = output_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != output_name_idx_.end(), GeTensorDesc());
  GE_CHK_BOOL_RET_STATUS_NOLOG(it->second < outputs_desc_.size(), GeTensorDesc());
  return *(outputs_desc_[it->second].get());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr OpDesc::MutableOutputDesc(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS(index < outputs_desc_.size(), nullptr, "Cann't find the output desc %u", index);
  return outputs_desc_[index];
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr OpDesc::MutableOutputDesc(const string &name) const {
  auto it = output_name_idx_.find(name);
  if (it == output_name_idx_.end()) {
    GELOGW("Failed to get [%s] output desc", name.c_str());
    return nullptr;
  }
  return MutableOutputDesc(it->second);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t OpDesc::GetAllOutputsDescSize() const {
  return static_cast<uint32_t>(outputs_desc_.size());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDesc> OpDesc::GetAllOutputsDesc() const {
  vector<GeTensorDesc> temp{};
  for (const auto &it : outputs_desc_) {
    temp.push_back(*it);
  }
  return OpDesc::Vistor<GeTensorDesc>(shared_from_this(), temp);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDescPtr> OpDesc::GetAllOutputsDescPtr() const {
  return OpDesc::Vistor<GeTensorDescPtr>(shared_from_this(), outputs_desc_);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetOutputsSize() const { return outputs_desc_.size(); }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr OpDesc::GetOutputDescPtr(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG((index) < static_cast<uint32_t>(outputs_desc_.size()), nullptr);
  return outputs_desc_[index];
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr OpDesc::GetInputDescPtr(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG((index) < static_cast<uint32_t>(inputs_desc_.size()), nullptr);
  if (inputs_desc_[index] == nullptr) {
    return nullptr;
  }
  if (inputs_desc_[index]->IsValid() != GRAPH_SUCCESS) {
    GELOGW("inputsDesc[%u] is InValid", index);
    return nullptr;
  } else {
    return inputs_desc_[static_cast<size_t>(index)];
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr
OpDesc::GetInputDescPtrDfault(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG((index) < (uint32_t)(inputs_desc_.size()), nullptr);
  return inputs_desc_[(int32_t)index];
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr OpDesc::GetInputDescPtr(const string &name) const {
  auto it = input_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != input_name_idx_.end(), shared_ptr<const GeTensorDesc>());
  return inputs_desc_[it->second];
}

graphStatus OpDesc::AddRegisterInputName(const std::string &name) {
  if (find(register_input_name_.begin(), register_input_name_.end(), name) == register_input_name_.end()) {
    register_input_name_.push_back(name);
  }

  return GRAPH_SUCCESS;
}

vector<string> OpDesc::GetRegisterInputName() const { return register_input_name_; }

graphStatus OpDesc::AddDynamicInputDesc(const string &name, const unsigned int num, bool is_push_back) {
  if (is_push_back) {
    for (unsigned int i = 0; i < num; i++) {
      if (AddInputDesc(name + std::to_string(i), GeTensorDesc()) != GRAPH_SUCCESS) return GRAPH_FAILED;
    }
  } else {
    if (AddInputDescForward(name, num) != GRAPH_SUCCESS) return GRAPH_FAILED;
  }
  if (AddRegisterInputName(name) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDesc::AddDynamicInputDescByIndex(const string &name, const unsigned int num, size_t index) {
  if (AddInputDescMiddle(name, num, index) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDesc::AddRegisterOutputName(const string &name) {
  if (find(register_output_name_.begin(), register_output_name_.end(), name) == register_output_name_.end()) {
    register_output_name_.push_back(name);
  }

  return GRAPH_SUCCESS;
}

vector<string> OpDesc::GetRegisterOutputName() const { return register_output_name_; }

graphStatus OpDesc::AddDynamicOutputDesc(const string &name, const unsigned int num, bool is_push_back) {
  if (is_push_back) {
    for (unsigned int i = 0; i < num; i++) {
      if (AddOutputDesc(name + std::to_string(i), GeTensorDesc()) != GRAPH_SUCCESS) return GRAPH_FAILED;
    }
  } else {
    if (AddOutputDescForward(name, num) != GRAPH_SUCCESS) return GRAPH_FAILED;
  }

  if (AddRegisterOutputName(name) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool OpDesc::IsOptionalInput(const string &name) const {
  return optional_input_names_.find(name) != optional_input_names_.end();
}

bool OpDesc::IsOptionalInput(uint32_t index) const { return IsOptionalInput(GetInputNameByIndex(index)); }

std::map<string, uint32_t> OpDesc::GetAllInputName() const { return input_name_idx_; }

std::map<string, uint32_t> OpDesc::GetAllOutputName() { return output_name_idx_; }

bool OpDesc::UpdateInputName(std::map<string, uint32_t> input_name_idx) {
  bool ret = true;
  //  Use inputDesc_.size() to contain the InValid OptionInput.GetInputsSize() will remove default OptionInput name.
  auto input_map_size = inputs_desc_.size();
  auto factory_map_size = input_name_idx.size();
  // It indicates that some inputs have no optionalname.
  // The redundant optionalname of factory needs to be deleted and then assigned
  if (input_map_size < factory_map_size) {
    GELOGI("UpdateInputName org inputname map size: %zu, factory inputname map size: %zu", input_map_size,
           factory_map_size);
    for (auto it = input_name_idx.begin(); it != input_name_idx.end();) {
      if (it->second >= input_map_size) {
        it = input_name_idx.erase(it);
      } else {
        ++it;
      }
    }
    if (input_name_idx.size() == input_map_size) {
      GELOGI("UpdateInputName");
      input_name_idx_ = input_name_idx;
    } else {
      ret = false;
      GELOGW("after UpdateInputName factoryName map size : %zu", input_name_idx.size());
    }
  } else if (input_map_size == factory_map_size) {
    input_name_idx_ = input_name_idx;
  } else {
    ret = false;
    GELOGW("org inputname map size: %zu, factory inputname map size: %zu", input_map_size, factory_map_size);
  }
  return ret;
}

bool OpDesc::UpdateOutputName(std::map<string, uint32_t> output_name_idx) {
  size_t output_map_size = GetAllOutputsDescSize();
  size_t factory_map_size = output_name_idx.size();
  if (output_map_size < factory_map_size) {
    GELOGI("UpdateOutputName org outputname map size: %zu, factory outputname map size: %zu", output_map_size,
           factory_map_size);
    for (auto it = output_name_idx.begin(); it != output_name_idx.end();) {
      if (it->second >= output_map_size) {
        it = output_name_idx.erase(it);
      } else {
        ++it;
      }
    }
    if (output_name_idx.size() == output_map_size) {
      GELOGI("UpdateoutputName");
      output_name_idx_ = output_name_idx;
      return true;
    }
  } else if (output_map_size == factory_map_size) {
    output_name_idx_ = output_name_idx;
    return true;
  } else {
    GELOGW("UpdateOutputName org name map size: %zu, factory map size: %zu", output_map_size, factory_map_size);
    return false;
  }
  GELOGW("UpdateOutputName org name map size: %zu, factory map size: %zu", output_map_size, factory_map_size);
  return false;
}

std::function<graphStatus(Operator &)> OpDesc::GetInferFunc() const { return infer_func_; }

std::function<graphStatus(Operator &)> OpDesc::GetVerifyFunc() const { return verifier_func_; }

void OpDesc::AddInferFunc(const std::function<graphStatus(Operator &)> &func) { infer_func_ = func; }

std::function<graphStatus(Operator &)> OpDesc::GetInferFormatFunc() const { return infer_format_func_; }

void OpDesc::AddInferFormatFunc(const std::function<graphStatus(Operator &)> &func) { infer_format_func_ = func; }

void OpDesc::AddVerifierFunc(const std::function<graphStatus(Operator &)> &func) { verifier_func_ = func; }

graphStatus OpDesc::InferShapeAndType() {
  if (infer_func_ == nullptr) {
    infer_func_ = OperatorFactoryImpl::GetInferShapeFunc(GetType());
    if (infer_func_ == nullptr) {
      GELOGW("%s does not have inferfunc_.", GetName().c_str());
      /// The infoshape function has not been added for each operator in the current operator information library.
      /// No infoshape added operator skips the call
      /// and directly uses the shape information passed down by the upper framework
      return GRAPH_SUCCESS;
    }
  }
  Operator op_proxy = ge::OpDescUtils::CreateOperatorFromOpDesc(shared_from_this());
  graphStatus ret = (graphStatus)infer_func_(op_proxy);
  op_proxy.BreakConnect();
  return ret;
}

graphStatus OpDesc::DefaultInferFormat() {
  ge::Format first_none_nd_format = FORMAT_ND;
  auto input_descs = GetAllInputsDescPtr();
  auto output_descs = GetAllOutputsDescPtr();
  // Overall input and output,get the first non-nd format
  for (const auto &input_desc : input_descs) {
    Format origin_format = input_desc->GetOriginFormat();
    if (origin_format != FORMAT_ND) {
      first_none_nd_format = origin_format;
      break;
    }
  }
  for (const auto &output_desc : output_descs) {
    Format origin_format = output_desc->GetOriginFormat();
    if (origin_format != FORMAT_ND) {
      first_none_nd_format = origin_format;
      break;
    }
  }
  // Refresh all input output format
  GELOGD("Default infer format.node[%s], first none nod format is:%d", GetName().c_str(), first_none_nd_format);

  for (const auto &input_desc : input_descs) {
    Format origin_format = input_desc->GetOriginFormat();
    GELOGD("Default infer format[in].node[%s].origin format is:%d", GetName().c_str(), origin_format);
    if (origin_format == FORMAT_ND) {
      input_desc->SetOriginFormat(first_none_nd_format);
      input_desc->SetFormat(first_none_nd_format);
    }
  }
  for (const auto &output_desc : output_descs) {
    Format origin_format = output_desc->GetOriginFormat();
    GELOGD("Default infer format[out].node[%s].origin format is:%d", GetName().c_str(), origin_format);
    if (origin_format == FORMAT_ND) {
      output_desc->SetOriginFormat(first_none_nd_format);
      output_desc->SetFormat(first_none_nd_format);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDesc::OpVerify() {
  if (verifier_func_ == nullptr) {
    verifier_func_ = OperatorFactoryImpl::GetVerifyFunc(GetType());
  }
  if (verifier_func_ != nullptr) {
    Operator op_proxy = ge::OpDescUtils::CreateOperatorFromOpDesc(shared_from_this());
    graphStatus ret = (graphStatus)verifier_func_(op_proxy);
    op_proxy.BreakConnect();
    return ret;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDesc::CommonVerify() const {
  for (const string &iname : GetAllInputNames()) {
    // Checking shape of all inputs
    vector<int64_t> ishape = GetInputDescPtr(iname)->GetShape().GetDims();
    for (int64_t dim : ishape) {
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        dim < -2, ErrorManager::GetInstance().ATCReportErrMessage(
                    "E19014", {"opname", "value", "reason"},
                    {GetName(), "input " + iname + " shape", "contains negative or zero dimension"});
        return GRAPH_FAILED, "Op[%s]'s input %s shape contains negative or zero dimension.", GetName().c_str(),
               iname.c_str());
    }
  }
  // Check all attributes defined
  const auto &all_attributes = GetAllAttrs();
  for (const auto &name : GetAllAttrNames()) {
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      all_attributes.find(name) == all_attributes.end(),
      ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
                                                      {GetName(), "attribute " + name, "is empty"});
      return GRAPH_FAILED, "operator attribute %s is empty.", name.c_str());
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY string OpDesc::GetInputNameByIndex(uint32_t index) const {
  auto it = input_name_idx_.begin();
  for (; it != input_name_idx_.end(); ++it) {
    if (it->second == index) {
      break;
    }
  }
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != input_name_idx_.end(), "");
  return it->first;
}

int OpDesc::GetInputIndexByName(const string &name) const {
  auto it_find = input_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it_find != input_name_idx_.end(), -1);
  return static_cast<int>(it_find->second);
}

int OpDesc::GetValidInputIndexByName(const string &name) const {
  map<string, uint32_t> valid_input_name_idx{};
  uint32_t j = 0;
  for (size_t i = 0; i < GetAllInputsSize(); i++) {
    if (MutableInputDesc(static_cast<uint32_t>(i)) != nullptr) {
      auto valid_name = GetInputNameByIndex(static_cast<uint32_t>(i));
      GE_CHK_BOOL_RET_STATUS_NOLOG(!valid_name.empty(), -1);
      valid_input_name_idx.insert({valid_name, j});
      j++;
    }
  }
  auto it_find = valid_input_name_idx.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it_find != valid_input_name_idx.end(), -1);
  return static_cast<int>(it_find->second);
}

string OpDesc::GetValidInputNameByIndex(uint32_t index) const {
  map<string, uint32_t> valid_input_name_idx{};
  uint32_t j = 0;
  for (size_t i = 0; i < GetAllInputsSize(); i++) {
    if (MutableInputDesc(static_cast<uint32_t>(i)) != nullptr) {
      auto valid_name = GetInputNameByIndex(static_cast<uint32_t>(i));
      GE_CHK_BOOL_RET_STATUS_NOLOG(!valid_name.empty(), "");
      valid_input_name_idx.insert({valid_name, j});
      j++;
    }
  }
  auto it = valid_input_name_idx.begin();
  for (; it != valid_input_name_idx.end(); ++it) {
    if (it->second == index) {
      break;
    }
  }
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != valid_input_name_idx.end(), "");
  return it->first;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY string OpDesc::GetOutputNameByIndex(uint32_t index) const {
  auto it = output_name_idx_.begin();
  for (; it != output_name_idx_.end(); ++it) {
    if (it->second == index) {
      break;
    }
  }
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != output_name_idx_.end(), "");
  return it->first;
}

int OpDesc::GetOutputIndexByName(const string &name) const {
  auto it_find = output_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it_find != output_name_idx_.end(), -1);
  return static_cast<int>(it_find->second);
}

ProtoAttrMapHelper OpDesc::MutableAttrMap() {
  if (op_def_.GetProtoMsg() == nullptr) {
    GELOGE(GRAPH_FAILED, "op def get proto msg failed");
    return GeIrProtoHelper<ProtoAttrMap>();
  }
  return ProtoAttrMapHelper(op_def_.GetProtoOwner(), op_def_.GetProtoMsg()->mutable_attr());
}

ConstProtoAttrMapHelper OpDesc::GetAttrMap() const {
  return ConstProtoAttrMapHelper(op_def_.GetProtoOwner(), &op_def_.GetProtoMsg()->attr());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetId(int64_t id) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_id(id);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int64_t OpDesc::GetId() const {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return proto_msg->id();
  }
  return 0;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetStreamId(int64_t stream_id) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_stream_id(stream_id);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int64_t OpDesc::GetStreamId() const {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return proto_msg->stream_id();
  }
  return 0;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetInputName(const vector<string> &input_name) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_input_name();
    for (auto &item : input_name) {
      proto_msg->add_input_name(item);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<string> OpDesc::GetInputName() const {
  vector<string> input_name;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->input_name()) {
      input_name.push_back(item);
    }
  }
  return input_name;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetSrcName(const vector<string> &src_name) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_src_name();
    for (auto &item : src_name) {
      proto_msg->add_src_name(item);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<string> OpDesc::GetSrcName() const {
  vector<string> src_name;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->src_name()) {
      src_name.push_back(item);
    }
  }
  return src_name;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetSrcIndex(const vector<int64_t> &src_index) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_src_index();
    for (auto &item : src_index) {
      proto_msg->add_src_index(item);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetSrcIndex() const {
  vector<int64_t> src_index;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->src_index()) {
      src_index.push_back(item);
    }
  }
  return src_index;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetInputOffset(const vector<int64_t> &input) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_input_i();
    for (auto &item : input) {
      proto_msg->add_input_i(item);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetInputOffset() const {
  vector<int64_t> input;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->input_i()) {
      input.push_back(item);
    }
  }
  return input;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOutputOffset(const vector<int64_t> &output) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_output_i();
    for (auto &item : output) {
      proto_msg->add_output_i(item);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetOutputOffset() const {
  vector<int64_t> output;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->output_i()) {
      output.push_back(item);
    }
  }
  return output;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetDstName(const vector<string> &dst_name) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_dst_name();
    for (auto &item : dst_name) {
      proto_msg->add_dst_name(item);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<string> OpDesc::GetDstName() const {
  vector<string> dst_name;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->dst_name()) {
      dst_name.push_back(item);
    }
  }
  return dst_name;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOpInferDepends(const vector<string> &depend_names) {
  auto ret = AttrUtils::SetListStr(this, ATTR_NAME_OP_INFER_DEPENDS, depend_names);
  if (ret != true) {
    GELOGE(GRAPH_FAILED, "set op_infer_depends fail.");
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<string> OpDesc::GetOpInferDepends() const {
  vector<string> depend_names;
  (void)AttrUtils::GetListStr(this, ATTR_NAME_OP_INFER_DEPENDS, depend_names);
  return depend_names;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetDstIndex(const vector<int64_t> &dst_index) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_dst_index();
    for (auto &item : dst_index) {
      proto_msg->add_dst_index(item);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetDstIndex() const {
  vector<int64_t> dst_index;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->dst_index()) {
      dst_index.push_back(item);
    }
  }
  return dst_index;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetWorkspace(const vector<int64_t> &workspace) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_workspace();
    for (auto &item : workspace) {
      proto_msg->add_workspace(item);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetWorkspace() const {
  vector<int64_t> workspace;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->workspace()) {
      workspace.push_back(item);
    }
  }
  return workspace;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetWorkspaceBytes(const vector<int64_t> &workspace_bytes) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_workspace_bytes();
    for (auto &item : workspace_bytes) {
      proto_msg->add_workspace_bytes(item);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetWorkspaceBytes() const {
  vector<int64_t> workspace_bytes;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->workspace_bytes()) {
      workspace_bytes.push_back(item);
    }
  }
  return workspace_bytes;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetIsInputConst(const vector<bool> &is_input_const) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_is_input_const();
    for (auto item : is_input_const) {
      proto_msg->add_is_input_const(item);
    }
  }
  // If comes from ME,which is_input_const exist as attrs, outside no need to check GE_TRAIN flag
  auto ret = AttrUtils::SetListBool(this, ATTR_NAME_IS_INPUT_CONST, is_input_const);
  if (ret != true) {
    GELOGE(GRAPH_FAILED, "set is_input_const fail.");
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<bool> OpDesc::GetIsInputConst() const {
  vector<bool> is_input_const;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto item : proto_msg->is_input_const()) {
      is_input_const.push_back(item);
    }
  }
  return is_input_const;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::RestoreInputNameIdx(const string &name,
                                                                                       const int &index) {
  if (input_name_idx_.find(name) != input_name_idx_.end()) {
    GELOGI("Restore input name index is existed. name[%s]", name.c_str());
  }
  (void)input_name_idx_.insert(make_pair(name, index));
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::RestoreOutputNameIdx(const string &name,
                                                                                        const int &index) {
  if (output_name_idx_.find(name) != output_name_idx_.end()) {
    GELOGI("Restore output name index is existed. name[%s]", name.c_str());
  }
  (void)output_name_idx_.insert(make_pair(name, index));
  return GRAPH_SUCCESS;
}
graphStatus OpDesc::CallInferFunc(Operator &op) {
  if (infer_func_ == nullptr) {
    infer_func_ = OperatorFactoryImpl::GetInferShapeFunc(GetType());
    if (infer_func_ == nullptr) {
      GELOGW("%s does not have infer func.", GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
  }
  graphStatus graph_status = (graphStatus)infer_func_(op);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "%s call infer func. ret: %u", GetName().c_str(), graph_status);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
graphStatus OpDesc::CallInferFormatFunc(Operator &op) {
  if (infer_format_func_ == nullptr) {
    infer_format_func_ = OperatorFactoryImpl::GetInferFormatFunc(GetType());
    if (infer_format_func_ == nullptr) {
      return DefaultInferFormat();
    }
  }
  return (graphStatus)infer_format_func_(op);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetSubgraphInstanceName(uint32_t index) const {
  if (static_cast<size_t>(index) >= subgraph_instance_names_.size()) {
    return "";
  }
  return subgraph_instance_names_.at(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<std::string> &OpDesc::GetSubgraphInstanceNames()
  const {
  return subgraph_instance_names_;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::RemoveSubgraphInstanceName(const std::string &name) {
  for (auto iter = subgraph_instance_names_.begin(); iter != subgraph_instance_names_.end(); ++iter) {
    if (*iter == name) {
      *iter = "";
      return;
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::AddSubgraphName(const std::string &name) {
  GELOGI("Add subgraph name is %s", name.c_str());
  auto iter = subgraph_names_to_index_.find(name);
  if (iter != subgraph_names_to_index_.end()) {
    GELOGW("The subgraph name %s exists, index %u", name.c_str(), iter->second);
    return GRAPH_FAILED;
  }
  auto size = subgraph_names_to_index_.size();
  subgraph_names_to_index_[name] = size;
  subgraph_instance_names_.resize(size + 1);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::map<std::string, uint32_t> &OpDesc::GetSubgraphNameIndexes()
  const {
  return subgraph_names_to_index_;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::SetSubgraphInstanceName(uint32_t index,
                                                                                           const std::string &name) {
  GELOGI("Add sub graph instans name is %s, index is %u", name.c_str(), index);
  if (index >= subgraph_instance_names_.size()) {
    GE_LOGE("The index %u exceeds the max instance coutn %zu", index, subgraph_instance_names_.size());
    return GRAPH_PARAM_INVALID;
  }
  subgraph_instance_names_[index] = name;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::RegisterSubgraphIrName(const string &name,
                                                                                   SubgraphType type) {
  subgraph_ir_names_to_type_[name] = type;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::map<std::string, SubgraphType> &OpDesc::GetSubgraphIrNames()
  const {
  return subgraph_ir_names_to_type_;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY SubgraphType
OpDesc::GetSubgraphTypeByIrName(const std::string &name) const {
  auto iter = subgraph_ir_names_to_type_.find(name);
  if (iter == subgraph_ir_names_to_type_.end()) {
    return kSubgraphTypeEnd;
  }
  return iter->second;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDesc::GetSubgraphNameByInstanceName(const std::string &instance_name, std::string &subgraph_name) const {
  for (size_t idx = 0; idx < subgraph_instance_names_.size(); ++idx) {
    if (subgraph_instance_names_[idx] != instance_name) {  // find subgraph index.
      continue;
    }

    for (auto name_to_index : subgraph_names_to_index_) {
      if (name_to_index.second != idx) {  // find subgraph name.
        continue;
      }

      subgraph_name = name_to_index.first;
      return GRAPH_SUCCESS;
    }
  }

  return GRAPH_PARAM_INVALID;
}

}  // namespace ge
