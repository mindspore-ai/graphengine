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

#ifndef INC_GRAPH_OP_DESC_H_
#define INC_GRAPH_OP_DESC_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "detail/attributes_holder.h"
#include "graph/range_vistor.h"

#define DYNAMIN_INPUT_NAME(name, index) (((name)) + std::to_string((index)))
#define DYNAMIN_OUTPUT_NAME(name, index) (((name)) + std::to_string((index)))
namespace ge {
using std::map;
using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;

class Operator;
class GeTensorDesc;

using GeTensorDescPtr = shared_ptr<GeTensorDesc>;
using ConstGeTensorDescPtr = shared_ptr<const GeTensorDesc>;

class OpDesc;

using OpDescPtr = shared_ptr<OpDesc>;
using ConstOpDescPtr = shared_ptr<const OpDesc>;

class GeAttrValue;

using ConstOpDesc = const OpDesc;

class OpDesc : public std::enable_shared_from_this<OpDesc>, public AttrHolder {
 public:
  template <class T>
  using Vistor = RangeVistor<T, shared_ptr<ConstOpDesc>>;

  friend class GraphBuilderImpl;

  friend class OperatorImpl;

  OpDesc(const string &name, const string &type);

  OpDesc();

  ~OpDesc();

  bool operator==(const OpDesc &r_op_desc) const;

  string GetName() const;

  void SetName(const string &name);

  string GetType() const;

  void SetType(const string &type);

  graphStatus AddInputDesc(const GeTensorDesc &input_desc);

  graphStatus AddInputDesc(const string &name, const GeTensorDesc &input_desc);

  graphStatus AddInputDesc(uint32_t index, const ge::GeTensorDesc &input_desc);

  graphStatus AddInputDescForward(const string &name, const unsigned int num);

  graphStatus AddOutputDescForward(const string &name, const unsigned int num);

  graphStatus AddOptionalInputDesc(const string &name, const GeTensorDesc &input_desc);

  graphStatus UpdateInputDesc(uint32_t index, const GeTensorDesc &tensor_desc);

  graphStatus UpdateInputDesc(const string &name, const GeTensorDesc &tensor_desc);

  bool InputIsSet(const string &name) const;

  GeTensorDesc GetInputDesc(uint32_t index) const;

  GeTensorDesc GetInputDesc(const string &name) const;

  Vistor<string> GetAllInputNames() const;

  GeTensorDescPtr MutableInputDesc(uint32_t index) const;

  Vistor<GeTensorDesc> GetAllInputsDesc() const;

  Vistor<GeTensorDescPtr> GetAllInputsDescPtr() const;

  size_t GetInputsSize() const;

  graphStatus AddOutputDesc(const GeTensorDesc &output_desc);

  graphStatus AddOutputDesc(const string &name, const GeTensorDesc &output_desc);

  graphStatus UpdateOutputDesc(uint32_t index, const GeTensorDesc &tensor_desc);

  graphStatus UpdateOutputDesc(const string &name, const GeTensorDesc &tensor_desc);

  GeTensorDesc GetOutputDesc(uint32_t index) const;

  GeTensorDesc GetOutputDesc(const string &name) const;

  GeTensorDescPtr MutableOutputDesc(uint32_t index) const;

  Vistor<GeTensorDesc> GetAllOutputsDesc() const;

  Vistor<GeTensorDescPtr> GetAllOutputsDescPtr() const;

  size_t GetOutputsSize() const;

  ConstGeTensorDescPtr GetOutputDescPtr(uint32_t index) const;

  ConstGeTensorDescPtr GetInputDescPtr(uint32_t index) const;

  graphStatus AddDynamicInputDesc(const string &name, const unsigned int num, bool isPushBack = true);

  graphStatus AddDynamicOutputDesc(const string &name, const unsigned int num, bool isPushBack = true);

  bool IsOptionalInput(const string &name) const;

  bool IsOptionalInput(uint32_t index) const;

  std::map<string, uint32_t> GetAllInputName();

  std::map<string, uint32_t> GetAllOutputName();

  bool UpdateInputName(std::map<string, uint32_t> inputNameIdx);

  bool UpdateOutputName(std::map<string, uint32_t> outputNameIdx);

  void AddInferFunc(const std::function<graphStatus(Operator &)> &func);

  std::function<graphStatus(Operator &)> GetInferFunc() const;

  graphStatus InferShapeAndType();

  void AddInferFormatFunc(const std::function<graphStatus(Operator &)> &func);

  std::function<graphStatus(Operator &)> GetInferFormatFunc() const;

  graphStatus DefaultInferFormat();

  std::function<graphStatus(Operator &)> GetVerifyFunc() const;

  void AddVerifierFunc(const std::function<graphStatus(Operator &)> &func);

  graphStatus CallInferFormatFunc(Operator &op);

  graphStatus OpVerify();

  graphStatus CommonVerify() const;

  using AttrHolder::AddRequiredAttr;
  using AttrHolder::DelAttr;
  using AttrHolder::GetAllAttrNames;
  using AttrHolder::GetAllAttrs;
  using AttrHolder::GetAttr;
  using AttrHolder::HasAttr;
  using AttrHolder::SetAttr;

  void SetId(int64_t id);
  int64_t GetId() const;
  void SetStreamId(int64_t stream_id);
  int64_t GetStreamId() const;
  void SetInputName(const vector<string> &input_name);
  vector<string> GetInputName() const;
  void SetSrcName(const vector<string> &src_name);
  vector<string> GetSrcName() const;
  void SetSrcIndex(const vector<int64_t> &src_index);
  vector<int64_t> GetSrcIndex() const;
  void SetInputOffset(const vector<int64_t> &input);
  vector<int64_t> GetInputOffset() const;
  void SetOutputOffset(const vector<int64_t> &input);
  vector<int64_t> GetOutputOffset() const;
  void SetDstName(const vector<string> &dst_name);
  vector<string> GetDstName() const;
  void SetDstIndex(const vector<int64_t> &dst_index);
  vector<int64_t> GetDstIndex() const;
  void SetWorkspace(const vector<int64_t> &workspace);
  vector<int64_t> GetWorkspace() const;
  void SetWorkspaceBytes(const vector<int64_t> &workspace_bytes);
  vector<int64_t> GetWorkspaceBytes() const;
  void SetIsInputConst(const vector<bool> &is_input_const);
  vector<bool> GetIsInputConst() const;

  string GetInputNameByIndex(uint32_t index) const;

  int GetInputIndexByName(const string &name) const;

  string GetOutputNameByIndex(uint32_t index) const;

  int GetOutputIndexByName(const string &name) const;

  graphStatus RestoreInputNameIdx(const string &name, const int &index);

  graphStatus RestoreOutputNameIdx(const string &name, const int &index);

  graphStatus CallInferFunc(Operator &op);

  void SetOpKernelLibName(const std::string &name);

  std::string GetOpKernelLibName() const;

  void SetOpEngineName(const std::string &name);

  std::string GetOpEngineName() const;

 protected:
  ProtoAttrMapHelper MutableAttrMap() override;
  ConstProtoAttrMapHelper GetAttrMap() const override;

 private:
  OpDesc(const ProtoMsgOwner &proto_msg_owner, ge::proto::OpDef *op_def);
  bool OpDescMembersAreEqual(const OpDesc &r_op_desc) const;
  bool OpDescAttrsAreEqual(const OpDesc &r_op_desc) const;
  bool OpDescGenTensorDescsAreEqual(const OpDesc &r_op_desc) const;

  GeIrProtoHelper<ge::proto::OpDef> op_def_;
  vector<GeTensorDescPtr> inputs_desc_{};
  map<string, uint32_t> input_name_idx_{};
  std::unordered_set<string> optional_input_names_{};
  vector<GeTensorDescPtr> outputs_desc_{};
  map<string, uint32_t> output_name_idx_{};
  std::function<graphStatus(Operator &)> infer_func_ = nullptr;
  std::function<graphStatus(Operator &)> infer_format_func_ = nullptr;
  std::function<graphStatus(Operator &)> verifier_func_ = nullptr;
  string op_kernel_lib_name_;
  string engine_name_;
  friend class OpDescUtils;
  friend class ModelSerializeImp;
  friend class AttrUtils;
  friend class GeAttrValueImp;
  friend class OnnxUtils;
};
}  // namespace ge
#endif  // INC_GRAPH_OP_DESC_H_
