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

#ifndef INC_EXTERNAL_GRAPH_OPERATOR_H_
#define INC_EXTERNAL_GRAPH_OPERATOR_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "./ge_error_codes.h"
#include "./inference_context.h"
#include "./tensor.h"

#ifndef USER_GE_LOGI
#define USER_GE_LOGI(...)
#endif  // USER_GE_LOGI

#ifndef USER_GE_LOGW
#define USER_GE_LOGW(...)
#endif  // USER_GE_LOGW

#ifndef USER_GE_LOGE
#define USER_GE_LOGE(...)
#endif  // USER_GE_LOGE

#define DYNAMIC_OUTPUT_TD_NUM(name) ("__dynamic_output_" + name + "_cnt")
#define DYNAMIC_INPUT_TD_NUM(name) ("__dynamic_input_" + name + "_cnt")

namespace ge {
class Operator;
class OperatorImpl;
class NodeUtils;
class NamedAttrs;
class Graph;
class AttrValue;
class Node;

using SubgraphBuilder = std::function<Graph()>;
using OperatorImplPtr = std::shared_ptr<OperatorImpl>;
using OperatorPtr = std::shared_ptr<Operator>;

class OpIO;
using OutHandler = std::shared_ptr<OpIO>;
using InHandler = std::shared_ptr<OpIO>;

using std::function;
using std::shared_ptr;
using std::string;

/*lint -e148*/
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Operator {
 public:
  friend class OperatorImpl;
  friend class GraphBuilderImpl;
  friend class NodeUtils;

  using OpInt = int64_t;
  using OpFloat = float;
  using OpString = string;
  using OpBool = bool;
  using OpTensor = Tensor;
  using OpType = ge::DataType;
  using OpNamedAttrs = ge::NamedAttrs;
  using OpListInt = std::vector<int64_t>;
  using OpListFloat = std::vector<float>;
  using OpListString = std::vector<string>;
  using OpListBool = std::vector<bool>;
  using OpListTensor = std::vector<Tensor>;
  using OpBytes = std::vector<uint8_t>;
  using OpListListInt = std::vector<std::vector<int64_t>>;
  using OpListType = std::vector<ge::DataType>;
  using OpListNamedAttrs = std::vector<ge::NamedAttrs>;

  Operator() {}

  explicit Operator(const string &type);

  Operator(const string &name, const string &type);  // lint !e148

  virtual ~Operator() = default;

  bool IsEmpty() const;

  string GetName() const;

  string GetOpType() const;

  // Only has one output index = 0
  Operator &SetInput(const string &dst_name, const Operator &src_oprt);

  Operator &SetInput(const string &dst_name, const Operator &src_oprt, const string &name);  // lint !e148

  Operator &SetInput(const string &dst_name, const Operator &src_oprt, uint32_t index);

  Operator &AddControlInput(const Operator &src_oprt);

  graphStatus GetInputConstData(const string &dst_name, Tensor &data) const;

  TensorDesc GetInputDesc(const string &name) const;

  TensorDesc GetInputDesc(uint32_t index) const;

  int GetDynamicOutputNum(const string &name) const;

  int GetDynamicInputNum(const string &name) const;

  graphStatus TryGetInputDesc(const string &name, TensorDesc &tensor_desc) const;

  graphStatus UpdateInputDesc(const string &name, const TensorDesc &tensor_desc);

  TensorDesc GetOutputDesc(const string &name) const;

  TensorDesc GetOutputDesc(uint32_t index) const;

  graphStatus UpdateOutputDesc(const string &name, const TensorDesc &tensor_desc);  // lint !e148

  TensorDesc GetDynamicInputDesc(const string &name, uint32_t index) const;

  graphStatus UpdateDynamicInputDesc(const string &name, uint32_t index, const TensorDesc &tensor_desc);  // lint !e148

  TensorDesc GetDynamicOutputDesc(const string &name, uint32_t index) const;

  graphStatus UpdateDynamicOutputDesc(const string &name, uint32_t index, const TensorDesc &tensor_desc);  // lint !e148

  graphStatus InferShapeAndType();  // lint !e148

  void SetInferenceContext(const InferenceContextPtr &inference_context);
  InferenceContextPtr GetInferenceContext() const;

  graphStatus VerifyAllAttr(bool disable_common_verifier = false);  // lint !e148

  size_t GetInputsSize() const;

  size_t GetOutputsSize() const;

  const std::map<std::string, std::string> GetAllAttrNamesAndTypes() const;

  Operator &SetAttr(const string &name, int64_t attr_value);
  Operator &SetAttr(const string &name, int32_t attr_value);
  Operator &SetAttr(const string &name, uint32_t attr_value);
  graphStatus GetAttr(const string &name, int64_t &attr_value) const;
  graphStatus GetAttr(const string &name, int32_t &attr_value) const;
  graphStatus GetAttr(const string &name, uint32_t &attr_value) const;
  Operator &SetAttr(const string &name, const std::vector<int64_t> &attr_value);
  Operator &SetAttr(const string &name, const std::vector<int32_t> &attr_value);
  Operator &SetAttr(const string &name, const std::vector<uint32_t> &attr_value);
  Operator &SetAttr(const string &name, std::initializer_list<int64_t> &&attr_value);
  graphStatus GetAttr(const string &name, std::vector<int64_t> &attr_value) const;
  graphStatus GetAttr(const string &name, std::vector<int32_t> &attr_value) const;
  graphStatus GetAttr(const string &name, std::vector<uint32_t> &attr_value) const;

  Operator &SetAttr(const string &name, float attr_value);
  graphStatus GetAttr(const string &name, float &attr_value) const;
  Operator &SetAttr(const string &name, const std::vector<float> &attr_value);
  graphStatus GetAttr(const string &name, std::vector<float> &attr_value) const;
  Operator &SetAttr(const string &name, AttrValue &&attr_value);
  graphStatus GetAttr(const string &name, AttrValue &attr_value) const;

  Operator &SetAttr(const string &name, const string &attr_value);
  graphStatus GetAttr(const string &name, string &attr_value) const;
  Operator &SetAttr(const string &name, const std::vector<string> &attr_value);
  graphStatus GetAttr(const string &name, std::vector<string> &attr_value) const;

  Operator &SetAttr(const string &name, bool attr_value);
  graphStatus GetAttr(const string &name, bool &attr_value) const;
  Operator &SetAttr(const string &name, const std::vector<bool> &attr_value);
  graphStatus GetAttr(const string &name, std::vector<bool> &attr_value) const;

  Operator &SetAttr(const string &name, const Tensor &attr_value);
  graphStatus GetAttr(const string &name, Tensor &attr_value) const;
  Operator &SetAttr(const string &name, const std::vector<Tensor> &attr_value);
  graphStatus GetAttr(const string &name, std::vector<Tensor> &attr_value) const;

  // Bytes type
  Operator &SetAttr(const string &name, const OpBytes &attr_value);
  // Bytes type
  graphStatus GetAttr(const string &name, OpBytes &attr_value) const;

  Operator &SetAttr(const string &name, const std::vector<std::vector<int64_t>> &attr_value);
  graphStatus GetAttr(const string &name, std::vector<std::vector<int64_t>> &attr_value) const;

  Operator &SetAttr(const string &name, const std::vector<ge::DataType> &attr_value);
  graphStatus GetAttr(const string &name, std::vector<ge::DataType> &attr_value) const;

  Operator &SetAttr(const string &name, const ge::DataType &attr_value);
  graphStatus GetAttr(const string &name, ge::DataType &attr_value) const;

  // func type
  Operator &SetAttr(const string &name, const ge::NamedAttrs &attr_value);
  graphStatus GetAttr(const string &name, ge::NamedAttrs &attr_value) const;
  Operator &SetAttr(const string &name, const std::vector<ge::NamedAttrs> &attr_value);
  graphStatus GetAttr(const string &name, std::vector<ge::NamedAttrs> &attr_value) const;

  void BreakConnect() const;

  size_t GetSubgraphNamesCount() const;
  std::vector<std::string> GetSubgraphNames() const;
  SubgraphBuilder GetSubgraphBuilder(const string &name) const;
  Graph GetSubgraph(const string &name) const;
  SubgraphBuilder GetDynamicSubgraphBuilder(const string &name, uint32_t index) const;
  Graph GetDynamicSubgraph(const string &name, uint32_t index) const;

 protected:
  void AttrRegister(const string &name, float attr_value);
  void AttrRegister(const string &name, const std::vector<float> &attr_value);
  void AttrRegister(const string &name, int64_t attr_value);
  void AttrRegister(const string &name, const std::vector<int64_t> &attr_value);
  void AttrRegister(const string &name, const string &attr_value);
  void AttrRegister(const string &name, const std::vector<string> &attr_value);
  void AttrRegister(const string &name, bool attr_value);
  void AttrRegister(const string &name, const std::vector<bool> &attr_value);
  void AttrRegister(const string &name, const Tensor &attr_value);
  void AttrRegister(const string &name, const std::vector<Tensor> &attr_value);
  void AttrRegister(const string &name, const OpBytes &attr_value);
  void AttrRegister(const string &name, const std::vector<std::vector<int64_t>> &attr_value);
  void AttrRegister(const string &name, const std::vector<ge::DataType> &attr_value);
  void AttrRegister(const string &name, const ge::DataType &attr_value);
  void AttrRegister(const string &name, const ge::NamedAttrs &attr_value);
  void AttrRegister(const string &name, const std::vector<ge::NamedAttrs> &attr_value);

  explicit Operator(OperatorImplPtr &&op_impl);

  void InputRegister(const string &name);

  void OptionalInputRegister(const string &name);

  void InferFuncRegister(const std::function<graphStatus(Operator &)> &func);

  void VerifierFuncRegister(const std::function<graphStatus(Operator &)> &func);

  void InferFormatFuncRegister(const std::function<graphStatus(Operator &)> &func);

  void OutputRegister(const string &name);

  void DynamicInputRegister(const string &name, const unsigned int num, bool is_push_back = true);

  void DynamicInputRegisterByIndex(const string &name, const unsigned int num, size_t index);

  void DynamicOutputRegister(const string &name, const unsigned int num, bool is_push_back = true);

  void RequiredAttrRegister(const string &name);

  graphStatus VerifyAll();  // lint !e148

  // Only has one output index = 0
  Operator &SetInput(const string &dst_name, uint32_t dst_index, const Operator &src_oprt);

  Operator &SetInput(const string &dst_name, uint32_t dst_index, const Operator &src_oprt,
                     const string &name);  // lint !e148

  void SubgraphRegister(const string &ir_name, bool dynamic);
  void SubgraphCountRegister(const string &ir_name, uint32_t count);
  void SetSubgraphBuilder(const string &ir_name, uint32_t index, const SubgraphBuilder &builder);

 private:
  Operator &SetInput(const string &dst_name, const OutHandler &out_handler);  // lint !e148

  OutHandler GetOutput(const string &name) const;

  OutHandler GetOutput(uint32_t index) const;

  OperatorImplPtr GetOperatorImplPtr() const;

  OperatorImplPtr operator_impl_{nullptr};

  graphStatus GetInputConstDataOut(const string &dst_name, Tensor &data) const;

  std::shared_ptr<const Node> GetNode() const;
};
/*lint +e148*/
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_OPERATOR_H_
