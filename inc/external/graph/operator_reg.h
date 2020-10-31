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

#ifndef INC_EXTERNAL_GRAPH_OPERATOR_REG_H_
#define INC_EXTERNAL_GRAPH_OPERATOR_REG_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "graph/graph.h"

namespace ge {
using std::function;
using std::string;
using std::vector;

class OpReg {
 public:
  OpReg &N() { return *this; }

  OpReg &ATTR() { return *this; }

  OpReg &REQUIRED_ATTR() { return *this; }

  OpReg &INPUT() { return *this; }

  OpReg &OPTIONAL_INPUT() { return *this; }

  OpReg &OUTPUT() { return *this; }

  OpReg &GRAPH() { return *this; }

  OpReg &DYNAMIC_GRAPH() { return *this; }

  OpReg &INFER_SHAPE_AND_TYPE() { return *this; }
};

#define REG_OP(x)                                                    \
  namespace op {                                                     \
  class x : public Operator {                                        \
    typedef x _THIS_TYPE;                                            \
                                                                     \
   public:                                                           \
    explicit x(const string &name) : Operator(name, #x) { __##x(); } \
    x() : Operator(#x) { __##x(); }                                  \
                                                                     \
   private:                                                          \
    void __##x() {                                                   \
      OpReg()

#define ATTR(x, Type, ...)                                                  \
  N();                                                                      \
  __attr_##x();                                                             \
  }                                                                         \
                                                                            \
 public:                                                                    \
  static const string name_attr_##x() { return #x; }                        \
  Op##Type get_attr_##x() const {                                           \
    Op##Type ret = __VA_ARGS__;                                             \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      return ret;                                                           \
    }                                                                       \
    return ret;                                                             \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const Op##Type &v) {                             \
    Operator::SetAttr(#x, v);                                               \
    return *this;                                                           \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const function<Op##Type()> &v) { return *this; } \
                                                                            \
 private:                                                                   \
  void __attr_##x() {                                                       \
    Operator::AttrRegister(#x, Op##Type(__VA_ARGS__));                      \
    string attr_name(#x);                                                   \
    (void)OpReg()

#define REQUIRED_ATTR(x, Type)                                              \
  N();                                                                      \
  __required_attr_##x();                                                    \
  }                                                                         \
                                                                            \
 public:                                                                    \
  static const string name_attr_##x() { return #x; }                        \
  Op##Type get_attr_##x() const {                                           \
    Op##Type ret;                                                           \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      return ret;                                                           \
    }                                                                       \
    return ret;                                                             \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const Op##Type &v) {                             \
    Operator::SetAttr(#x, v);                                               \
    return *this;                                                           \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const function<Op##Type()> &v) { return *this; } \
                                                                            \
 private:                                                                   \
  void __required_attr_##x() {                                              \
    Operator::RequiredAttrRegister(#x);                                     \
    string attr_name(#x);                                                   \
    (void)OpReg()

#define INPUT(x, t)                                                            \
  N();                                                                         \
  __input_##x();                                                               \
  }                                                                            \
                                                                               \
 public:                                                                       \
  static const string name_in_##x() { return #x; }                             \
  _THIS_TYPE &set_input_##x(Operator &v, const string &srcName) {              \
    Operator::SetInput(#x, v, srcName);                                        \
    return *this;                                                              \
  }                                                                            \
  _THIS_TYPE &set_input_##x(Operator &v, uint32_t index) {                     \
    Operator::SetInput(#x, v, index);                                          \
    return *this;                                                              \
  }                                                                            \
  _THIS_TYPE &set_input_##x(Operator &v) {                                     \
    Operator::SetInput(#x, v);                                                 \
    return *this;                                                              \
  }                                                                            \
  TensorDesc get_input_desc_##x() const { return Operator::GetInputDesc(#x); } \
  graphStatus update_input_desc_##x(const TensorDesc &tensorDesc) {            \
    return Operator::UpdateInputDesc(#x, tensorDesc);                          \
  }                                                                            \
                                                                               \
 private:                                                                      \
  void __input_##x() {                                                         \
    Operator::InputRegister(#x);                                               \
    (void)OpReg()

#define OPTIONAL_INPUT(x, t)                                                   \
  N();                                                                         \
  __optional_input_##x();                                                      \
  }                                                                            \
                                                                               \
 public:                                                                       \
  static const string name_in_##x() { return #x; }                             \
  _THIS_TYPE &set_input_##x(Operator &v) {                                     \
    Operator::SetInput(#x, v);                                                 \
    return *this;                                                              \
  }                                                                            \
  _THIS_TYPE &set_input_##x(Operator &v, const string &srcName) {              \
    Operator::SetInput(#x, v, srcName);                                        \
    return *this;                                                              \
  }                                                                            \
  _THIS_TYPE &set_input_##x(Operator &v, uint32_t index) {                     \
    Operator::SetInput(#x, v, index);                                          \
    return *this;                                                              \
  }                                                                            \
  TensorDesc get_input_desc_##x() const { return Operator::GetInputDesc(#x); } \
  graphStatus update_input_desc_##x(const TensorDesc &tensorDesc) {            \
    return Operator::UpdateInputDesc(#x, tensorDesc);                          \
  }                                                                            \
                                                                               \
 private:                                                                      \
  void __optional_input_##x() {                                                \
    Operator::OptionalInputRegister(#x);                                       \
    (void)OpReg()

#define OUTPUT(x, t)                                                             \
  N();                                                                           \
  __out_##x();                                                                   \
  }                                                                              \
                                                                                 \
 public:                                                                         \
  static const string name_out_##x() { return #x; }                              \
  TensorDesc get_output_desc_##x() const { return Operator::GetOutputDesc(#x); } \
  graphStatus update_output_desc_##x(const TensorDesc &tensorDesc) {             \
    return Operator::UpdateOutputDesc(#x, tensorDesc);                           \
  }                                                                              \
                                                                                 \
 private:                                                                        \
  void __out_##x() {                                                             \
    Operator::OutputRegister(#x);                                                \
    (void)OpReg()

#define DYNAMIC_INPUT(x, t)                                                                                        \
  N();                                                                                                             \
  __dy_input_##x();                                                                                                \
  }                                                                                                                \
                                                                                                                   \
 public:                                                                                                           \
  _THIS_TYPE &create_dynamic_input_##x(uint32_t num, bool isPushBack = true) {                                     \
    Operator::DynamicInputRegister(#x, num, isPushBack);                                                           \
    return *this;                                                                                                  \
  }                                                                                                                \
  _THIS_TYPE &create_dynamic_input_byindex_##x(uint32_t num, size_t index) {                                       \
    Operator::DynamicInputRegisterByIndex(#x, num, index);                                                         \
    return *this;                                                                                                  \
  }                                                                                                                \
  TensorDesc get_dynamic_input_desc_##x(uint32_t index) const { return Operator::GetDynamicInputDesc(#x, index); } \
  graphStatus update_dynamic_input_desc_##x(uint32_t index, const TensorDesc &tensorDesc) {                        \
    return Operator::UpdateDynamicInputDesc(#x, index, tensorDesc);                                                \
  }                                                                                                                \
  _THIS_TYPE &set_dynamic_input_##x(uint32_t dstIndex, Operator &v) {                                              \
    Operator::SetInput(#x, dstIndex, v);                                                                           \
    return *this;                                                                                                  \
  }                                                                                                                \
  _THIS_TYPE &set_dynamic_input_##x(uint32_t dstIndex, Operator &v, const string &srcName) {                       \
    Operator::SetInput(#x, dstIndex, v, srcName);                                                                  \
    return *this;                                                                                                  \
  }                                                                                                                \
                                                                                                                   \
 private:                                                                                                          \
  void __dy_input_##x() {                                                                                          \
    Operator::DynamicInputRegister(#x, 0, true);                                                                   \
    (void)OpReg()

#define DYNAMIC_OUTPUT(x, t)                                                                                         \
  N();                                                                                                               \
  __dy_output_##x();                                                                                                 \
  }                                                                                                                  \
                                                                                                                     \
 public:                                                                                                             \
  _THIS_TYPE &create_dynamic_output_##x(uint32_t num, bool isPushBack = true) {                                      \
    Operator::DynamicOutputRegister(#x, num, isPushBack);                                                            \
    return *this;                                                                                                    \
  }                                                                                                                  \
  TensorDesc get_dynamic_output_desc_##x(uint32_t index) const { return Operator::GetDynamicOutputDesc(#x, index); } \
  graphStatus update_dynamic_output_desc_##x(uint32_t index, const TensorDesc &tensorDesc) {                         \
    return Operator::UpdateDynamicOutputDesc(#x, index, tensorDesc);                                                 \
  }                                                                                                                  \
                                                                                                                     \
 private:                                                                                                            \
  void __dy_output_##x() {                                                                                           \
    Operator::DynamicOutputRegister(#x, 0, true);                                                                    \
    (void)OpReg()

#define GRAPH(x)                                                                                \
  N();                                                                                          \
  __graph_##x();                                                                                \
  }                                                                                             \
                                                                                                \
 public:                                                                                        \
  static const string name_graph_##x() { return #x; }                                           \
  SubgraphBuilder get_subgraph_builder_##x() const { return Operator::GetSubgraphBuilder(#x); } \
  _THIS_TYPE &set_subgraph_builder_##x(const SubgraphBuilder &v) {                              \
    Operator::SetSubgraphBuilder(#x, 0, v);                                                     \
    return *this;                                                                               \
  }                                                                                             \
  Graph get_subgraph_##x() const { return Operator::GetSubgraph(#x); }                          \
                                                                                                \
 private:                                                                                       \
  void __graph_##x() {                                                                          \
    Operator::SubgraphRegister(#x, false);                                                      \
    Operator::SubgraphCountRegister(#x, 1);                                                     \
    (void)OpReg()

#define DYNAMIC_GRAPH(x)                                                                                   \
  N();                                                                                                     \
  __graph_##x();                                                                                           \
  }                                                                                                        \
                                                                                                           \
 public:                                                                                                   \
  static const string name_graph_##x() { return #x; }                                                      \
  _THIS_TYPE &create_dynamic_subgraph_##x(uint32_t num) {                                                  \
    Operator::SubgraphCountRegister(#x, num);                                                              \
    return *this;                                                                                          \
  }                                                                                                        \
  SubgraphBuilder get_dynamic_subgraph_builder_##x(uint32_t index) const {                                 \
    return Operator::GetDynamicSubgraphBuilder(#x, index);                                                 \
  }                                                                                                        \
  Graph get_dynamic_subgraph_##x(uint32_t index) const { return Operator::GetDynamicSubgraph(#x, index); } \
  _THIS_TYPE &set_dynamic_subgraph_builder_##x(uint32_t index, const SubgraphBuilder &v) {                 \
    Operator::SetSubgraphBuilder(#x, index, v);                                                            \
    return *this;                                                                                          \
  }                                                                                                        \
                                                                                                           \
 private:                                                                                                  \
  void __graph_##x() {                                                                                     \
    Operator::SubgraphRegister(#x, true);                                                                  \
    (void)OpReg()

#define PASTE(g_register, y) g_register##y
#define __OP_END_IMPL__(x, y)                                                                                     \
  N();                                                                                                            \
  }                                                                                                               \
  static_assert(                                                                                                  \
    std::is_same<x, _THIS_TYPE>::value,                                                                           \
    "The class name entered into the OP_END_FACTORY_REG needs to be the same as the operator name you define.");  \
  }                                                                                                               \
  ;                                                                                                               \
  static const OperatorCreatorRegister PASTE(g_register, y)(#x, [](const std::string &name) { return x(name); }); \
  }
#define OP_END_FACTORY_REG(x) __OP_END_IMPL__(x, __COUNTER__)

// Specialized shape inferencer macro

#define IMPLEMT_INFERFUNC(op_name, func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus func_name(op::op_name &op)

#define IMPLEMT_COMMON_INFERFUNC(func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus func_name(Operator &op)

#define IMPLEMT_INFERFORMAT_FUNC(op_name, func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus func_name(op::op_name &op)

// Specialized verifier macro

#define IMPLEMT_VERIFIER(op_name, func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus func_name(op::op_name op)

#define INFER_VERIFY_FUNC(op_name, x) [&](Operator &v) { return x((op::op_name &)v); }

#define COMMON_INFER_VERIFY_FUNC(x) [&](Operator &v) { return x(v); }

#define INFER_FORMAT_FUNC(op_name, x) [&](Operator &v) { return x((op::op_name &)v); }

#define __INFER_FUNC_REG_IMPL__(op_name, x, n) static const InferShapeFuncRegister PASTE(if_register, n)(#op_name, x)

#define __VERIFY_FUNC_REG_IMPL__(op_name, x, n) static const VerifyFuncRegister PASTE(vf_register, n)(#op_name, x)
// Infer format func register
#define __INFER_FORMAT_FUNC_REG_IMPL__(op_name, x, n) \
  static const InferFormatFuncRegister PASTE(ff_register, n)(#op_name, x)

// Shape inferencer & verifier register macro

#define INFER_FUNC_REG(op_name, x) __INFER_FUNC_REG_IMPL__(op_name, INFER_VERIFY_FUNC(op_name, x), __COUNTER__)

#define COMMON_INFER_FUNC_REG(op_name, x) __INFER_FUNC_REG_IMPL__(op_name, COMMON_INFER_VERIFY_FUNC(x), __COUNTER__)

#define VERIFY_FUNC_REG(op_name, x) __VERIFY_FUNC_REG_IMPL__(op_name, INFER_VERIFY_FUNC(op_name, x), __COUNTER__)

// Infer format func reg
#define INFER_FORMAT_FUNC_REG(op_name, x) \
  __INFER_FORMAT_FUNC_REG_IMPL__(op_name, INFER_FORMAT_FUNC(op_name, x), __COUNTER__)

// Common shape inferencer

#define ELMTWISE_INFER_SHAPEANDTYPE(in_name, out_name)            \
  [](Operator op) -> graphStatus {                                \
    auto x_shape = op.GetInputDesc(in_name).GetShape().GetDims(); \
    auto x_type = op.GetInputDesc(in_name).GetDataType();         \
    TensorDesc op_output_desc = op.GetOutputDesc(out_name);       \
    op_output_desc.SetShape(ge::Shape(x_shape));                  \
    op_output_desc.SetOriginShape(ge::Shape(x_shape));            \
    op_output_desc.SetDataType(x_type);                           \
    return op.UpdateOutputDesc(out_name, op_output_desc);         \
  }

graphStatus BroadCastInfer(const function<vector<int64_t>()> &get_in1_shape,
                           const function<vector<int64_t>()> &get_in2_shape,
                           const function<void(const vector<int64_t> &y_shape)> &set_out_shape);

#define BROADCAST_INFER(in1_name, in2_name, out_name)                                       \
  [](Operator op) -> graphStatus {                                                          \
    return BroadCastInfer([&]() { return op.GetInputDesc(in1_name).GetShape().GetDims(); }, \
                          [&]() { return op.GetInputDesc(in2_name).GetShape().GetDims(); }, \
                          [&](const vector<int64_t> &y_shape) {                             \
                            TensorDesc op_output_desc = op.GetOutputDesc(out_name);         \
                            op_output_desc.SetShape(ge::Shape(y_shape));                    \
                            (void)op.UpdateOutputDesc(out_name, op_output_desc);            \
                          });                                                               \
  }
}  // namespace ge
#endif  // INC_EXTERNAL_GRAPH_OPERATOR_REG_H_
