/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef INC_GRAPH_UTILS_OP_DESC_UTILS_H_
#define INC_GRAPH_UTILS_OP_DESC_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include "graph/def_types.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/range_vistor.h"

namespace ge {
class OpDesc;
using OpDescPtr = std::shared_ptr<OpDesc>;

class OpDescUtils {
 public:
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<OpDesc>>;

  OpDescUtils() = default;
  ~OpDescUtils() = default;
  static bool HasQuantizeFactorParams(const OpDescPtr& op_desc);
  static bool HasQuantizeFactorParams(const OpDesc& op_desc);
  static graphStatus GetQuantizeFactorParams(const OpDescPtr& op_desc, QuantizeFactorParams& quant);
  static graphStatus GetQuantizeFactorParams(const OpDesc& op_desc, QuantizeFactorParams& quant);
  static graphStatus SetQuantizeFactorParams(const OpDescPtr &op_desc, const QuantizeFactorParams& quant);
  static graphStatus SetQuantizeFactorParams(OpDesc& op_desc, const QuantizeFactorParams& quant);

  static vector<ge::NodePtr> GetConstInputNode(const ge::Node& node);
  static vector<ConstGeTensorPtr> GetInputData(const vector<ge::NodePtr>& input_nodes);

  static vector<ConstGeTensorPtr> GetWeights(const ge::Node& node);
  static vector<ConstGeTensorPtr> GetWeights(const ge::ConstNodePtr& node);
  static vector<GeTensorPtr> MutableWeights(const ge::Node& node);
  static vector<GeTensorPtr> MutableWeights(const ge::NodePtr node);
  static graphStatus SetWeights(ge::Node& node, const vector<ge::GeTensorPtr>& weights);
  static graphStatus SetWeights(ge::NodePtr node, const vector<ge::GeTensorPtr> &weights);
  static graphStatus SetWeights(ge::Node &node, const map<int, ge::GeTensorPtr> &weights_map);
  static graphStatus ClearWeights(ge::NodePtr node);

  static bool ClearInputDesc(ge::OpDescPtr op_desc, uint32_t index);
  static bool ClearInputDesc(const ge::NodePtr& node);
  static bool ClearOutputDesc(const ge::OpDescPtr& op_desc, uint32_t index);
  static bool ClearOutputDesc(const ge::NodePtr& node);
  static vector<ge::NodePtr> GetConstInputs(const ge::Node& node);
  static vector<ge::NodePtr> GetConstInputs(const ge::ConstNodePtr& node);
  static size_t GetNonConstInputsSize(const ge::Node& node);
  static size_t GetNonConstInputsSize(ge::ConstNodePtr node);
  // Index: Indicates the index of all non const inputs
  static GeTensorDesc GetNonConstInputTensorDesc(const ge::Node& node, size_t index_non_const = 0);
  static GeTensorDesc GetNonConstInputTensorDesc(const ge::ConstNodePtr& node, size_t index_non_const = 0);
  static bool GetNonConstInputIndex(const ge::Node& node, size_t index_non_const, size_t& index);
  static bool GetNonConstInputIndex(const ge::ConstNodePtr& node, size_t index_non_const, size_t& index);
  // Index: Indicates the index of all inputs
  static bool IsNonConstInput(const ge::Node& node, size_t index = 0);
  static bool IsNonConstInput(const ge::ConstNodePtr& node, size_t index = 0);

  static vector<ge::GeTensorDesc> GetNonConstTensorDesc(const ge::ConstNodePtr& node);
  static graphStatus AddConstOpToAnchor(InDataAnchorPtr in_anchor, const GeTensorPtr& tensor_ptr);

  static Operator CreateOperatorFromOpDesc(OpDescPtr op_desc);
  static Operator CreateOperatorFromNode(ge::ConstNodePtr node_ptr);
  static OpDescPtr GetOpDescFromOperator(const Operator& oprt);

  static OpDescPtr CreateConstOp(const GeTensorPtr& tensor_ptr);

  static graphStatus SetSubgraphInstanceName(const std::string &subgraph_name,
      const std::string &subgraph_instance_name, OpDescPtr &op_desc);

 private:
  static GeTensorPtr MutableWeights(ge::OpDesc& op_desc);
  static GeTensorPtr MutableWeights(ge::OpDescPtr op_desc);
  static graphStatus SetWeights(ge::OpDesc& op_desc, const GeTensorPtr weight);
  static graphStatus SetWeights(ge::OpDescPtr op_desc, const GeTensorPtr weight);
};

class OpDescBuilder {
 public:
  OpDescBuilder(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)) {}
  OpDescBuilder(const OpDescBuilder &) = delete;
  OpDescBuilder &operator=(const OpDescBuilder &) = delete;
  OpDescBuilder(const OpDescBuilder &&) = delete;
  OpDescBuilder &operator=(const OpDescBuilder &&) = delete;
  ~OpDescBuilder() = default;

  ///
  /// @brief Add input
  /// @param [in] name
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddInput(const std::string &name);

  ///
  /// @brief Add input
  /// @param [in] name
  /// @param [in] tensor
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddInput(const std::string &name, const GeTensorDesc &tensor);

  ///
  /// @brief Add dynamic input
  /// @param [in] name
  /// @param [in] num
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddDynamicInput(const std::string &name, uint32_t num);

  ///
  /// @brief Add dynamic input
  /// @param [in] name
  /// @param [in] num
  /// @param [in] tensor
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddDynamicInput(const std::string &name, uint32_t num, const GeTensorDesc &tensor);

  ///
  /// @brief Add output
  /// @param [in] name
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddOutput(const std::string &name);

  ///
  /// @brief Add output
  /// @param [in] name
  /// @param [in] tensor
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddOutput(const std::string &name, const GeTensorDesc &tensor);

  ///
  /// @brief Add dynamic output
  /// @param [in] name
  /// @param [in] num
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddDynamicOutput(const std::string &name, uint32_t num);

  ///
  /// @brief Add dynamic output
  /// @param [in] name
  /// @param [in] num
  /// @param [in] tensor
  /// @return OpDescBuilder
  ///
  OpDescBuilder& AddDynamicOutput(const std::string &name, uint32_t num, const GeTensorDesc &tensor);

  ///
  /// @brief Build op_desc
  /// @return OpDescPtr
  ///
  OpDescPtr Build();

 private:
  std::string name_;
  std::string type_;
  std::vector<std::pair<std::string, GeTensorDesc>> inputs_;
  std::vector<std::pair<std::string, GeTensorDesc>> outputs_;
};
}  // namespace ge

#endif  // INC_GRAPH_UTILS_OP_DESC_UTILS_H_
