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

#ifndef COMMON_GRAPH_UTILS_GE_IR_UTILS_H_
#define COMMON_GRAPH_UTILS_GE_IR_UTILS_H_

#include <google/protobuf/map.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/stubs/port.h>

#include <graph/anchor.h>
#include <graph/debug/ge_log.h>
#include <graph/debug/ge_util.h>
#include <graph/detail/attributes_holder.h>
#include <graph/ge_tensor.h>
#include <graph/graph.h>
#include <graph/model.h>
#include <graph/node.h>
#include <graph/utils/graph_utils.h>
#include <graph/utils/type_utils.h>

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "proto/ge_ir.pb.h"
#include "proto_inner/ge_onnx.pb.h"

namespace ge {
const int kOffsetToString = 2;

///
///  @ingroup ge_ir_utils
///  @brief RepeatedField->String
///  @param [in] const rpd_field  RepeatedField
///  @return String
///
template <typename T>
const std::string ToString(const google::protobuf::RepeatedField<T> &rpd_field) {
  std::stringstream ss;
  ss << "[";
  for (const T &x : rpd_field) {
    ss << x;
    ss << ", ";
  }
  std::string str_ret = ss.str().substr(0, ss.str().length() - kOffsetToString);
  str_ret += "]";
  return str_ret;
}

///
///  @ingroup ge_ir_utils
///  @brief RepeatedPtrField->String
///  @param [in] const rpd_field  RepeatedPtrField
///  @return String
///
template <typename T>
const std::string ToString(const google::protobuf::RepeatedPtrField<T> &rpd_ptr_field) {
  std::stringstream ss;
  ss << "[";
  for (const T &x : rpd_ptr_field) {
    ss << x;
    ss << ", ";
  }
  std::string str_ret = ss.str().substr(0, ss.str().length() - kOffsetToString);
  str_ret += "]";
  return str_ret;
}

///
///  @ingroup ge_ir_utils
///  @brief check, if not equal, log with tag
///  @param [in] const left_value, right_value reference, log_info_tag
///  @return bool
///
template <typename T>
bool IsEqual(const T &l_value, const T &r_value, const std::string &log_info_tag) {
  if (l_value == r_value) {
    return true;
  } else {
    GELOGE(GRAPH_FAILED, "Check failed with %s", log_info_tag.c_str());
    return false;
  }
}

class OnnxUtils {
 public:
  enum DumpLevel { NO_DUMP = 0, DUMP_ALL = 1, DUMP_WITH_OUT_DATA = 2, DUMP_WITH_OUT_DESC = 3, DUMP_LEVEL_END };

  static bool ConvertGeModelToModelProto(const ge::Model &model, ge::onnx::ModelProto &model_proto);

  static bool ConvertModelProtoToGeModel(const ge::onnx::ModelProto &model_proto, ge::Model &model);

 private:
  // Part 1: from IR convert to ONNX Protobuf
  static void AddAttrProto(ge::onnx::NodeProto *node_proto, ge::onnx::AttributeProto_AttributeType type,
                           const std::string &name, void *data);

  static void AddAttrProto(ge::onnx::NodeProto *node_proto, ge::onnx::AttributeProto_AttributeType type,
                           const std::string &name, ::google::protobuf::RepeatedField<::google::protobuf::int64> data);

  static void AddAttrProto(ge::onnx::NodeProto *node_proto, ge::onnx::AttributeProto_AttributeType type,
                           const std::string &name, ::google::protobuf::RepeatedField<bool> data);

  static void AddAttrProto(ge::onnx::NodeProto *node_proto, ge::onnx::AttributeProto_AttributeType type,
                           const std::string &name, ::google::protobuf::RepeatedField<float> data);

  static void AddAttrProto(ge::onnx::NodeProto *node_proto, ge::onnx::AttributeProto_AttributeType type,
                           const std::string &name, ::google::protobuf::RepeatedPtrField<::std::string> data);

  static void AddAttrProtoFromNodeMembers(const NodePtr &node, ge::onnx::NodeProto *node_proto);

  static void AddAttrProtoFromAttribute(const std::pair<const std::string, ge::GeAttrValue> &string_attr_value,
                                        ge::onnx::NodeProto *node_proto);

  static void AddAttrProtoForOpInAndOutDesc(ge::onnx::NodeProto *node_proto, const OpDescPtr &op_desc);

  static void AddAttrProtoForAttrsFromAttrMap(const ::google::protobuf::Map<std::string,
                                                                            ge::proto::AttrDef> &attr_map,
                                              ge::onnx::NodeProto *node_proto,
                                              const std::string& prefix = "",
                                              const std::string& suffix = "");

  static void AddAttrProtoForAttrsFromOpDef(const ge::proto::OpDef *op_def, ge::onnx::NodeProto *node_proto);

  static ge::onnx::TensorProto_DataType EncodeDataType(ge::DataType data_type);

  static void EncodeNodeLinkForNetronVisual(const NodePtr &node, ge::onnx::NodeProto *node_proto);

  static bool EncodeNodeLink(const NodePtr &node, ge::onnx::NodeProto *node_proto);

  static bool EncodeNodeDesc(const NodePtr &node, ge::onnx::NodeProto *node_proto);

  static bool EncodeNode(const NodePtr &node, ge::onnx::NodeProto *node_proto);

  static void EncodeTypeProtoTensorType(const NodePtr &node, ge::onnx::TypeProto_Tensor *tensor_type);

  static void EncodeValueInfo(const NodePtr &n, ge::onnx::ValueInfoProto *v);

  static bool EncodeGraph(const ConstComputeGraphPtr &graph, ge::onnx::GraphProto *graph_proto);

  /// Part 2: from ONNX Protobuf convert to IR
  /// Describes node's link relationships
  struct NodeLinkInfo {
    std::string src_node_name;
    int32_t src_out_index;
    NodePtr dst_node;
    int32_t dst_in_index;
    std::string dst_node_name;
  };

  // Parse node name and index
  static bool ParseNameIndex(const std::string &node_name_index, std::string &node_name, int32_t &index);

  static ge::DataType DecodeDataType(ge::onnx::TensorProto_DataType data_type);

  static void DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, std::vector<std::string> &strings);

  static void DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, std::vector<int64_t> &ints);

  static void DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, int64_t &value);

  static void DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, std::string &value);

  static void DecodeNodeAttributeForOpOutDesc(const ge::onnx::AttributeProto &attr_proto,
                                              const std::string &attr_name_for_output_desc, int32_t index,
                                              OpDescPtr &op_desc);

  static void DecodeNodeAttributeForOpInDesc(const ge::onnx::AttributeProto &attr_proto,
                                             const std::string &attr_name_for_input_desc, int32_t index,
                                             OpDescPtr &op_desc);

  static void DecodeNodeAttributeForOpInAndOutDesc(const ge::onnx::AttributeProto &attr_proto,
                                                   const std::string &attr_name_for_input_output_desc, int32_t index,
                                                   OpDescPtr &op_desc);

  static void DecodeNodeAttributeForOpDef(const ge::onnx::AttributeProto &attr_proto, ge::proto::OpDef &op_def);

  static void DecodeNodeAttributeForOpDesc(const ge::onnx::AttributeProto &attr_proto, OpDescPtr &op_desc);

  static bool DecodeNodeLinkImp(const NodeLinkInfo &item, NodePtr &node_ptr);

  static bool DecodeNodeLink(const std::vector<ge::onnx::NodeProto> &node_proto_vector,
                             const std::map<std::string, NodePtr> &node_map);

  static bool DecodeNodeDesc(const ge::onnx::NodeProto *node_proto, OpDescPtr &node);

  static bool DecodeGraph(int recursion_depth, const ge::onnx::GraphProto &graph_proto, ComputeGraphPtr &graph);
};
}  // namespace ge

#endif  // COMMON_GRAPH_UTILS_GE_IR_UTILS_H_
