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

#include "graph/utils/ge_ir_utils.h"
#include <utility>
#include "framework/common/debug/ge_log.h"

namespace {
const char *const kControlAnchorIndex = ":-1";
const char *const kNodeTypeForSubgraph = "subgraph";
const char *const kPrefixForInputDesc = "input_desc_attr_";
const char *const kPrefixForOutputDesc = "output_desc_attr_";
const char *const kDumpGEGraph = "DUMP_GE_GRAPH";
const int8_t kMaxRecursionDepth = 10;
const char *const kDumpGeGraph = std::getenv(kDumpGEGraph);
const int64_t kDumpLevel = (kDumpGeGraph != nullptr) ? std::strtol(kDumpGeGraph, nullptr, 10) : ge::OnnxUtils::NO_DUMP;
const int64_t kInputPrefixLength = 5;
const int64_t kOutputPrefixLength = 6;
using AttrDefPair = ::google::protobuf::MapPair<std::string, ge::proto::AttrDef>;
}  // namespace

namespace ge {
// Part 1: from IR convert to ONNX Protobuf
static const std::map<ge::DataType, onnx::TensorProto_DataType> kGeDataTypeToOnnxMap = {
  {DT_INT64, onnx::TensorProto_DataType_INT64},   {DT_UINT64, onnx::TensorProto_DataType_UINT64},
  {DT_FLOAT, onnx::TensorProto_DataType_FLOAT},   {DT_INT32, onnx::TensorProto_DataType_INT32},
  {DT_UINT32, onnx::TensorProto_DataType_UINT32}, {DT_INT8, onnx::TensorProto_DataType_INT8},
  {DT_UINT8, onnx::TensorProto_DataType_UINT8},   {DT_INT16, onnx::TensorProto_DataType_INT16},
  {DT_UINT16, onnx::TensorProto_DataType_UINT16}, {DT_FLOAT16, onnx::TensorProto_DataType_FLOAT16},
  {DT_DOUBLE, onnx::TensorProto_DataType_DOUBLE}, {DT_BOOL, onnx::TensorProto_DataType_BOOL},
};

onnx::TensorProto_DataType OnnxUtils::EncodeDataType(DataType data_type) {
  auto it = kGeDataTypeToOnnxMap.find(data_type);
  if (it != kGeDataTypeToOnnxMap.end()) {
    return it->second;
  } else {
    GELOGW("EncodeDataType: datatype not support %u", data_type);
    return onnx::TensorProto_DataType_UNDEFINED;
  }
}

void OnnxUtils::AddAttrProtoFromAttribute(const std::pair<const std::string, ge::GeAttrValue> &string_attr_value,
                                          onnx::NodeProto *node_proto) {
  if (node_proto == nullptr) {
    GELOGE(FAILED, "Node proto is nullptr.");
    return;
  }
  auto attr = node_proto->add_attribute();
  if (attr == nullptr) {
    GELOGE(GRAPH_FAILED, "attr is nullptr.");
    return;
  }
  auto attr_name = string_attr_value.first;
  attr->set_name(attr_name);
  auto attr_value = string_attr_value.second;
  auto value_type = attr_value.GetValueType();
  switch (value_type) {
    case GeAttrValue::VT_FLOAT: {
      GeAttrValue::FLOAT data_f = 0;
      (void)attr_value.GetValue(data_f);
      attr->set_f(data_f);
      attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
      break;
    }
    case GeAttrValue::VT_LIST_FLOAT: {
      GeAttrValue::LIST_FLOAT data_fs = {};
      (void)attr_value.GetValue(data_fs);
      attr->set_type(onnx::AttributeProto_AttributeType_FLOATS);
      for (auto &v : data_fs) {
        attr->add_floats(v);
      }
      break;
    }
    case GeAttrValue::VT_INT: {
      GeAttrValue::INT data_i = 0;
      (void)attr_value.GetValue(data_i);
      attr->set_type(onnx::AttributeProto_AttributeType_INT);
      attr->set_i(data_i);
      break;
    }
    case GeAttrValue::VT_LIST_INT: {
      GeAttrValue::LIST_INT data_is = {};
      (void)attr_value.GetValue(data_is);
      attr->set_type(onnx::AttributeProto_AttributeType_INTS);
      for (auto &v : data_is) {
        attr->add_ints(v);
      }
      break;
    }
    case GeAttrValue::VT_STRING: {
      GeAttrValue::STR data_s;
      (void)attr_value.GetValue(data_s);
      attr->set_type(onnx::AttributeProto_AttributeType_STRING);
      attr->set_s(data_s);
      break;
    }
    case GeAttrValue::VT_LIST_STRING: {
      GeAttrValue::LIST_STR data_ss = {};
      (void)attr_value.GetValue(data_ss);
      attr->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      for (auto &v : data_ss) {
        attr->add_strings(v);
      }
      break;
    }
    default:
      GELOGW("GeAttrValue ValueType: %u is not supported for now", value_type);
      break;
  }
}

void OnnxUtils::AddAttrProto(onnx::NodeProto *node_proto, onnx::AttributeProto_AttributeType type, const string &name,
                             void *data) {
  if (node_proto == nullptr) {
    GELOGE(FAILED, "Node_proto %s is nullptr.", name.c_str());
    return;
  }
  auto attr = node_proto->add_attribute();
  if (attr == nullptr) {
    GELOGE(GRAPH_FAILED, "attr is nullptr.");
    return;
  }
  attr->set_name(name);
  switch (type) {
    case onnx::AttributeProto_AttributeType_FLOAT:
      attr->set_f((*(static_cast<float *>(data))));
      attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
      break;

    case onnx::AttributeProto_AttributeType_FLOATS:
      attr->set_type(onnx::AttributeProto_AttributeType_FLOATS);
      for (auto &v : (*(static_cast<std::vector<float> *>(data)))) {
        attr->add_floats(v);
      }
      break;

    case onnx::AttributeProto_AttributeType_INT:
      attr->set_type(onnx::AttributeProto_AttributeType_INT);
      attr->set_i((*(static_cast<int64_t *>(data))));
      break;

    case onnx::AttributeProto_AttributeType_INTS:
      attr->set_type(onnx::AttributeProto_AttributeType_INTS);
      for (auto &v : *(static_cast<std::vector<int64_t> *>(data))) {
        attr->add_ints(v);
      }
      break;

    case onnx::AttributeProto_AttributeType_STRING:
      attr->set_type(onnx::AttributeProto_AttributeType_STRING);
      attr->set_s((*(static_cast<std::string *>(data))));
      break;

    case onnx::AttributeProto_AttributeType_STRINGS:
      attr->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      for (auto &v : *(static_cast<std::vector<std::string> *>(data))) {
        attr->add_strings(v);
      }
      break;

    default:
      GELOGW("AttributeProto AttributeType: %u is not supported for now", type);
      break;
  }
}

void OnnxUtils::AddAttrProto(onnx::NodeProto *node_proto, onnx::AttributeProto_AttributeType type, const string &name,
                             ::google::protobuf::RepeatedField<::google::protobuf::int64> data) {
  if (node_proto == nullptr) {
    GELOGE(FAILED, "Node_proto %s is nullptr.", name.c_str());
    return;
  }
  if (!data.empty()) {
    auto attr = node_proto->add_attribute();
    if (attr == nullptr) {
      GELOGE(GRAPH_FAILED, "attr is nullptr.");
      return;
    }
    attr->set_name(name);
    for (auto &v : data) {
      attr->add_ints(v);
    }
    attr->set_type(type);
  }
}

void OnnxUtils::AddAttrProto(onnx::NodeProto *node_proto, onnx::AttributeProto_AttributeType type, const string &name,
                             ::google::protobuf::RepeatedField<bool> data) {
  if (node_proto == nullptr) {
    GELOGE(FAILED, "Node proto %s is nullptr.", name.c_str());
    return;
  }
  if (!data.empty()) {
    auto attr = node_proto->add_attribute();
    if (attr == nullptr) {
      GELOGE(GRAPH_FAILED, "attr is nullptr.");
      return;
    }
    attr->set_name(name);
    for (auto &v : data) {
      attr->add_ints(static_cast<int64_t>(v));
    }
    attr->set_type(type);
  }
}

void OnnxUtils::AddAttrProto(onnx::NodeProto *node_proto, onnx::AttributeProto_AttributeType type, const string &name,
                             ::google::protobuf::RepeatedField<float> data) {
  if (node_proto == nullptr) {
    GELOGE(FAILED, "Node_proto %s is nullptr.", name.c_str());
    return;
  }
  if (!data.empty()) {
    auto attr = node_proto->add_attribute();
    if (attr == nullptr) {
      GELOGE(GRAPH_FAILED, "attr is nullptr.");
      return;
    }
    attr->set_name(name);
    for (auto &v : data) {
      attr->add_floats(v);
    }
    attr->set_type(type);
  }
}

void OnnxUtils::AddAttrProto(onnx::NodeProto *node_proto, onnx::AttributeProto_AttributeType type, const string &name,
                             ::google::protobuf::RepeatedPtrField<::std::string> data) {
  if (node_proto == nullptr) {
    GELOGE(FAILED, "Node proto %s is nullptr.", name.c_str());
    return;
  }
  if (!data.empty()) {
    auto attr = node_proto->add_attribute();
    if (attr == nullptr) {
      GELOGE(GRAPH_FAILED, "attr is nullptr.");
      return;
    }
    attr->set_name(name);
    for (auto &v : data) {
      attr->add_strings(v);
    }
    attr->set_type(type);
  }
}

void OnnxUtils::AddAttrProtoForOpInAndOutDesc(onnx::NodeProto *node_proto, const OpDescPtr &op_desc) {
  if (node_proto == nullptr || op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "node_proto or op_desc is nullptr");
    return;
  }
  // Input describes
  auto size_in = op_desc->GetAllInputsSize();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "input_desc_nums", &size_in);
  if (size_in > 0) {
    for (uint32_t i = 0; i < size_in; i++) {
      auto input_desc = op_desc->GetInputDescPtrDfault(i);
      if (input_desc != nullptr) {
        auto data_type = TypeUtils::DataTypeToSerialString(input_desc->GetDataType());
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING, "input_desc_dtype:" + std::to_string(i),
                     &data_type);
        auto data_type_origin = TypeUtils::DataTypeToSerialString(input_desc->GetOriginDataType());
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                     "input_desc_origin_dtype:" + std::to_string(i), &data_type_origin);
        auto dims = input_desc->GetShape().GetDims();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "input_desc_shape:" + std::to_string(i),
                     &dims);
        auto dims_origin = input_desc->GetOriginShape().GetDims();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS,
                     "input_desc_origin_shape:" + std::to_string(i), &dims_origin);
        auto layout = TypeUtils::FormatToSerialString(input_desc->GetFormat());
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING, "input_desc_layout:" + std::to_string(i),
                     &layout);
        auto layout_origin = TypeUtils::FormatToSerialString(input_desc->GetOriginFormat());
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                     "input_desc_origin_layout:" + std::to_string(i), &layout_origin);
        auto tensor_descriptor = input_desc->tensor_descriptor_.GetProtoMsg();
        if (tensor_descriptor != nullptr) {
          auto size = tensor_descriptor->size();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "input_desc_size:" + std::to_string(i),
                       &size);
          auto weight_size = tensor_descriptor->weight_size();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                       "input_desc_weight_size:" + std::to_string(i), &weight_size);
          auto reuse_input = tensor_descriptor->reuse_input();
          auto reuse_input_int = static_cast<int64_t>(reuse_input);
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                       "input_desc_reuse_input:" + std::to_string(i), &reuse_input_int);
          auto output_tensor = tensor_descriptor->output_tensor();
          auto output_tensor_int = static_cast<int64_t>(output_tensor);
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                       "input_desc_output_tensor:" + std::to_string(i), &output_tensor_int);
          auto device_type = tensor_descriptor->device_type();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                       "input_desc_device_type:" + std::to_string(i), &device_type);
          auto input_tensor = tensor_descriptor->input_tensor();
          auto input_tensor_int = static_cast<int64_t>(input_tensor);
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                       "input_desc_input_tensor:" + std::to_string(i), &input_tensor_int);
          auto real_dim_cnt = tensor_descriptor->real_dim_cnt();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                       "input_desc_real_dim_cnt:" + std::to_string(i), &real_dim_cnt);
          auto data_offset = tensor_descriptor->data_offset();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                       "input_desc_data_offset:" + std::to_string(i), &data_offset);
          auto cmps_size = tensor_descriptor->cmps_size();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "input_desc_cmps_size:" + std::to_string(i),
                       &cmps_size);
          auto cmps_tab = tensor_descriptor->cmps_tab();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                       "input_desc_cmps_tab:" + std::to_string(i), &cmps_tab);
          auto cmps_tab_offset = tensor_descriptor->cmps_tab_offset();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                       "input_desc_cmps_tab_offset:" + std::to_string(i), &cmps_tab_offset);
          const auto &tensor_desc_map = tensor_descriptor->attr();
          std::string suffix = ":" + std::to_string(i);
          AddAttrProtoForAttrsFromAttrMap(tensor_desc_map, node_proto, kPrefixForInputDesc, suffix);
        } else {
          GELOGW("Tensor descriptor is nullptr");
          continue;
        }
      } else {
        GELOGW("Input desc is nullptr");
        continue;
      }
    }
  }
  // Output describes
  auto size_out = op_desc->GetOutputsSize();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "output_desc_nums", &size_out);
  if (size_out > 0) {
    for (uint32_t i = 0; i < size_out; i++) {
      auto output_desc = op_desc->GetOutputDescPtr(i);
      if (output_desc != nullptr) {
        auto data_type = TypeUtils::DataTypeToSerialString(output_desc->GetDataType());
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING, "output_desc_dtype:" + std::to_string(i),
                     &data_type);
        auto origin_data_type = TypeUtils::DataTypeToSerialString(output_desc->GetOriginDataType());
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                     "output_desc_origin_dtype:" + std::to_string(i), &origin_data_type);
        auto dims = output_desc->GetShape().GetDims();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "output_desc_shape:" + std::to_string(i),
                     &dims);
        auto dims_origin = output_desc->GetOriginShape().GetDims();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS,
                     "output_desc_origin_shape:" + std::to_string(i), &dims_origin);
        auto layout = TypeUtils::FormatToSerialString(output_desc->GetFormat());
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING, "output_desc_layout:" + std::to_string(i),
                     &layout);
        auto layout_origin = TypeUtils::FormatToSerialString(output_desc->GetOriginFormat());
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                     "output_desc_origin_layout:" + std::to_string(i), &layout_origin);
        auto tensor_descriptor = output_desc->tensor_descriptor_.GetProtoMsg();
        if (tensor_descriptor != nullptr) {
          auto size = tensor_descriptor->size();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "output_desc_size:" + std::to_string(i),
                       &size);
          auto weight_size = tensor_descriptor->weight_size();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                       "output_desc_weight_size:" + std::to_string(i), &weight_size);
          auto device_type = tensor_descriptor->device_type();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                       "output_desc_device_type:" + std::to_string(i), &device_type);
          auto real_dim_cnt = tensor_descriptor->real_dim_cnt();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                       "output_desc_real_dim_cnt:" + std::to_string(i), &real_dim_cnt);
          const auto &tensor_desc_map = tensor_descriptor->attr();
          std::string suffix = ":" + std::to_string(i);
          AddAttrProtoForAttrsFromAttrMap(tensor_desc_map, node_proto, kPrefixForOutputDesc, suffix);
        } else {
          GELOGW("Tensor descriptor is nullptr");
          continue;
        }
      } else {
        GELOGW("Output desc is nullptr");
        continue;
      }
    }
  }
}

void OnnxUtils::AddAttrProtoForAttrsFromAttrMap(
  const ::google::protobuf::Map<std::string, ::ge::proto::AttrDef> &attr_map, onnx::NodeProto *node_proto,
  const std::string &prefix, const std::string &suffix) {
  for (const auto &item : attr_map) {
    auto attr_name = item.first;
    auto attr_def = item.second;
    auto attr_type = attr_def.value_case();
    if (attr_type == ge::proto::AttrDef::kT) {
      const auto &tensor_def = attr_def.t();
      const auto &tensor_desc = tensor_def.desc();
      auto data_type = ge::proto::DataType_Name(tensor_desc.dtype());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING, prefix + attr_name + "_desc_dtype" + suffix,
                   &data_type);
      auto dims = tensor_desc.shape().dim();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, prefix + attr_name + "_desc_shape" + suffix,
                   dims);
      auto layout = tensor_desc.layout();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING, prefix + attr_name + "_desc_layout" + suffix,
                   &layout);
      auto device_type = tensor_desc.device_type();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                   prefix + attr_name + "_desc_device_type" + suffix, &device_type);
      if (kDumpLevel == DUMP_ALL) {
        auto data = tensor_def.data();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING, prefix + attr_name + "_data" + suffix,
                     &data);
      }
    }
    if (attr_type == ge::proto::AttrDef::kS) {
      if (kDumpLevel == DUMP_ALL) {
        auto str_value = attr_def.s();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING, prefix + attr_name + suffix, &str_value);
      }
    }
    if (attr_type == ge::proto::AttrDef::kI) {
      auto int_value = attr_def.i();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, prefix + attr_name + suffix, &int_value);
    }
    if (attr_type == ge::proto::AttrDef::kF) {
      auto float_value = attr_def.f();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_FLOAT, prefix + attr_name + suffix, &float_value);
    }
    if (attr_type == ge::proto::AttrDef::kB) {
      auto int_value = static_cast<int64_t>(attr_def.b());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, prefix + attr_name + suffix, &int_value);
    }
    if (attr_type == ge::proto::AttrDef::kList) {
      const auto &list_value = attr_def.list();
      auto list_value_type = list_value.val_type();
      if (list_value_type ==
          ge::proto::AttrDef_ListValue_ListValueType::AttrDef_ListValue_ListValueType_VT_LIST_STRING) {
        if (kDumpLevel == DUMP_ALL) {
          const auto &strings = list_value.s();
          AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRINGS, prefix + attr_name + suffix, strings);
        }
      }
      if (list_value_type ==
          ge::proto::AttrDef_ListValue_ListValueType::AttrDef_ListValue_ListValueType_VT_LIST_FLOAT) {
        const auto &floats = list_value.f();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_FLOATS, prefix + attr_name + suffix, floats);
      }
      if (list_value_type == ge::proto::AttrDef_ListValue_ListValueType::AttrDef_ListValue_ListValueType_VT_LIST_INT) {
        const auto &ints = list_value.i();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, prefix + attr_name + suffix, ints);
      }
      if (list_value_type == ge::proto::AttrDef_ListValue_ListValueType::AttrDef_ListValue_ListValueType_VT_LIST_BOOL) {
        const auto &bools = list_value.b();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, prefix + attr_name + suffix, bools);
      }
    }
  }
}

void OnnxUtils::AddAttrProtoFromNodeMembers(const NodePtr &node, onnx::NodeProto *node_proto) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "node is nullptr");
    return;
  }
  // 1.Attributes added from node's methods
  auto send_list = node->send_event_id_list_;
  if (!send_list.empty()) {
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "send_event_id_list", &send_list);
  }
  auto recv_list = node->recv_event_id_list_;
  if (!recv_list.empty()) {
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "recv_event_id_list", &recv_list);
  }
  auto op_desc = node->op_;
  if (op_desc != nullptr) {
    // for input_name_idx_ in opdesc
    auto input_name_2_indexs = op_desc->GetAllInputName();
    ::google::protobuf::RepeatedPtrField<::std::string> input_names;
    ::google::protobuf::RepeatedField<::google::protobuf::int64> input_indexes;
    for (const auto &input_name_2_index : input_name_2_indexs) {
      std::string input_name = input_name_2_index.first;
      input_names.Add(std::move(input_name));
      input_indexes.Add(input_name_2_index.second);
    }
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRINGS, "_input_name_key", input_names);
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "_input_name_value", input_indexes);
    // 2.Attributes added from node's op_(message OpDef)
    // Input and out describes
    AddAttrProtoForOpInAndOutDesc(node_proto, op_desc);
    // Others
    auto op_def = op_desc->op_def_.GetProtoMsg();
    if (op_def != nullptr) {
      auto id = op_def->id();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "id", &id);
      auto stream_id = op_def->stream_id();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "stream_id", &stream_id);
      const auto &input_name = op_def->input_name();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRINGS, "input_name", input_name);
      const auto &src_name = op_def->src_name();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRINGS, "src_name", src_name);
      const auto &src_index = op_def->src_index();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "src_index", src_index);
      const auto &dst_name = op_def->dst_name();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRINGS, "dst_name", dst_name);
      const auto &dst_index = op_def->dst_index();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "dst_index", dst_index);
      const auto &input_i = op_def->input_i();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "input_i", input_i);
      const auto &output_i = op_def->output_i();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "output_i", output_i);
      const auto &workspace = op_def->workspace();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "workspace", workspace);
      const auto &workspace_bytes = op_def->workspace_bytes();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "workspace_bytes", workspace_bytes);
      const auto &is_input_const = op_def->is_input_const();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "is_input_const", is_input_const);
      const auto &op_def_attr_map = op_def->attr();
      AddAttrProtoForAttrsFromAttrMap(op_def_attr_map, node_proto);
    } else {
      GELOGE(FAILED, "Opdef is nullptr");
      return;
    }
  } else {
    GELOGE(FAILED, "Opdesc is nullptr");
    return;
  }
}

bool OnnxUtils::EncodeNodeDesc(const NodePtr &node, onnx::NodeProto *node_proto) {
  if ((node == nullptr) || (node_proto == nullptr)) {
    GELOGE(GRAPH_FAILED, "EncodeOpDesc: Input Para Node Invalid");
    return false;
  }

  // 2.Encode map<string, GeAttrValue> attrs_ to AttributeProto
  for (auto &node_attr : node->attrs_) {
    AddAttrProtoFromAttribute(node_attr, node_proto);
  }
  // 3.Encode ge::Node members to AttributeProto
  AddAttrProtoFromNodeMembers(node, node_proto);
  return true;
}

void OnnxUtils::EncodeNodeLinkForNetronVisual(const NodePtr &node, onnx::NodeProto *node_proto) {
  if ((node == nullptr) || (node_proto == nullptr)) {
    GELOGE(GRAPH_FAILED, "EncodeNodeLinkForNetronVisual: Input Para Node Invalid");
    return;
  }
  const auto &node_name = node->GetName();
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    if ((out_data_anchor != nullptr) && (!out_data_anchor->GetPeerInDataAnchors().empty())) {
      node_proto->add_output(node_name + ":" + std::to_string(out_data_anchor->GetIdx()));
    }
  }
  auto out_control_anchor = node->GetOutControlAnchor();
  if ((out_control_anchor != nullptr) && (!out_control_anchor->GetPeerInControlAnchors().empty())) {
    node_proto->add_output(node_name + kControlAnchorIndex);
  }
}

bool OnnxUtils::EncodeNodeLink(const NodePtr &node, onnx::NodeProto *node_proto) {
  if ((node == nullptr) || (node_proto == nullptr)) {
    GELOGE(GRAPH_FAILED, "EncodeNodeLink: Input Para Node Invalid");
    return false;
  }
  node_proto->clear_input();
  // 1. Add input by in data edge
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if ((peer_out_anchor != nullptr) && (peer_out_anchor->GetOwnerNode() != nullptr)) {
      node_proto->add_input(peer_out_anchor->GetOwnerNode()->GetName() + ":" +
                            std::to_string(peer_out_anchor->GetIdx()));
    } else {
      // Add "" input
      node_proto->add_input("");
    }
  }

  // 2. Add input by in control edge
  auto in_control_anchor = node->GetInControlAnchor();
  if (in_control_anchor != nullptr) {
    auto peer_out_anchors = in_control_anchor->GetPeerOutControlAnchors();
    for (const auto &peer_out_anchor : peer_out_anchors) {
      if (peer_out_anchor->GetOwnerNode()) {
        node_proto->add_input(peer_out_anchor->GetOwnerNode()->GetName() + kControlAnchorIndex);
      }
    }
  } else {
    GELOGE(FAILED, "Incontrol anchor is nullptr");
    return false;
  }

  // 3. Add output for Netron visual support
  EncodeNodeLinkForNetronVisual(node, node_proto);
  return true;
}

bool OnnxUtils::EncodeNode(const NodePtr &node, onnx::NodeProto *node_proto) {
  if ((node == nullptr) || (node_proto == nullptr)) {
    GELOGE(GRAPH_FAILED, "EncodeNode: Input Para Node Invalid");
    return false;
  }
  // 1. Encode name and type
  node_proto->set_name(node->GetName());
  /// Netron believes that some operators, such as the activation operator of softplus, only have one input,
  /// while the link relation of control anchor may exist in ge, resulting in two inputs. Therefore, "ge:" prefix
  /// is added to correctly display the link relation at the expense of some color features
  node_proto->set_op_type("ge:" + node->GetType());

  if (kDumpLevel != DUMP_WITH_OUT_DESC) {
    // 2.for attr
    if (!EncodeNodeDesc(node, node_proto)) {
      GELOGE(GRAPH_FAILED, "Encode NodeDesc: %s failed", node->GetName().c_str());
      return false;
    }
  }
  // 3.for link info
  return EncodeNodeLink(node, node_proto);
}

void OnnxUtils::EncodeTypeProtoTensorType(const NodePtr &node, onnx::TypeProto_Tensor *tensor_type) {
  if ((node == nullptr) || (tensor_type == nullptr)) {
    GELOGE(GRAPH_FAILED, "EncodeTypeProtoTensorType: Input Para Node or tensor_type Invalid");
    return;
  }
  const auto &op_desc = node->GetOpDesc();
  if (op_desc != nullptr) {
    uint32_t size_out = static_cast<uint32_t>(op_desc->GetOutputsSize());
    if (size_out > 0) {
      for (uint32_t i = 0; i < size_out; i++) {
        const ConstGeTensorDescPtr &ge_tensor = op_desc->GetOutputDescPtr(i);
        if (ge_tensor != nullptr) {
          auto ge_data_type = ge_tensor->GetDataType();
          auto onnx_data_type = EncodeDataType(ge_data_type);
          tensor_type->set_elem_type(onnx_data_type);
          onnx::TensorShapeProto *shape = tensor_type->mutable_shape();
          if (shape != nullptr) {
            for (auto d : ge_tensor->GetShape().GetDims()) {
              auto dim = shape->add_dim();
              dim->set_dim_value(d);
            }
          } else {
            GELOGW("Shape is nullptr");
            continue;
          }
        } else {
          GELOGW("Ge tensor is nullptr");
          continue;
        }
      }
    }
  } else {
    GELOGW("OpDesc  Is Empty, nodeName %s nodeType %s", node->GetName().c_str(), node->GetType().c_str());
    return;
  }
}

void OnnxUtils::EncodeValueInfo(const NodePtr &node, onnx::ValueInfoProto *value_info_proto) {
  if ((node == nullptr) || (value_info_proto == nullptr)) {
    GELOGE(GRAPH_FAILED, "EncodeValueInfo: Input Para Node or value_info_proto Invalid");
    return;
  }
  value_info_proto->set_name(node->GetName());
  onnx::TypeProto *t = value_info_proto->mutable_type();
  onnx::TypeProto_Tensor *tensor_type = t->mutable_tensor_type();
  EncodeTypeProtoTensorType(node, tensor_type);
}

bool OnnxUtils::EncodeGraph(const ConstComputeGraphPtr &graph, onnx::GraphProto *graph_proto) {
  if ((graph == nullptr) || (graph_proto == nullptr)) {
    GELOGE(GRAPH_FAILED, "EncodeGraph: Input para Invalid");
    return false;
  }
  graph_proto->set_name(graph->GetName());
  // 1. Add graph inputs
  for (const auto &input : graph->GetInputNodes()) {
    auto value_info_proto = graph_proto->add_input();
    EncodeValueInfo(input, value_info_proto);
  }
  // 2. Add graph outputs
  for (const auto &output : graph->GetOutputNodes()) {
    auto value_info_proto = graph_proto->add_output();
    EncodeValueInfo(output, value_info_proto);
  }
  // 3. Add nodes
  for (const auto &node : graph->GetDirectNode()) {
    if (!EncodeNode(node, graph_proto->add_node())) {
      GELOGW("EncodeNode failed");
      continue;
    }
  }
  return true;
}

bool OnnxUtils::ConvertGeModelToModelProto(const ge::Model &model, onnx::ModelProto &model_proto) {
  model_proto.set_model_version(model.GetVersion());
  model_proto.set_ir_version(onnx::IR_VERSION);
  model_proto.set_producer_name(model.GetName());
  auto &graph = model.graph_;
  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGE(GRAPH_FAILED, "GetComputeGraph: return nullptr");
    return false;
  }
  auto graph_proto = model_proto.mutable_graph();
  if (graph_proto == nullptr) {
    GELOGE(GRAPH_FAILED, "mutable_graph: %s return nullptr", compute_graph->GetName().c_str());
    return false;
  }
  if (!EncodeGraph(compute_graph, graph_proto)) {
    GELOGE(GRAPH_FAILED, "EncodeGraph: %s fail", compute_graph->GetName().c_str());
    return false;
  }

  // For subgraphs: a subgraph is represented by a node
  for (const auto &sub_compute_graph : compute_graph->GetAllSubgraphs()) {
    if (sub_compute_graph != nullptr) {
      auto node_proto = graph_proto->add_node();
      if (node_proto == nullptr) {
        GELOGW("Node proto is nullptr");
        continue;
      }
      node_proto->set_name(sub_compute_graph->GetName());
      node_proto->set_op_type(kNodeTypeForSubgraph);
      auto attr = node_proto->add_attribute();
      attr->set_name("graph");
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto sub_graph_proto = attr->mutable_g();
      if (sub_graph_proto == nullptr) {
        GELOGW("Sub graph proto is nullptr");
        continue;
      }
      if (!EncodeGraph(sub_compute_graph, sub_graph_proto)) {
        GELOGW("Encode sub graph: %s fail", sub_compute_graph->GetName().c_str());
        continue;
      }
    } else {
      GELOGW("Graph: %s subgraph is nullptr, skip EncodeGraph", compute_graph->GetName().c_str());
      continue;
    }
  }
  return true;
}

// Part 2: from ONNX Protobuf convert to IR
static std::map<onnx::TensorProto_DataType, ge::DataType> onnxDataTypeToGeMap = {
  {onnx::TensorProto_DataType_INT64, DT_INT64},   {onnx::TensorProto_DataType_UINT64, DT_UINT64},
  {onnx::TensorProto_DataType_FLOAT, DT_FLOAT},   {onnx::TensorProto_DataType_INT32, DT_INT32},
  {onnx::TensorProto_DataType_UINT32, DT_UINT32}, {onnx::TensorProto_DataType_INT8, DT_INT8},
  {onnx::TensorProto_DataType_UINT8, DT_UINT8},   {onnx::TensorProto_DataType_INT16, DT_INT16},
  {onnx::TensorProto_DataType_UINT16, DT_UINT16}, {onnx::TensorProto_DataType_FLOAT16, DT_FLOAT16},
  {onnx::TensorProto_DataType_DOUBLE, DT_DOUBLE}, {onnx::TensorProto_DataType_BOOL, DT_BOOL},
};

ge::DataType OnnxUtils::DecodeDataType(onnx::TensorProto_DataType data_type) {
  auto it = onnxDataTypeToGeMap.find(data_type);
  if (it != onnxDataTypeToGeMap.end()) {
    return it->second;
  } else {
    GELOGW("DecodeDataType: datatype not support %u", data_type);
    return ge::DT_UNDEFINED;
  }
}

bool OnnxUtils::ParseNameIndex(const std::string &node_name_index, std::string &node_name, int32_t &index) {
  auto sep = node_name_index.rfind(':');
  if (sep == std::string::npos) {
    return false;
  }
  node_name = node_name_index.substr(0, sep);
  auto index_str = node_name_index.substr(sep + 1);
  index = static_cast<int32_t>(std::strtol(index_str.c_str(), nullptr, 10));
  return true;
}

bool OnnxUtils::DecodeNodeLinkImp(const NodeLinkInfo &item, NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "DecodeNodeLinkImp: node_ptr is nullptr");
    return false;
  }
  // Data edge
  if (item.src_out_index >= 0) {
    auto src_anchor = node_ptr->GetOutDataAnchor(item.src_out_index);
    auto dst_anchor = item.dst_node->GetInDataAnchor(item.dst_in_index);
    if ((src_anchor == nullptr) || (dst_anchor == nullptr)) {
      GELOGE(GRAPH_FAILED, "Get data anchor failed %s:%d, %s:%d ", item.src_node_name.c_str(), item.src_out_index,
             item.dst_node_name.c_str(), item.dst_in_index);
      return false;
    }
    if (src_anchor->LinkTo(dst_anchor) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Data Anchor: src_anchor->LinkTo(dst_anchor) failed");
      return false;
    }
    // Control edge
  } else {
    auto src_anchor = node_ptr->GetOutControlAnchor();
    auto dst_anchor = item.dst_node->GetInControlAnchor();
    if ((src_anchor == nullptr) || (dst_anchor == nullptr)) {
      GELOGE(GRAPH_FAILED, "Get control anchor failed %s:%d, %s:%d ", item.src_node_name.c_str(), item.src_out_index,
             item.dst_node_name.c_str(), item.dst_in_index);
      return false;
    }
    if (src_anchor->LinkTo(dst_anchor) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Control Anchor: src_anchor->LinkTo(dst_anchor) failed");
      return false;
    }
  }
  return true;
}

bool OnnxUtils::DecodeNodeLink(const std::vector<onnx::NodeProto> &node_proto_vector,
                               const std::map<std::string, NodePtr> &node_map) {
  for (const auto &node_proto : node_proto_vector) {
    const auto &node_name = node_proto.name();
    auto dst_node = node_map.find(node_name);
    if ((dst_node == node_map.end()) || (dst_node->second == nullptr)) {
      GELOGE(GRAPH_FAILED, "destination node: %s find failed or is nullptr", node_name.c_str());
      return false;
    }
    int32_t dst_index = 0;
    for (const auto &input : node_proto.input()) {
      std::string input_node_name;
      int32_t index = 0;
      if (ParseNameIndex(input, input_node_name, index)) {
        auto item = NodeLinkInfo{input_node_name, index, dst_node->second, dst_index, node_proto.name()};
        auto src_node = node_map.find(input_node_name);
        if (src_node == node_map.end()) {
          GELOGE(GRAPH_FAILED, "find src node: %s failed", input_node_name.c_str());
          return false;
        }
        auto node_ptr = src_node->second;
        if (node_ptr == nullptr) {
          GELOGE(GRAPH_FAILED, "src node: %s is nullptr", input_node_name.c_str());
          return false;
        }
        if (!DecodeNodeLinkImp(item, node_ptr)) {
          GELOGE(GRAPH_FAILED, "DecodeNodeLinkImp node: %s failed", input_node_name.c_str());
          return false;
        }
      }
      if (index >= 0) {
        dst_index++;
      }
    }
  }
  return true;
}

void OnnxUtils::DecodeAttribute(const onnx::AttributeProto &attr_proto, std::vector<std::string> &strings) {
  if (attr_proto.type() != onnx::AttributeProto_AttributeType_STRINGS) {
    GELOGE(GRAPH_FAILED, "Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    return;
  }
  for (int i = 0; i < attr_proto.strings_size(); i++) {
    strings.push_back(attr_proto.strings(i));
  }
}

void OnnxUtils::DecodeAttribute(const onnx::AttributeProto &attr_proto, std::string &value) {
  if (attr_proto.type() != onnx::AttributeProto_AttributeType_STRING) {
    GELOGE(GRAPH_FAILED, "Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    return;
  }
  value = attr_proto.s();
}

void OnnxUtils::DecodeAttribute(const onnx::AttributeProto &attr_proto, std::vector<int64_t> &ints) {
  if (attr_proto.type() != onnx::AttributeProto_AttributeType_INTS) {
    GELOGE(GRAPH_FAILED, "Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    return;
  }
  for (int i = 0; i < attr_proto.ints_size(); i++) {
    ints.push_back(attr_proto.ints(i));
  }
}

void OnnxUtils::DecodeAttribute(const onnx::AttributeProto &attr_proto, int64_t &value) {
  if (attr_proto.type() != onnx::AttributeProto_AttributeType_INT) {
    GELOGE(GRAPH_FAILED, "Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    return;
  }
  value = attr_proto.i();
}

void OnnxUtils::DecodeNodeAttributeForOpInDesc(const onnx::AttributeProto &attr_proto,
                                               const std::string &attr_name_for_input_desc, int32_t index,
                                               OpDescPtr &op_desc) {
  if (op_desc->MutableInputDesc(static_cast<uint32_t>(index)) == nullptr) {
    GELOGE(GRAPH_FAILED, "[op name %s,attr name %s]op_desc->MutableInputDesc(static_cast<uint32_t>(index)) is nullptr",
           op_desc->GetName().c_str(), attr_name_for_input_desc.c_str());
    return;
  }
  if (attr_name_for_input_desc == "input_desc_dtype") {
    auto data_type = TypeUtils::SerialStringToDataType(attr_proto.s());
    op_desc->MutableInputDesc(static_cast<uint32_t>(index))->SetDataType(data_type);
  } else if (attr_name_for_input_desc == "input_desc_shape") {
    std::vector<std::int64_t> ints;
    DecodeAttribute(attr_proto, ints);
    GeShape ge_shape(ints);
    op_desc->MutableInputDesc(static_cast<uint32_t>(index))->SetShape(ge_shape);
  } else if (attr_name_for_input_desc == "input_desc_layout") {
    auto data_format = TypeUtils::SerialStringToFormat(attr_proto.s());
    op_desc->MutableInputDesc(static_cast<uint32_t>(index))->SetFormat(data_format);
  } else if (attr_name_for_input_desc == "input_desc_origin_shape") {
    std::vector<std::int64_t> ints;
    DecodeAttribute(attr_proto, ints);
    GeShape ge_shape(ints);
    op_desc->MutableInputDesc(static_cast<uint32_t>(index))->SetOriginShape(ge_shape);
  } else if (attr_name_for_input_desc == "input_desc_origin_layout") {
    auto data_format = TypeUtils::SerialStringToFormat(attr_proto.s());
    op_desc->MutableInputDesc(static_cast<uint32_t>(index))->SetOriginFormat(data_format);
  } else if (attr_name_for_input_desc == "input_desc_size") {
    int64_t input_size = 0;
    auto tensor_descriptor = op_desc->MutableInputDesc(static_cast<uint32_t>(index))->tensor_descriptor_.GetProtoMsg();
    DecodeAttribute(attr_proto, input_size);
    tensor_descriptor->set_size(input_size);
  } else if (attr_name_for_input_desc == "input_desc_data_offset") {
    auto tensor_descriptor = op_desc->MutableInputDesc(static_cast<uint32_t>(index))->tensor_descriptor_.GetProtoMsg();
    int64_t offset = 0;
    DecodeAttribute(attr_proto, offset);
    tensor_descriptor->set_data_offset(offset);
  } else {
    return;
  }
}

void OnnxUtils::DecodeNodeAttributeForOpOutDesc(const onnx::AttributeProto &attr_proto,
                                                const std::string &attr_name_for_output_desc, int32_t index,
                                                OpDescPtr &op_desc) {
  if (op_desc->MutableOutputDesc(static_cast<uint32_t>(index)) == nullptr) {
    GELOGE(GRAPH_FAILED, "[op name %s,attr name %s]op_desc->MutableOutputDesc(static_cast<uint32_t>(index)) is nullptr",
           op_desc->GetName().c_str(), attr_name_for_output_desc.c_str());
    return;
  }
  if (attr_name_for_output_desc == "output_desc_dtype") {
    auto data_type = TypeUtils::SerialStringToDataType(attr_proto.s());
    op_desc->MutableOutputDesc(static_cast<uint32_t>(index))->SetDataType(data_type);
  } else if (attr_name_for_output_desc == "output_desc_shape") {
    std::vector<std::int64_t> ints;
    DecodeAttribute(attr_proto, ints);
    GeShape ge_shape(ints);
    op_desc->MutableOutputDesc(static_cast<uint32_t>(index))->SetShape(ge_shape);
  } else if (attr_name_for_output_desc == "output_desc_layout") {
    auto data_format = TypeUtils::SerialStringToFormat(attr_proto.s());
    op_desc->MutableOutputDesc(static_cast<uint32_t>(index))->SetFormat(data_format);
  } else if (attr_name_for_output_desc == "output_desc_origin_shape") {
    std::vector<std::int64_t> ints;
    DecodeAttribute(attr_proto, ints);
    GeShape ge_shape(ints);
    op_desc->MutableOutputDesc(static_cast<uint32_t>(index))->SetOriginShape(ge_shape);
  } else if (attr_name_for_output_desc == "output_desc_origin_layout") {
    auto data_format = TypeUtils::SerialStringToFormat(attr_proto.s());
    op_desc->MutableOutputDesc(static_cast<uint32_t>(index))->SetOriginFormat(data_format);
  } else if (attr_name_for_output_desc == "output_desc_size") {
    int64_t output_size = 0;
    auto tensor_descriptor = op_desc->MutableOutputDesc(static_cast<uint32_t>(index))->tensor_descriptor_.GetProtoMsg();
    DecodeAttribute(attr_proto, output_size);
    tensor_descriptor->set_size(output_size);
  } else if (attr_name_for_output_desc == "output_desc_data_offset") {
    auto tensor_descriptor = op_desc->MutableOutputDesc(static_cast<uint32_t>(index))->tensor_descriptor_.GetProtoMsg();
    int64_t offset = 0;
    DecodeAttribute(attr_proto, offset);
    tensor_descriptor->set_data_offset(offset);
  } else {
    return;
  }
}

void OnnxUtils::DecodeNodeAttributeForOpInAndOutDesc(const onnx::AttributeProto &attr_proto,
                                                     const std::string &attr_name_for_input_output_desc, int32_t index,
                                                     OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "op_desc is nullptr");
    return;
  }
  if (attr_name_for_input_output_desc.substr(0, kInputPrefixLength) == "input") {
    DecodeNodeAttributeForOpInDesc(attr_proto, attr_name_for_input_output_desc, index, op_desc);
  } else if (attr_name_for_input_output_desc.substr(0, kOutputPrefixLength) == "output") {
    DecodeNodeAttributeForOpOutDesc(attr_proto, attr_name_for_input_output_desc, index, op_desc);
  } else {
    return;
  }
}

void OnnxUtils::DecodeNodeAttributeForOpDef(const onnx::AttributeProto &attr_proto, ge::proto::OpDef &op_def) {
  auto attr_map = op_def.mutable_attr();
  const auto &attr_name = attr_proto.name();
  ge::proto::AttrDef op_attr;
  int64_t value = 0;
  DecodeAttribute(attr_proto, value);
  op_attr.set_i(value);
  attr_map->insert(AttrDefPair(attr_name, op_attr));
}

void OnnxUtils::DecodeNodeAttributeForOpDesc(const onnx::AttributeProto &attr_proto, OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "DecodeNodeAttributeForOpDesc: op_desc is nullptr");
    return;
  }
  const auto &attr_name = attr_proto.name();
  std::string attr_name_for_input_output_desc;
  int32_t index = 0;
  if (!ParseNameIndex(attr_name, attr_name_for_input_output_desc, index)) {
    if (attr_name == "id") {
      op_desc->SetId(attr_proto.i());
    } else if (attr_name == "stream_id") {
      op_desc->SetStreamId(attr_proto.i());
    } else if (attr_name == "src_name") {
      std::vector<std::string> strings;
      DecodeAttribute(attr_proto, strings);
      op_desc->SetSrcName(strings);
    } else if (attr_name == "dst_name") {
      std::vector<std::string> strings;
      DecodeAttribute(attr_proto, strings);
      op_desc->SetDstName(strings);
    } else if (attr_name == "src_index") {
      std::vector<std::int64_t> ints;
      DecodeAttribute(attr_proto, ints);
      op_desc->SetSrcIndex(ints);
    } else if (attr_name == "dst_index") {
      std::vector<std::int64_t> ints;
      DecodeAttribute(attr_proto, ints);
      op_desc->SetDstIndex(ints);
    } else if (attr_name == "fusion_scope") {
      DecodeNodeAttributeForOpDef(attr_proto, *op_desc->op_def_.GetProtoMsg());
    } else if (attr_name == "input_i") {
      std::vector<std::int64_t> ints;
      DecodeAttribute(attr_proto, ints);
      op_desc->SetInputOffset(ints);
    } else if (attr_name == "output_i") {
      std::vector<std::int64_t> ints;
      DecodeAttribute(attr_proto, ints);
      op_desc->SetOutputOffset(ints);
    } else {
      return;
    }
    // Update input and output desc
  } else {
    DecodeNodeAttributeForOpInAndOutDesc(attr_proto, attr_name_for_input_output_desc, index, op_desc);
  }
}

bool OnnxUtils::DecodeNodeDesc(const onnx::NodeProto *node_proto, OpDescPtr &op_desc) {
  if (op_desc == nullptr || node_proto == nullptr) {
    GELOGE(GRAPH_FAILED, " Op_desc is nullptr or node_proto is nullptr");
    return false;
  }
  // 1. Decode node_proto name and type
  op_desc->SetName(node_proto->name());
  const auto &node_type_with_ge_prefix = node_proto->op_type();
  auto sep = node_type_with_ge_prefix.find(':');
  if (sep == std::string::npos) {
    return false;
  }
  auto node_type = node_type_with_ge_prefix.substr(sep + 1);
  op_desc->SetType(node_type);
  // 2. Add empty input and output desc
  for (const auto &attr : node_proto->attribute()) {
    if (attr.name() == "input_desc_nums") {
      auto size_in = attr.i();
      for (int64_t i = 0; i < size_in; i++) {
        GeTensorDesc ge_tensor_desc;
        GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(ge_tensor_desc) == GRAPH_SUCCESS, continue, "Add inputdesc failed.");
      }
    }
    if (attr.name() == "output_desc_nums") {
      auto size_out = attr.i();
      for (int64_t i = 0; i < size_out; i++) {
        GeTensorDesc ge_tensor_desc;
        GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(ge_tensor_desc) == GRAPH_SUCCESS, continue, "Add outputdesc failed.");
      }
    }
  }
  // 3.Decode node_proto attributes
  for (int i = 0; i < node_proto->attribute_size(); i++) {
    DecodeNodeAttributeForOpDesc(node_proto->attribute(i), op_desc);
  }
  return true;
}

bool OnnxUtils::DecodeGraph(int recursion_depth, const onnx::GraphProto &graph_proto, ComputeGraphPtr &graph) {
  if (recursion_depth > kMaxRecursionDepth) {
    GELOGE(GRAPH_FAILED, "DecodeGraph: recursion depth is too large, abort");
    return false;
  }

  graph = ComGraphMakeShared<ge::ComputeGraph>(graph_proto.name());
  GE_CHK_BOOL_EXEC(graph != nullptr, return false, "ComputeGraph make shared failed");
  /// 1. Decode all nodes first, node should include input
  /// and output nodes and nodes which represent sub graphs
  std::map<std::string, NodePtr> node_map;
  std::vector<onnx::NodeProto> node_proto_vector;
  for (const auto &node_proto : graph_proto.node()) {
    // a. nodes represent sub graphs
    if (node_proto.op_type() == kNodeTypeForSubgraph) {
      ComputeGraphPtr compute_graph;
      // in this case, node only have one attr, whose type is AttributeProto_AttributeType_GRAPH
      const auto &node_attr = node_proto.attribute(0);
      if ((node_attr.type() == onnx::AttributeProto_AttributeType_GRAPH) &&
          DecodeGraph(recursion_depth + 1, node_attr.g(), compute_graph)) {
        (void)graph->AddSubGraph(compute_graph);
      } else {
        GELOGE(GRAPH_FAILED, "Decode sub graph %s failed with node type:%d", node_proto.name().c_str(),
               node_attr.type());
        return false;
      }
      // b. direct nodes in graph
    } else {
      node_proto_vector.push_back(node_proto);
      OpDescPtr op_desc = ComGraphMakeShared<OpDesc>();
      // b.1 For node desc
      if (!DecodeNodeDesc(&node_proto, op_desc)) {
        GELOGE(GRAPH_FAILED, "Decode node desc %s failed ", node_proto.name().c_str());
        return false;
      }
      auto node = graph->AddNode(op_desc);
      node_map.insert(std::make_pair(node_proto.name(), node));
    }
  }
  /// We get all nodes in graph here
  /// b.2 For node link
  if (!DecodeNodeLink(node_proto_vector, node_map)) {
    GELOGE(GRAPH_FAILED, "Decode node link failed");
    return false;
  }

  // 2. Add inputs nodes for graph
  for (const auto &input : graph_proto.input()) {
    const auto &input_node_name = input.name();
    auto input_node_item = node_map.find(input_node_name);
    if (input_node_item == node_map.end()) {
      GELOGE(GRAPH_FAILED, "cannot find graph's input node %s in node_", input_node_name.c_str());
      return false;
    }
    auto ret = graph->AddInputNode(input_node_item->second);
    GE_CHK_BOOL_EXEC(ret != nullptr, continue, "Add inputnode failed");
  }
  // 3. Add outputs nodes for graph
  for (const auto &output : graph_proto.output()) {
    const auto &output_node_name = output.name();
    auto output_node_item = node_map.find(output_node_name);
    if (output_node_item == node_map.end()) {
      GELOGE(GRAPH_FAILED, "cannot find graph's output node %s in node_", output_node_name.c_str());
      return false;
    }
    auto ret = graph->AddOutputNode(output_node_item->second);
    if (ret == nullptr) {
      GELOGW("Add outputnode failed,out put node is %s", output_node_name.c_str());
      continue;
    }
  }
  return true;
}

bool OnnxUtils::ConvertModelProtoToGeModel(const onnx::ModelProto &model_proto, ge::Model &model) {
  model.name_ = model_proto.producer_name();
  model.version_ = static_cast<uint32_t>(model_proto.model_version());

  auto &graph_proto = model_proto.graph();
  ComputeGraphPtr compute_graph;
  // 0 means recursion depth, father call
  if (!DecodeGraph(0, graph_proto, compute_graph)) {
    GELOGE(GRAPH_FAILED, "Decode compute graph from graph_proto failed");
    return false;
  }
  model.graph_ = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  return true;
}
}  // namespace ge
