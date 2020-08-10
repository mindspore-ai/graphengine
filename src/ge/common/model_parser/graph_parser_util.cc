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

#include "graph_parser_util.h"
#include <memory>
#include "common/auth/file_saver.h"
#include "common/convert/pb2json.h"
#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/model_parser/base.h"
#include "common/model_saver.h"
#include "common/properties_manager.h"
#include "common/string_util.h"
#include "common/types.h"
#include "common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "external/register/register_types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/optimize/common/params.h"
#include "graph/utils/type_utils.h"
#include "omg/omg_inner_types.h"
#include "omg/parser/model_parser.h"
#include "omg/parser/parser_factory.h"
#include "omg/parser/weights_parser.h"
#include "parser/common/pre_checker.h"
#include "proto/ge_ir.pb.h"
#include "register/op_registry.h"

namespace ge {
namespace {
// The function is incomplete. Currently, only l2_optimize, off_optimize is supported.
const char *const kInputShapeSample1 = "\"input_name1:n1,c1,h1,w1\"";
const char *const kInputShapeSample2 = "\"input_name1:1,3,224,224\"";
const char *const kSplitError1 = "size not equal to 2 split by \":\"";
const char *const kEmptyError = "can not be empty";
const char *const kFloatNumError = "exist float number";
const char *const kDigitError = "is not digit";
const char *const kOutputTypeSample = "correct sample is \"opname:index:dtype\"";
const char *const kOutputTypeSupport = "only support FP32, FP16, UINT8";
const char *const kOutputTypeError = "The multiple out nodes set in output_type must be found in out_nodes.";

vector<string> SplitInputShape(const std::string &input_shape) {
  vector<string> shape_pair_vec;
  size_t pos = input_shape.rfind(":");
  if (pos != std::string::npos) {
    shape_pair_vec.emplace_back(input_shape.substr(0, pos));
    shape_pair_vec.emplace_back(input_shape.substr(pos + 1, input_shape.size() - pos));
  }
  return shape_pair_vec;
}

static std::map<std::string, ge::DataType> output_type_str_to_datatype = {
  {"FP32", ge::DT_FLOAT}, {"FP16", ge::DT_FLOAT16}, {"UINT8", ge::DT_UINT8}};

static bool CheckInputTrueOrFalse(const std::string &s, const std::string &atc_param) {
  if ((s == "true") || (s == "false")) {
    return true;
  } else {
    ErrorManager::GetInstance().ATCReportErrMessage("E10033", {"parameter", "value"}, {atc_param, s});
    GELOGE(PARAM_INVALID, "Input parameter[--%s]'s value[%s] must be true or false.", atc_param.c_str(), s.c_str());
    return false;
  }
}

bool CheckDigitStr(std::string &str) {
  for (char c : str) {
    if (!isdigit(c)) {
      GELOGE(domi::FAILED, "value[%s] is not positive integer", str.c_str());
      return false;
    }
  }
  return true;
}

Status StringToInt(std::string &str, int32_t &value) {
  try {
    if (!CheckDigitStr(str)) {
      GELOGE(PARAM_INVALID, "Invalid of digit string: %s ", str.c_str());
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"--output_type", str, "is not positive integer"});
      return PARAM_INVALID;
    }
    value = stoi(str);
  } catch (std::invalid_argument &) {
    GELOGE(PARAM_INVALID, "Invalid of digit string: %s, catch invalid_argument.", str.c_str());
    ErrorManager::GetInstance().ATCReportErrMessage("E10014", {"parameter", "value"}, {"output_type", str});
    return PARAM_INVALID;
  } catch (std::out_of_range &) {
    GELOGE(PARAM_INVALID, "Invalid of digit string: %s, catch out_of_range.", str.c_str());
    ErrorManager::GetInstance().ATCReportErrMessage("E10013", {"parameter", "value"}, {"output_type", str});
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status VerifyOutputTypeAndOutNodes(std::vector<std::string> &out_type_vec) {
  std::vector<std::pair<std::string, int32_t>> user_out_nodes = domi::GetContext().user_out_nodes;
  std::set<std::string> out_nodes_info;
  for (uint32_t i = 0; i < user_out_nodes.size(); ++i) {
    // out_nodes set should include output_type and output_format
    std::string tmp = user_out_nodes[i].first + ":" + to_string(user_out_nodes[i].second);
    out_nodes_info.emplace(tmp);
  }
  for (uint32_t i = 0; i < out_type_vec.size(); ++i) {
    if (out_nodes_info.find(out_type_vec[i]) == out_nodes_info.end()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"--output_type", out_type_vec[i], kOutputTypeError});
      GELOGE(domi::FAILED, "Invalid value for --output_type[%s], %s.", out_type_vec[i].c_str(), kOutputTypeError);
      return domi::FAILED;
    }
  }
  return domi::SUCCESS;
}

Status ParseOutputType(const std::string &output_type, std::map<std::string, vector<uint32_t>> &out_type_index_map,
                       std::map<std::string, vector<ge::DataType>> &out_type_dt_map) {
  if (output_type.find(':') == std::string::npos) {
    GELOGI("output_type is not multiple nodes, means all out nodes");
    auto it = output_type_str_to_datatype.find(output_type);
    if (it == output_type_str_to_datatype.end()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"--output_type", output_type, kOutputTypeSupport});
      GELOGE(PARAM_INVALID, "Invalid value for --output_type[%s], %s.", output_type.c_str(), kOutputTypeSupport);
      return domi::FAILED;
    }
    return domi::SUCCESS;
  }
  std::vector<std::string> out_type_vec;
  vector<string> nodes_v = StringUtils::Split(output_type, ';');
  for (const string &node : nodes_v) {
    vector<string> node_index_type_v = StringUtils::Split(node, ':');
    if (node_index_type_v.size() != 3) {  // The size must be 3.
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"--output_type", node, kOutputTypeSample});
      GELOGE(PARAM_INVALID, "Invalid value for --output_type[%s], %s.", node.c_str(), kOutputTypeSample);
      return domi::FAILED;
    }
    ge::DataType tmp_dt;
    std::string node_name = StringUtils::Trim(node_index_type_v[0]);
    std::string index_str = StringUtils::Trim(node_index_type_v[1]);
    int32_t index;
    if (StringToInt(index_str, index) != SUCCESS) {
      GELOGE(PARAM_INVALID, "This str must be digit string, while the actual input is %s.", index_str.c_str());
      return domi::FAILED;
    }
    std::string dt_value = StringUtils::Trim(node_index_type_v[2]);
    auto it = output_type_str_to_datatype.find(dt_value);
    if (it == output_type_str_to_datatype.end()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"--output_type", dt_value, kOutputTypeSupport});
      GELOGE(ge::PARAM_INVALID, "Invalid value for --output_type[%s], %s.", dt_value.c_str(), kOutputTypeSupport);
      return domi::FAILED;
    } else {
      tmp_dt = it->second;
    }
    out_type_vec.push_back(node_name + ":" + index_str);
    auto it_index = out_type_index_map.find(node_name);
    if (it_index == out_type_index_map.end()) {
      vector<uint32_t> tmp_vec;
      tmp_vec.push_back(index);
      out_type_index_map.emplace(node_name, tmp_vec);
    } else {
      it_index->second.push_back(index);
    }

    auto it_dt = out_type_dt_map.find(node_name);
    if (it_dt == out_type_dt_map.end()) {
      vector<ge::DataType> tmp_vec;
      tmp_vec.push_back(tmp_dt);
      out_type_dt_map.emplace(node_name, tmp_vec);
    } else {
      it_dt->second.push_back(tmp_dt);
    }
  }
  return VerifyOutputTypeAndOutNodes(out_type_vec);
}

Status CheckOutNode(ge::OpDescPtr op_desc, int32_t index) {
  int32_t out_size = op_desc->GetOutputsSize();
  if (index < 0 || index >= out_size) {
    GELOGE(domi::FAILED,
           "out_node [%s] output index:%d must be smaller "
           "than node output size:%d and can not be negative!",
           op_desc->GetName().c_str(), index, out_size);
    std::string fail_reason = "output index:" + to_string(index) +
                              " must be smaller than output size:" + to_string(out_size) + " and can not be negative!";
    ErrorManager::GetInstance().ATCReportErrMessage("E10003", {"parameter", "value", "reason"},
                                                    {"out_nodes", op_desc->GetName(), fail_reason});
    return domi::FAILED;
  }
  return domi::SUCCESS;
}

Status GetOutputLeaf(NodePtr node, std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info) {
  ge::OpDescPtr tmpDescPtr = node->GetOpDesc();
  if (tmpDescPtr == nullptr) {
    GELOGE(domi::FAILED, "Get outnode op desc fail.");
    return domi::FAILED;
  }
  size_t size = tmpDescPtr->GetOutputsSize();
  if (node->GetType() != NETOUTPUT) {
    for (size_t index = 0; index < size; ++index) {
      output_nodes_info.push_back(std::make_pair(node, index));
    }
  } else {
    const auto in_anchors = node->GetAllInDataAnchors();
    for (auto in_anchor : in_anchors) {
      auto out_anchor = in_anchor->GetPeerOutAnchor();
      if (out_anchor == nullptr) {
        GELOGE(domi::FAILED, "Get leaf node op desc fail.");
        return domi::FAILED;
      }
      auto out_node = out_anchor->GetOwnerNode();
      output_nodes_info.push_back(std::make_pair(out_node, out_anchor->GetIdx()));
    }
  }
  return SUCCESS;
}

void GetOutputNodesNameAndIndex(std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info,
                                std::vector<std::string> &output_nodes_name) {
  output_nodes_name.clear();
  if (domi::GetContext().out_top_names.empty()) {
    // tf process, no top name.
    for (const auto output_node_info : output_nodes_info) {
      std::string node_name = output_node_info.first->GetName();
      int32_t index = output_node_info.second;
      output_nodes_name.push_back(node_name + ":" + std::to_string(index));
    }
    return;
  }
  // caffe process, need add top name after node_name:index
  for (size_t i = 0; i < output_nodes_info.size(); ++i) {
    std::string node_name = output_nodes_info[i].first->GetName();
    int32_t index = output_nodes_info[i].second;
    if (i < domi::GetContext().out_top_names.size()) {
      output_nodes_name.push_back(node_name + ":" + std::to_string(index) + ":" + domi::GetContext().out_top_names[i]);
    } else {
      GELOGW("Get top name of node [%s] fail.", node_name.c_str());
      output_nodes_name.push_back(node_name + ":" + std::to_string(index));
    }
  }
}
}  // namespace

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ParseOutputFp16NodesFormat(const string &is_output_fp16) {
  if (is_output_fp16.empty()) {
    return SUCCESS;
  }

  vector<domiTensorFormat_t> &output_formats = domi::GetContext().output_formats;
  output_formats.clear();
  vector<string> node_format_vec = StringUtils::Split(is_output_fp16, ',');
  for (auto &is_fp16 : node_format_vec) {
    StringUtils::Trim(is_fp16);
    if (!CheckInputTrueOrFalse(is_fp16, "is_output_adjust_hw_layout")) {
      GELOGE(PARAM_INVALID, "Invalid Param, is_output_adjust_hw_layout only support true/false: but is [%s]",
             is_output_fp16.c_str());
      return PARAM_INVALID;
    }
    if (is_fp16 == "false") {
      output_formats.push_back(DOMI_TENSOR_ND);
    } else if (is_fp16 == "true") {
      output_formats.push_back(domi::DOMI_TENSOR_NC1HWC0);
    }
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status SetOutputNodeInfo(ge::Graph &graph,
                                                                          const std::string &output_type,
                                                                          const std::string &output) {
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  std::vector<std::pair<std::string, int32_t>> user_out_nodes = domi::GetContext().user_out_nodes;
  std::vector<domiTensorFormat_t> output_formats = domi::GetContext().output_formats;
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes_info;
  std::vector<std::string> output_nodes_name;
  std::map<std::string, vector<uint32_t>> out_type_index_map;
  std::map<std::string, vector<ge::DataType>> out_type_dt_map;
  if (!output_type.empty()) {
    if (ParseOutputType(output_type, out_type_index_map, out_type_dt_map) != SUCCESS) {
      GELOGE(domi::FAILED, "Parse output_type failed.");
      return domi::FAILED;
    }
  }

  // User declared outputs
  for (uint32_t i = 0; i < user_out_nodes.size(); ++i) {
    ge::NodePtr out_node = compute_graph->FindNode(user_out_nodes[i].first);
    if (out_node == nullptr) {
      GELOGE(domi::FAILED, "Can not find src node (%s) in graph.", user_out_nodes[i].first.c_str());
      return domi::FAILED;
    }
    auto op_desc = out_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (CheckOutNode(op_desc, user_out_nodes[i].second) != SUCCESS) {
      GELOGE(domi::FAILED, "Check out node (%s) fail.", user_out_nodes[i].first.c_str());
      return domi::FAILED;
    }
    if (i < output_formats.size()) {
      if (output_formats[i] == domi::DOMI_TENSOR_NC1HWC0) {
        GELOGI("The output node [%s] should be set NC1HWC0", user_out_nodes[i].first.c_str());
        if (!ge::AttrUtils::SetBool(op_desc, "output_set_fp16_nc1hwc0", true)) {
          GELOGW("The output node [%s] set NC1HWC0 failed", user_out_nodes[i].first.c_str());
        }
      }
    }
    auto it_index = out_type_index_map.find(user_out_nodes[i].first);
    auto it_dt = out_type_dt_map.find(user_out_nodes[i].first);
    if ((it_index != out_type_index_map.end()) && (it_dt != out_type_dt_map.end())) {
      GELOGI("The output node [%s] need to be set output_type", user_out_nodes[i].first.c_str());
      (void)ge::AttrUtils::SetListDataType(op_desc, "_output_dt_list", it_dt->second);
      (void)ge::AttrUtils::SetListInt(op_desc, "_output_dt_index", it_index->second);
    }
    output_nodes_info.push_back(std::make_pair(out_node, user_out_nodes[i].second));
  }
  // default output node (leaf)
  if (user_out_nodes.empty()) {
    for (ge::NodePtr node : compute_graph->GetDirectNode()) {
      if (!node->GetInDataNodes().empty() && node->GetOutDataNodes().empty()) {
        Status ret = GetOutputLeaf(node, output_nodes_info);
        GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "find leaf fail.");
      }
    }
  }
  GetOutputNodesNameAndIndex(output_nodes_info, output_nodes_name);
  compute_graph->SetGraphOutNodesInfo(output_nodes_info);
  domi::GetContext().net_out_nodes = output_nodes_name;
  return domi::SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ParseInputShape(
  const string &input_shape, unordered_map<string, vector<int64_t>> &shape_map,
  vector<pair<string, vector<int64_t>>> &user_shape_map, bool is_dynamic_input) {
  vector<string> shape_vec = StringUtils::Split(input_shape, ';');
  const int DEFAULT_SHAPE_PAIR_SIZE = 2;
  for (const auto &shape : shape_vec) {
    vector<string> shape_pair_vec = SplitInputShape(shape);
    if (shape_pair_vec.size() != DEFAULT_SHAPE_PAIR_SIZE) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10002", {"shape", "reason", "sample"},
                                                      {shape, kSplitError1, kInputShapeSample1});
      GELOGW("Parse input parameter [--input_shape]'s shape[%s] failed, reason: %s, correct sample is %s.",
             shape.c_str(), kSplitError1, kInputShapeSample1);
      return false;
    }
    if (shape_pair_vec[1].empty()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10002", {"shape", "reason", "sample"},
                                                      {shape, kEmptyError, kInputShapeSample1});
      GELOGW("Parse input parameter [--input_shape]'s shape[%s] failed, reason: %s, correct sample is %s.",
             shape.c_str(), kEmptyError, kInputShapeSample1);
      return false;
    }

    vector<string> shape_value_strs = StringUtils::Split(shape_pair_vec[1], ',');
    vector<int64_t> shape_values;
    for (auto &shape_value_str : shape_value_strs) {
      // stoul: The method may throw an exception: invalid_argument/out_of_range
      if (std::string::npos != shape_value_str.find('.')) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10002", {"shape", "reason", "sample"},
                                                        {shape, kFloatNumError, kInputShapeSample2});
        GELOGW("Parse input parameter [--input_shape]'s shape[%s] failed, reason: %s, correct sample is %s.",
               shape.c_str(), kFloatNumError, kInputShapeSample2);
        return false;
      }

      long left_result = 0;
      try {
        left_result = stol(StringUtils::Trim(shape_value_str));
        if (!shape_value_str.empty() && (shape_value_str.front() == '-')) {
          // The value maybe dynamic shape [-1], need substr it and verify isdigit.
          shape_value_str = shape_value_str.substr(1);
        }
        for (char c : shape_value_str) {
          if (!isdigit(c)) {
            ErrorManager::GetInstance().ATCReportErrMessage("E10002", {"shape", "reason", "sample"},
                                                            {shape, kDigitError, kInputShapeSample2});
            GELOGE(PARAM_INVALID, "--input_shape's shape value[%s] is not digit", shape_value_str.c_str());
            return false;
          }
        }
      } catch (const std::out_of_range &) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10013", {"parameter", "value"},
                                                        {"input_shape", shape_value_str});
        GELOGW("Input parameter[--input_shape]’s value[%s] cause out of range execption!", shape_value_str.c_str());
        return false;
      } catch (const std::invalid_argument &) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10014", {"parameter", "value"},
                                                        {"input_shape", shape_value_str});
        GELOGW("Input parameter[--input_shape]’s value[%s] cause invalid argument!", shape_value_str.c_str());
        return false;
      } catch (...) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10015", {"parameter", "value"},
                                                        {"input_shape", shape_value_str});
        GELOGW("Input parameter[--input_shape]’s value[%s] cause unkown execption!", shape_value_str.c_str());
        return false;
      }
      int64_t result = left_result;
      // - 1 is not currently supported
      if (!is_dynamic_input && result <= 0) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10011", {"shape", "result"}, {shape, std::to_string(result)});
        GELOGW(
          "Input parameter[--input_shape]’s shape value[%s] is invalid, "
          "expect positive integer, but value is %ld.",
          shape.c_str(), result);
        return false;
      }
      shape_values.push_back(result);
    }

    shape_map.emplace(make_pair(StringUtils::Trim(shape_pair_vec[0]), shape_values));
    user_shape_map.push_back(make_pair(StringUtils::Trim(shape_pair_vec[0]), shape_values));
  }

  return true;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ParseOutputNodes(const string &out_nodes) {
  try {
    // parse output node
    if (!out_nodes.empty()) {
      domi::GetContext().out_nodes_map.clear();
      domi::GetContext().user_out_nodes.clear();

      vector<string> nodes_v = StringUtils::Split(out_nodes, ';');
      for (const string &node : nodes_v) {
        vector<string> key_value_v = StringUtils::Split(node, ':');
        if (key_value_v.size() != 2) {  // The size must be 2.
          ErrorManager::GetInstance().ATCReportErrMessage(
            "E10001", {"parameter", "value", "reason"},
            {"--out_nodes", node, "the correct format is \"node_name1:0;node_name1:1;node_name2:0\""});
          GELOGE(PARAM_INVALID,
                 "The input format of --out_nodes is invalid, the correct format is "
                 "\"node_name1:0;node_name1:1;node_name2:0\", while the actual input is %s.",
                 node.c_str());
          return PARAM_INVALID;
        }
        auto iter = domi::GetContext().out_nodes_map.find(key_value_v[0]);
        // stoi: The method may throw an exception: invalid_argument/out_of_range
        if (!CheckDigitStr(key_value_v[1])) {
          ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                          {"--out_nodes", out_nodes, "is not positive integer"});
          GELOGE(PARAM_INVALID, "This str must be digit string, while the actual input is %s", out_nodes.c_str());
          return PARAM_INVALID;
        }
        int32_t index = stoi(StringUtils::Trim(key_value_v[1]));
        if (iter != domi::GetContext().out_nodes_map.end()) {
          iter->second.emplace_back(index);
        } else {
          std::vector<int32_t> index_v;
          index_v.emplace_back(index);
          domi::GetContext().out_nodes_map.emplace(key_value_v[0], index_v);
        }
        domi::GetContext().user_out_nodes.push_back(std::make_pair(key_value_v[0], index));
      }
    }
  } catch (std::invalid_argument &) {
    GELOGE(PARAM_INVALID, "Invalid of out_nodes: %s ", out_nodes.c_str());
    ErrorManager::GetInstance().ATCReportErrMessage("E10014", {"parameter", "value"}, {"out_nodes", out_nodes});
    return PARAM_INVALID;
  } catch (std::out_of_range &) {
    GELOGE(PARAM_INVALID, "Invalid of out_nodes: %s ", out_nodes.c_str());
    ErrorManager::GetInstance().ATCReportErrMessage("E10013", {"parameter", "value"}, {"out_nodes", out_nodes});
    return PARAM_INVALID;
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ParseOpConf(const char *op_conf) {
  if (op_conf != nullptr && *op_conf != '\0') {
    // divided by ":"
    PropertiesManager::Instance().SetPropertyDelimiter(OP_CONF_DELIMITER);
    // Parsing the op_conf configuration item file
    if (!PropertiesManager::Instance().Init(op_conf)) {
      GELOGE(FAILED, "op_name_map init failed!");
      return FAILED;
    }
    // Return map and put it into ATC global variable
    domi::GetContext().op_conf_map = PropertiesManager::Instance().GetPropertyMap();
  }
  return SUCCESS;
}
}  // namespace ge
