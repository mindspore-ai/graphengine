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
#include "atc_ir_common.h"
#include "common/util/error_manager/error_manager.h"
#include "external/ge/ge_api_types.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"

using std::pair;
using std::string;
using std::vector;

namespace ge {
namespace {
const int64_t kDynamicInputDim = -1;
const int64_t kDynamicImageSizeNum = 2;
const size_t kMaxDynamicDimNum = 100;
const size_t kMaxNDDimNum = 4;
const size_t kMinNDDimNum = 1;
// datatype/formats from user to GE, Unified to util interface file later
const std::map<std::string, ge::DataType> kOutputTypeSupportDatatype = {
    {"FP32", ge::DT_FLOAT}, {"FP16", ge::DT_FLOAT16}, {"UINT8", ge::DT_UINT8}};
const char *const kOutputTypeSupport = "only support FP32, FP16, UINT8";
const std::set<std::string> kBufferOptimizeSupportOption = {"l1_optimize", "l2_optimize", "off_optimize",
                                                            "l1_and_l2_optimize"};
// The function is incomplete. Currently, only l2_optimize, off_optimize is supported.
const char *const kBufferOptimizeSupport = "only support l2_optimize, off_optimize";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_DEFAULT = "high_performance";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_PRECISON = "high_precision";
const char *const kInputShapeSample1 = "\"input_name1:n1,c1,h1,w1\"";
const char *const kInputShapeSample2 = "\"input_name1:1,3,224,224\"";
const char *const kSplitError1 = "size not equal to 2 split by \":\"";
const char *const kEmptyError = "can not be empty";
const char *const kFloatNumError = "exist float number";
const char *const kDigitError = "is not digit";
const char *const kCompressWeightError = "it must be appointed when appoint parameter[--optypelist_for_implmode]";
const char *const kSelectImplmodeError = "only support high_performance, high_precision";
const char *const kDynamicBatchSizeError = "It can only contains digit, \",\", \" \"";

vector<string> SplitInputShape(const std::string &input_shape) {
  vector<string> shape_pair_vec;
  size_t pos = input_shape.rfind(":");
  if (pos != std::string::npos) {
    shape_pair_vec.emplace_back(input_shape.substr(0, pos));
    shape_pair_vec.emplace_back(input_shape.substr(pos + 1, input_shape.size() - pos));
  }
  return shape_pair_vec;
}
}  // namespace

bool CheckDynamicBatchSizeInputShapeValid(unordered_map<string, vector<int64_t>> shape_map,
                                          std::string &dynamic_batch_size) {
  int32_t size = 0;
  for (auto iter = shape_map.begin(); iter != shape_map.end(); ++iter) {
    vector<int64_t> shape = iter->second;
    if (shape.empty()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10012");
      GELOGE(ge::PARAM_INVALID, "--input_shape's shape size can not be less than 1 when set --dynamic_batch_size.");
      return false;
    }

    if (std::count(shape.begin(), shape.end(), kDynamicInputDim) == 0) {
      continue;
    }

    bool ret = multibatch::CheckDynamicBatchShape(shape, iter->first);
    if (ret) {
      size++;
    }
  }

  if (size == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10031");
    GELOGE(ge::PARAM_INVALID, "At least one batch n must be equal to -1 when set --dynamic_batch_size.");
    return false;
  }

  for (char c : dynamic_batch_size) {
    if (!isdigit(c) && (c != ',') && (c != ' ')) {
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E10033", {"value", "reason"}, {dynamic_batch_size, kDynamicBatchSizeError});
      GELOGE(ge::PARAM_INVALID, "Input parameter[--dynamic_batch_size]'s value[%s] is invalid. reason: %s",
             dynamic_batch_size.c_str(), kDynamicBatchSizeError);
      return false;
    }
  }
  if (dynamic_batch_size.back() == ',') {
    dynamic_batch_size.erase(dynamic_batch_size.end() - 1);
  }
  return true;
}

bool CheckDynamicImagesizeInputShapeValid(unordered_map<string, vector<int64_t>> shape_map,
                                          const std::string input_format, std::string &dynamic_image_size) {
  int32_t size = 0;
  for (auto iter = shape_map.begin(); iter != shape_map.end(); ++iter) {
    vector<int64_t> shape = iter->second;
    // only support four dim
    if (shape.size() != DIM_DEFAULT_SIZE) {
      if (std::count(shape.begin(), shape.end(), kDynamicInputDim) > 0) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10019");
        GELOGE(ge::PARAM_INVALID,
               "--input_shape's shape is invalid, only height and width can be -1 when set --dynamic_image_size.");
        return false;
      }
      continue;
    }

    if (std::count(shape.begin(), shape.end(), kDynamicInputDim) == 0) {
      continue;
    }
    auto ret = multibatch::CheckDynamicImageSizeShape(shape, iter->first, input_format);
    if (ret) {
      size++;
    } else {
      return ret;
    }
  }
  if (size == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10019");
    GELOGE(ge::PARAM_INVALID,
           "--input_shape's shape is invalid, only height and width can be -1 when set --dynamic_image_size.");
    return false;
  }

  EraseEndSemicolon(dynamic_image_size);
  // Different parameter sets are split string by ';'
  std::vector<std::string> split_set = StringUtils::Split(dynamic_image_size, ';');
  // Different dimensions are split by ','
  std::vector<std::string> split_dim;
  for (auto str : split_set) {
    split_dim = StringUtils::Split(str, ',');
    if (split_dim.size() != static_cast<size_t>(kDynamicImageSizeNum)) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10020", {"DynamicImageSizeNum"},
                                                      {std::to_string(kDynamicImageSizeNum)});
      GELOGE(ge::PARAM_INVALID,
             "--dynamic_image_size's number of dimensions of each "
             "group must be %ld.",
             kDynamicImageSizeNum);
      return false;
    }
  }

  return true;
}

bool CheckDynamicDimsInputShapeValid(const unordered_map<string, vector<int64_t>> &shape_map,
                                     string input_format, string &dynamic_dims) {
  if (input_format != "ND") {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"},
        {"--input_format", input_format.c_str(), "input_format must be ND when set dynamic_dims"});
    GELOGE(ge::PARAM_INVALID, "input_format must be ND when set dynamic_dims.");
    return false;
  }

  int32_t dynamic_dim = 0;
  for (auto &info_shapes : shape_map) {
    auto &shapes = info_shapes.second;
    if (shapes.size() > kMaxNDDimNum || shapes.size() < kMinNDDimNum) {
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E10001", {"parameter", "value", "reason"},
          {"--input_shape's dim", std::to_string(shapes.size()), "Dim num must within [1, 4] when set dynamic_dims"});
      GELOGE(ge::PARAM_INVALID, "Dim num must within [%zu, %zu] when set dynamic_dims.", kMinNDDimNum, kMaxNDDimNum);
      return false;
    }
    dynamic_dim += std::count(shapes.begin(), shapes.end(), kDynamicInputDim);
  }
  if (dynamic_dim == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"},
        {"--input_shape's dynamic dim num", "0", "at least one dim should be -1 when set dynamic_dims"});
    GELOGE(ge::PARAM_INVALID, "input_shape's shape is invalid, at least one dim should be -1 when set dynamic_dims.");
    return false;
  }

  if (!CheckAndParseDynamicDims(dynamic_dim, dynamic_dims)) {
    GELOGE(ge::PARAM_INVALID, "Check and parse dynamic dims: %s failed.", dynamic_dims.c_str());
    return false;
  }

  return true;
}

bool CheckAndParseDynamicDims(int32_t dynamic_dim_num, std::string &dynamic_dims) {
  EraseEndSemicolon(dynamic_dims);
  if (dynamic_dims.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"},
        {"--dynamic_dims", dynamic_dims.c_str(), "dynamic_dims can not be empty"});
    GELOGE(ge::PARAM_INVALID, "dynamic_dims can not be empty.");
    return false;
  }
  // Different parameter sets are split by ';'
  vector<string> split_set = StringUtils::Split(dynamic_dims, ';');
  if (split_set.size() > kMaxDynamicDimNum) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10042", {"parameter", "reason"}, {"dynamic_dims", "dynamic_dims's num of parameter set can not exceed 100"});
    GELOGE(ge::PARAM_INVALID, "dynamic_dims's num of parameter set can not exceed %zu.", kMaxDynamicDimNum);
    return false;
  }
  for (auto split_dim : split_set) {
    vector<string> one_set = StringUtils::Split(split_dim, ',');
    if (one_set.size() != static_cast<size_t>(dynamic_dim_num)) {
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E10042", {"parameter", "reason"},
          {"dynamic_dims", "Each gear setting needs to be consistent with the number of -1 in the inputshape"});
      GELOGE(ge::PARAM_INVALID, "Input parameter --dynamic_dims parse failed, "
          "reason: Each gear setting needs to be consistent with the number of -1 in the inputshape.");
      return false;
    }
    for (auto dim : one_set) {
      for (auto c : dim) {
        if (!isdigit(c)) {
          ErrorManager::GetInstance().ATCReportErrMessage(
              "E10001", {"parameter", "value", "reason"},
              {"--dynamic_dims's parameter", dim.c_str(), "must be positive integer"});
          GELOGE(ge::PARAM_INVALID, "dynamic_dims's parameter must be positive integer.");
          return false;
        }
      }
    }
  }
  return true;
}

Status CheckDynamicInputParamValid(string &dynamic_batch_size, string &dynamic_image_size, string &dynamic_dims,
                                   const string input_shape, const string input_format, bool &is_dynamic_input) {
  int32_t param_size = static_cast<int32_t>(!dynamic_batch_size.empty()) +
                       static_cast<int32_t>(!dynamic_image_size.empty()) + static_cast<int32_t>(!dynamic_dims.empty());
  if (param_size > 1) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10009", {"parameter0", "parameter1", "parameter2"},
                                                    {"dynamic_batch_size", "dynamic_image_size", "dynamic_dims"});
    GELOGE(ge::PARAM_INVALID, "dynamic_batch_size, dynamic_image_size and dynamic_dims can only be set one");
    return ge::PARAM_INVALID;
  }

  if (param_size == 0) {
    return ge::SUCCESS;
  }

  unordered_map<string, vector<int64_t>> shape_map;
  vector<pair<string, vector<int64_t>>> user_shape_map;
  is_dynamic_input = true;
  if (input_shape.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10004", {"parameter"}, {"input_shape"});
    GELOGE(ge::PARAM_INVALID, "The input_shape can not be empty in dynamic input size scenario.");
    return ge::PARAM_INVALID;
  }

  if (!ParseInputShape(input_shape, shape_map, user_shape_map, is_dynamic_input)) {
    GELOGE(ge::PARAM_INVALID, "Failed to parse input shape: %s", input_shape.c_str());
    return ge::PARAM_INVALID;
  }

  if (!dynamic_batch_size.empty()) {
    if (!CheckDynamicBatchSizeInputShapeValid(shape_map, dynamic_batch_size)) {
      GELOGE(ge::PARAM_INVALID, "Check dynamic batch size input shape failed: %s", input_shape.c_str());
      return ge::PARAM_INVALID;
    }
  }

  if (!dynamic_image_size.empty()) {
    if (!CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size)) {
      GELOGE(ge::PARAM_INVALID, "Check dynamic image size input shape failed: %s", input_shape.c_str());
      return ge::PARAM_INVALID;
    }
  }

  if (!dynamic_dims.empty()) {
    if (!CheckDynamicDimsInputShapeValid(shape_map, input_format, dynamic_dims)) {
      GELOGE(ge::PARAM_INVALID, "Check dynamic dims: %s of input shape: %s failed.", dynamic_dims.c_str(),
             input_shape.c_str());
      return ge::PARAM_INVALID;
    }
  }
  return ge::SUCCESS;
}

bool ParseInputShape(const string &input_shape, unordered_map<string, vector<int64_t>> &shape_map,
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
                                                        {"--input_shape", shape_value_str});
        GELOGW("Input parameter[--input_shape]â€™s value[%s] cause out of range execption!", shape_value_str.c_str());
        return false;
      } catch (const std::invalid_argument &) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10014", {"parameter", "value"},
                                                        {"--input_shape", shape_value_str});
        GELOGW("Input parameter[--input_shape]â€™s value[%s] cause invalid argument!", shape_value_str.c_str());
        return false;
      } catch (...) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10015", {"parameter", "value"},
                                                        {"--input_shape", shape_value_str});
        GELOGW("Input parameter[--input_shape]â€™s value[%s] cause unkown execption!", shape_value_str.c_str());
        return false;
      }
      int64_t result = left_result;
      // - 1 is not currently supported
      if (!is_dynamic_input && result <= 0) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10011", {"shape", "result"}, {shape, std::to_string(result)});
        GELOGW(
            "Input parameter[--input_shape]â€™s shape value[%s] is invalid, "
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

Status CheckOutputTypeParamValid(const std::string output_type) {
  if ((!output_type.empty()) && (kOutputTypeSupportDatatype.find(output_type) == kOutputTypeSupportDatatype.end())) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"}, {"--output_type", output_type, kOutputTypeSupport});
    GELOGE(ge::PARAM_INVALID,
        "Invalid value for --output_type[%s], %s.", output_type.c_str(), kOutputTypeSupport);
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckBufferOptimizeParamValid(const std::string buffer_optimize) {
  if ((!buffer_optimize.empty()) &&
      (kBufferOptimizeSupportOption.find(buffer_optimize) == kBufferOptimizeSupportOption.end())) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"}, {"--buffer_optimize", buffer_optimize, kBufferOptimizeSupport});
    GELOGE(ge::PARAM_INVALID,
        "Invalid value for --buffer_optimize[%s], %s.", buffer_optimize.c_str(), kBufferOptimizeSupport);
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckCompressWeightParamValid(const std::string enable_compress_weight, const std::string compress_weight_conf) {
  if ((!compress_weight_conf.empty()) &&
      (!CheckInputPathValid(compress_weight_conf, "--compress_weight_conf"))) {
    GELOGE(ge::PARAM_INVALID, "compress weight config file not found, file_name:%s", compress_weight_conf.c_str());
    return ge::PARAM_INVALID;
  }
  if ((enable_compress_weight != "") && (enable_compress_weight != "true") && (enable_compress_weight != "false")) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10005", {"parameter", "value"}, {"enable_compress_weight", enable_compress_weight});
    GELOGE(ge::PARAM_INVALID,
        "Input parameter[--enable_compress_weight]'s value[%s] must be true or false.", enable_compress_weight.c_str());
    return ge::PARAM_INVALID;
  }

  if ((enable_compress_weight == "true") && (!compress_weight_conf.empty())) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10009", {"parameter0", "parameter1"},
                                                    {"enable_compress_weight", "compress_weight_conf"});
    GELOGE(ge::PARAM_INVALID, "enable_compress_weight and compress_weight_conf can not both exist!!");
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

int CheckLogParamValidAndSetLogLevel(const std::string log) {
  int ret = -1;
  if (log == "default") {
    ret = 0;
  } else if (log == "null") {
    ret = dlog_setlevel(-1, DLOG_NULL, 0);
  } else if (log == "debug") {
    ret = dlog_setlevel(-1, DLOG_DEBUG, 1);
  } else if (log == "info") {
    ret = dlog_setlevel(-1, DLOG_INFO, 1);
  } else if (log == "warning") {
    ret = dlog_setlevel(-1, DLOG_WARN, 1);
  } else if (log == "error") {
    ret = dlog_setlevel(-1, DLOG_ERROR, 1);
  } else {
    GELOGE(ge::PARAM_INVALID, "invalid value for log:%s, only support debug, info, warning, error, null", log.c_str());
    return ret;
  }
  if (ret != 0) {
    GELOGE(ge::PARAM_INVALID, "Log setlevel fail !");
  }
  return ret;
}

Status CheckInsertOpConfParamValid(const std::string insert_op_conf) {
  if ((!insert_op_conf.empty()) &&
      (!CheckInputPathValid(insert_op_conf, "--insert_op_conf"))) {
    GELOGE(ge::PARAM_INVALID, "insert op config file not found: %s", insert_op_conf.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckDisableReuseMemoryParamValid(const std::string disable_reuse_memory) {
  if ((disable_reuse_memory != "") && (disable_reuse_memory != "0") && (disable_reuse_memory != "1")) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10006", {"parameter"}, {"disable_reuse_memory"});
    GELOGE(ge::PARAM_INVALID, "Input parameter[--disable_reuse_memory]'s value must be 1 or 0.");
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckEnableSingleStreamParamValid(const std::string enable_single_stream) {
  if ((enable_single_stream != "") && (enable_single_stream != "true") && (enable_single_stream != "false")) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10005", {"parameter", "value"}, {"enable_single_stream", enable_single_stream});
    GELOGE(ge::PARAM_INVALID, "Input parameter[--enable_single_stream]'s value[%s] must be true or false.",
           enable_single_stream.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckImplmodeParamValid(const std::string &optypelist_for_implmode, std::string &op_select_implmode) {
  // only appointed op_select_implmode, can user appoint optypelist_for_implmode
  if (optypelist_for_implmode != "" && op_select_implmode == "") {
    ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
        {"--op_select_implmode", op_select_implmode.c_str(), kCompressWeightError});
    GELOGE(ge::PARAM_INVALID, "Invalid value for --op_select_implmode[%s], %s.",
        op_select_implmode.c_str(), kCompressWeightError);
    return ge::PARAM_INVALID;
  }
  // op_select_implmode default value is high_performance
  if (op_select_implmode == "") {
    op_select_implmode = IR_OPTION_OP_SELECT_IMPLMODE_DEFAULT;
  } else {
    if (op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_DEFAULT &&
      op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_PRECISON) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
          {"--op_select_implmode", op_select_implmode.c_str(), kSelectImplmodeError});
      GELOGE(ge::PARAM_INVALID, "Invalid value for --op_select_implmode[%s], %s.",
          op_select_implmode.c_str(), kSelectImplmodeError);
      return ge::PARAM_INVALID;
    }
  }

  return ge::SUCCESS;
}

void PrintOptionMap(std::map<std::string, std::string> &options, std::string tips) {
  for (auto iter = options.begin(); iter != options.end(); iter++) {
    std::string key = iter->first;
    std::string option_name = iter->second;
    GELOGD("%s set successfully, option_key=%s, option_value=%s", tips.c_str(), key.c_str(), option_name.c_str());
  }
}

void EraseEndSemicolon(string &param) {
  if (param.empty()) {
    return;
  }
  if (param.back() == ';') {
    param.erase(param.end() - 1);
  }
}
}  // namespace ge
