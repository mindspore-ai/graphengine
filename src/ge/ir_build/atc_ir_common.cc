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

#include "atc_ir_common.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "external/ge/ge_api_types.h"

using std::pair;
using std::string;
using std::vector;

namespace ge {
namespace {
const int64_t kDynamicInputDim = -1;
const int64_t kDynamicImageSizeNum = 2;
// datatype/formats from user to GE, Unified to util interface file later
const std::map<std::string, ge::DataType> kOutputTypeSupportDatatype = {
  {"FP32", ge::DT_FLOAT}, {"FP16", ge::DT_FLOAT16}, {"UINT8", ge::DT_UINT8}};
const std::set<std::string> kBufferOptimizeSupportOption = {"l1_optimize", "l2_optimize", "off_optimize",
                                                            "l1_and_l2_optimize"};
}  // namespace

bool CheckDynamicBatchSizeInputShapeValid(unordered_map<string, vector<int64_t>> shape_map,
                                          std::string &dynamic_batch_size) {
  int32_t size = 0;
  for (auto iter = shape_map.begin(); iter != shape_map.end(); ++iter) {
    vector<int64_t> shape = iter->second;
    if (shape.size() < 1) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10017");
      GELOGE(ge::PARAM_INVALID, "--input_shape's shape size can not be less than 1 when set --dynamic_batch_size.");
      return false;
    }
    if (shape[0] == kDynamicInputDim) {
      for (size_t i = 1; i < shape.size(); ++i) {
        if (shape[i] < 1) {
          ErrorManager::GetInstance().ATCReportErrMessage("E10018", {"index", "shape"},
                                                          {std::to_string(i), std::to_string(shape[i])});
          GELOGE(ge::PARAM_INVALID, "Only batch N can be -1 when set --dynamic_batch_size, current shape[%zu] is %ld",
                 i, shape[i]);
          return false;
        }
      }
      size++;
    }
  }

  if (size == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10043");
    GELOGE(ge::PARAM_INVALID, "At least one batch n must be equal to -1 when set --dynamic_batch_size.");
    return false;
  }

  for (char c : dynamic_batch_size) {
    if (!isdigit(c) && (c != ',') && (c != ' ')) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10047", {"value"}, {dynamic_batch_size});
      GELOGE(ge::PARAM_INVALID, "Input parameter[--dynamic_batch_size]'s value[%s] is invalid.",
             dynamic_batch_size.c_str());
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
  for (unordered_map<string, vector<int64_t>>::iterator iter = shape_map.begin(); iter != shape_map.end(); ++iter) {
    vector<int64_t> shape = iter->second;
    // only support four dim
    if (shape.size() != DIM_DEFAULT_SIZE) {
      if (std::count(shape.begin(), shape.end(), kDynamicInputDim) > 0) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10019");
        GELOGE(ge::PARAM_INVALID,
               "--input_shape's shape is invalid, only height or width can be -1 when set --dynamic_image_size.");
        return false;
      }
      continue;
    }

    int64_t height = 0;
    int64_t width = 0;
    if (input_format == "NCHW") {
      height = shape[NCHW_DIM_H];
      width = shape[NCHW_DIM_W];
    }

    if (input_format == "NHWC") {
      height = shape[NHWC_DIM_H];
      width = shape[NHWC_DIM_W];
    }

    if (height == kDynamicInputDim && width == kDynamicInputDim &&
        std::count(shape.begin(), shape.end(), kDynamicInputDim) == kDynamicImageSizeNum) {
      size++;
    } else if (std::count(shape.begin(), shape.end(), kDynamicInputDim) == 0) {
      continue;
    } else {
      ErrorManager::GetInstance().ATCReportErrMessage("E10019");
      GELOGE(ge::PARAM_INVALID,
             "--input_shape's shape is invalid, only height or width can be -1 when set --dynamic_image_size.");
      return false;
    }
  }
  if (size == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10019");
    GELOGE(ge::PARAM_INVALID,
           "--input_shape's shape is invalid, only height or width can be -1 when set --dynamic_image_size.");
    return false;
  }

  if (dynamic_image_size.back() == ';') {
    dynamic_image_size.erase(dynamic_image_size.end() - 1);
  }

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

Status CheckDynamicBatchSizeOrImageSizeParamValid(std::string &dynamic_batch_size, std::string &dynamic_image_size,
                                                  const std::string input_shape, const std::string input_format,
                                                  bool &is_dynamic_input) {
  if (!dynamic_batch_size.empty() && !dynamic_image_size.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10009", {"parameter0", "parameter1"},
                                                    {"dynamic_batch_size", "dynamic_image_size"});
    GELOGE(ge::PARAM_INVALID, "dynamic_batch_size and dynamic_image_size can not both exist");
    return ge::PARAM_INVALID;
  }

  if (dynamic_batch_size.empty() && dynamic_image_size.empty()) {
    return ge::SUCCESS;
  }

  unordered_map<string, vector<int64_t>> shape_map;
  vector<pair<string, vector<int64_t>>> user_shape_map;
  is_dynamic_input = true;
  if (!ParseInputShape(input_shape, shape_map, user_shape_map, is_dynamic_input)) {
    GELOGE(ge::PARAM_INVALID, "Failed to parse input shape: %s", input_shape.c_str());
    return ge::PARAM_INVALID;
  }

  if (shape_map.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10000", {"parameter"}, {"input_shape"});
    GELOGE(ge::PARAM_INVALID, "The input_shape can not be empty in dynamic batchsize scenario.");
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
  return ge::SUCCESS;
}

bool ParseInputShape(const string &input_shape, unordered_map<string, vector<int64_t>> &shape_map,
                     vector<pair<string, vector<int64_t>>> &user_shape_map, bool is_dynamic_input) {
  vector<string> shape_vec = StringUtils::Split(input_shape, ';');
  const int DEFAULT_SHAPE_PAIR_SIZE = 2;
  for (const auto &shape : shape_vec) {
    vector<string> shape_pair_vec = StringUtils::Split(shape, ':');
    if (shape_pair_vec.size() != DEFAULT_SHAPE_PAIR_SIZE) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10010", {"shape"}, {shape});
      GELOGW(
        "Input parameter[--input_shape]’s shape is [%s], "
        "correct sample is input_name1:n1,c1,h1,w1",
        shape.c_str());
      return false;
    }
    if (shape_pair_vec[1].empty()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10011", {"shape"}, {shape});
      GELOGW(
        "Input parameter[--input_shape]’s shape is [%s], can not empty, "
        "correct sample is input_name1:n1,c1,h1,w1",
        shape.c_str());
      return false;
    }

    vector<string> shape_value_strs = StringUtils::Split(shape_pair_vec[1], ',');
    vector<int64_t> shape_values;
    for (auto &shape_value_str : shape_value_strs) {
      // stoul: The method may throw an exception: invalid_argument/out_of_range
      if (std::string::npos != shape_value_str.find('.')) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10012", {"shape"}, {shape_value_str});
        GELOGW("--input_shape's shape value[%s] exist float number the correct sample is \"input_name1:1,3,224,224\"",
               shape_value_str.c_str());
        return false;
      }

      long left_result = 0;
      try {
        left_result = stol(StringUtils::Trim(shape_value_str));
      } catch (const std::out_of_range &) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10013", {"parameter", "shape"}, {"input_shape", shape});
        GELOGW("--input_shape’s shape_value_str[%s] cause out of range execption!", shape_value_str.c_str());
        return false;
      } catch (const std::invalid_argument &) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10014", {"parameter", "shape"},
                                                        {"input_shape", shape_value_str});
        GELOGW("--input_shape’s shape_value_str[%s] cause invalid argument!", shape_value_str.c_str());
        return false;
      } catch (...) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10015", {"parameter", "shape"},
                                                        {"input_shape", shape_value_str});
        GELOGW("--input_shape’s shape_value_str[%s] stol fail!", shape_value_str.c_str());
        return false;
      }
      int64_t result = left_result;
      // - 1 is not currently supported
      if (!is_dynamic_input && result <= 0) {
        GELOGW("Invalid parameter for input shape: %s ,expect positive integer , but value = %ld", shape.c_str(),
               result);
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
    ErrorManager::GetInstance().ATCReportErrMessage("E10042", {"value"}, {output_type});
    GELOGE(ge::PARAM_INVALID, "Invalid value for --output_type[%s], only support DT_FLOAT, DT_FLOAT16, DT_UINT8!!",
           output_type.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckBufferOptimizeParamValid(const std::string buffer_optimize) {
  if ((!buffer_optimize.empty()) &&
      (kBufferOptimizeSupportOption.find(buffer_optimize) == kBufferOptimizeSupportOption.end())) {
    GELOGE(ge::PARAM_INVALID,
           "buffer_optimize flag %s is invalid, only support"
           "l1_optimize,l2_optimize, off_optimize, l1_and_l2_optimize",
           buffer_optimize.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckCompressWeightParamValid(const std::string enable_compress_weight, const std::string compress_weight_conf) {
  if ((!compress_weight_conf.empty()) &&
      (!CheckInputPathValid(compress_weight_conf, ge::ir_option::COMPRESS_WEIGHT_CONF))) {
    GELOGE(ge::PARAM_INVALID, "compress weight config file %s not found!!", compress_weight_conf.c_str());
    return ge::PARAM_INVALID;
  }
  if ((enable_compress_weight != "") && (enable_compress_weight != "true") && (enable_compress_weight != "false")) {
    GELOGE(ge::PARAM_INVALID, "Input parameter[--enable_compress_weight]'s value[%s] must be true or false.",
           enable_compress_weight.c_str());
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
}  // namespace ge
