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

#ifndef INC_FRAMEWORK_OMG_OMG_INNER_TYPES_H_
#define INC_FRAMEWORK_OMG_OMG_INNER_TYPES_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/common/fmk_error_codes.h"
#include "framework/common/types.h"
#include "register/register_fmk_types.h"

using std::map;
using std::string;
using std::unordered_map;
using std::vector;
using domi::domiTensorFormat_t;
using domi::DOMI_TENSOR_ND;
using domi::DOMI_TENSOR_RESERVED;
using domi::FMK_TYPE_RESERVED;
using domi::FrameworkType;

namespace ge {
///
/// @ingroup domi_omg
/// @brief run model
///
enum RunMode {
  kGeOmModel = 0,     // generate offline model file
  kModelToJson = 1,   // convert to JSON file
  kOnlyPreCheck = 3,  // only for pre-check
  kPbtxtToJson = 5    // pbtxt to json
};

///
/// @ingroup domi_omg
/// @brief high-precision mode
///
enum HighPrecisionMode {
  // in common mode, the FP16 high-precision function is disabled
  kHighPrecisonDefault = 0,

  // high-precision mode, enabling FP16 high-precision mode (Convolution/FullConnect/AvgPooling are involved)
  kHighPrecisionFP16 = 1
};

///
/// @ingroup domi_omg
/// @brief description buffer data
///
struct OMGBufferData {
  void *data;
  uint32_t length;
};

struct OmgContext {
  OmgContext() { format = DOMI_TENSOR_ND; }
  domiTensorFormat_t format;

  // format of the input specified by the command line
  std::unordered_map<std::string, domiTensorFormat_t> input_nodes_format_map;
  std::vector<domiTensorFormat_t> output_formats;

  // user-designate input dims
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
  // global input dims
  std::unordered_map<std::string, std::vector<int64_t>> input_dims;

  // resolve the mapping between operators with the same name and corresponding network. format e.g.
  // Detectionoutput:SsdDetectiontOutput
  std::map<std::string, std::string> op_conf_map;
  // saves the network output node. key = operator name, value = index, index indicates the output index of the operator
  std::map<std::string, std::vector<int32_t>> out_nodes_map;
  // user-designate out nodes (this is used for determing the orders)
  std::vector<std::pair<std::string, int32_t>> user_out_nodes;
  // path for the aicpu custom operator so file
  std::vector<std::string> aicpu_op_run_paths;
  // ddk version
  std::string ddk_version;
  // preferential format used by the entire network
  domiTensorFormat_t net_format = DOMI_TENSOR_RESERVED;
  domi::FrameworkType type = domi::FMK_TYPE_RESERVED;
  RunMode run_mode = kOnlyPreCheck;
  bool train_flag = false;
  // whether to use FP16 high precision
  int32_t fp16_high_precision = kHighPrecisonDefault;

  std::string output_type;

  // Save the name of the entire network: Some special operators are used to determine a network. Some operators in the
  // network require special processing based on the specific network.
  // e.g：faster-rcnn, the FirstStageProcessor module is determined as the Faster-R-CNN network based on the scope
  // fusion. Then, the conv+reshape operators in the FirstStageBoxPredictor/BoxEncodingPredictor scope are combined. The
  // convolution kernel rearrangement reshape operator needs to be deleted for the convolution kernel.
  std::string net_name;
  // whether to enable dynamic batch
  bool enable_l2dynamic = false;
};
}  // namespace ge

namespace domi {
///
/// @ingroup domi_omg
/// @brief get OMG context
/// @return OmgContext context
///
ge::OmgContext &GetContext();
}  // namespace domi

#endif  // INC_FRAMEWORK_OMG_OMG_INNER_TYPES_H_
