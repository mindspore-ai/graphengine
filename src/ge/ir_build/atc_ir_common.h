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

#ifndef FRAMEWORK_DOMI_ATC_IR_COMMON_H_
#define FRAMEWORK_DOMI_ATC_IR_COMMON_H_

#include <unistd.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/omg/omg_inner_types.h"

namespace ge {
bool CheckDynamicBatchSizeInputShapeValid(unordered_map<string, vector<int64_t>> shape_map,
                                          std::string &dynamic_batch_size);

bool CheckDynamicImagesizeInputShapeValid(unordered_map<string, vector<int64_t>> shape_map,
                                          const std::string input_format, std::string &dynamic_image_size);

Status CheckDynamicBatchSizeOrImageSizeParamValid(std::string &dynamic_batch_size, std::string &dynamic_image_size,
                                                  const std::string input_shape, const std::string input_format,
                                                  bool &is_dynamic_input);

bool ParseInputShape(const std::string &input_shape, std::unordered_map<string, std::vector<int64_t>> &shape_map,
                     std::vector<std::pair<string, vector<int64_t>>> &user_shape_map, bool is_dynamic_input = false);
}  // namespace ge
#endif  // FRAMEWORK_DOMI_ATC_IR_COMMON_H_