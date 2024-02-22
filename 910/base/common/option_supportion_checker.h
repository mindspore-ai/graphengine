/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GE_GRAPH_COMMON_OPTION_SUPPORTION_CHECKER_H_
#define GE_GRAPH_COMMON_OPTION_SUPPORTION_CHECKER_H_

#include <map>
#include "external/graph/types.h"

namespace ge {
  using Status = uint32_t;
  Status IrbuildCheckSupportedGlobalOptions(const std::map<std::string, std::string> &input_options);
  Status GEAPICheckSupportedGlobalOptions(const std::map<std::string, std::string> &input_options);
  Status GEAPICheckSupportedSessionOptions(const std::map<std::string, std::string> &input_options);
  Status GEAPICheckSupportedGraphOptions(const std::map<std::string, std::string> &input_options);
} // namespace ge
#endif