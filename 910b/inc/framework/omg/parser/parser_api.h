/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef INC_FRAMEWORK_OMG_PARSER_PARSER_API_H_
#define INC_FRAMEWORK_OMG_PARSER_PARSER_API_H_

#include <map>
#include <string>
#include "external/ge/ge_api_error_codes.h"
#include "graph/ascend_string.h"

namespace ge {
// Initialize parser
GE_FUNC_VISIBILITY Status ParserInitialize(const std::map<std::string, std::string>& options);
// Finalize parser, release all resources
GE_FUNC_VISIBILITY Status ParserFinalize();
// Initialize parser
GE_FUNC_VISIBILITY Status ParserInitialize(const std::map<ge::AscendString, ge::AscendString>& options);
}  // namespace ge
#endif // INC_FRAMEWORK_OMG_PARSER_PARSER_API_H_
