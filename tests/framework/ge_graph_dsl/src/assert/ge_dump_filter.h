/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef INC_4C6224E8F7474EF89B18CCB0E4B19FD6
#define INC_4C6224E8F7474EF89B18CCB0E4B19FD6

#include <vector>
#include <string>
#include "ge_graph_dsl/ge.h"
#include "easy_graph/infra/keywords.h"

GE_NS_BEGIN

INTERFACE(GeDumpFilter) {
  ABSTRACT(void Update(const std::vector<std::string> &));
  ABSTRACT(void Reset());
};

GE_NS_END

#endif