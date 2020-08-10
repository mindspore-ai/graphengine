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

#ifndef GE_COMMON_GRAPH_PARSER_UTIL_H_
#define GE_COMMON_GRAPH_PARSER_UTIL_H_

#include <google/protobuf/message.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "framework/common/types.h"
#include "framework/omg/omg_inner_types.h"
#include "proto/ge_ir.pb.h"
#include "proto/om.pb.h"

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "runtime/kernel.h"

using domi::Status;
using std::pair;
using std::string;
using std::unordered_map;
using std::vector;

namespace ge {
Status SetOutputNodeInfo(ge::Graph &graph, const std::string &output_type, const std::string &output_format);

Status ParseOutputFp16NodesFormat(const string &is_output_fp16);

Status ParseOutputNodes(const string &out_nodes);

bool ParseInputShape(const string &input_shape, unordered_map<string, vector<int64_t>> &shape_map,
                     vector<pair<string, vector<int64_t>>> &user_shape_map, bool is_dynamic_input);

Status ParseOpConf(const char *op_conf);
}  // namespace ge

namespace domi {
/**
 * @ingroup domi_omg
 * @brief get omg context
 * @return reference of OmgContext
 */
ge::OmgContext &GetContext();
}  // namespace domi

#endif  // GE_COMMON_GRAPH_PARSER_UTIL_H_
