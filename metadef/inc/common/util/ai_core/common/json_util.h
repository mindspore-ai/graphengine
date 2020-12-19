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

#ifndef PROJECT_JSON_UTIL_H
#define PROJECT_JSON_UTIL_H

#include "graph/compute_graph.h"

#include "common/aicore_util_types.h"
#include "fusion_engine/graph_tuner/graph_tuner_errorcode.h"

const std::string L1_FUSION_EXTEND_CONTENT = "_l1_fusion_extend_content";
const std::string L2_FUSION_EXTEND_CONTENT = "l2_fusion_extend_content";
const std::string TASK_L2_FUSION_INFO_EXTEND_CONTENT = "task_l2_fusion_info_extend_content";
const std::string L1_FUSION_TO_OP_STRUCT = "_l1fusion_ToOpStruct";
const std::string L2_FUSION_TO_OP_STRUCT = "_l2fusion_ToOpStruct";
const std::string TASK_L2_FUSION_INFO = "_task_L2FusionInfo";

namespace tune {
using ToOpStructPtr = std::shared_ptr<fe::ToOpStruct_t>;
using L2FusionInfoPtr = std::shared_ptr<fe::TaskL2FusionInfo_t>;

Status GetL1InfoFromJson(ge::OpDescPtr opDescPtr);

Status GetL2InfoFromJson(ge::OpDescPtr opDescPtr);

Status GetTaskL2FusionInfoFromJson(ge::OpDescPtr opDescPtr);

Status ReadGraphInfoFromJson(ge::ComputeGraph &graph);

Status WriteGraphInfoToJson(ge::ComputeGraph &graph);

void GetL2ToOpStructFromJson(ge::OpDescPtr &opDescPtr, ToOpStructPtr &l2InfoPtr);

void GetL1ToOpStructFromJson(ge::OpDescPtr &opDescPtr, ToOpStructPtr &l1InfoPtr);

L2FusionInfoPtr GetL2FusionInfoFromJson(ge::OpDescPtr &opDescPtr);

void SetL2FusionInfoToNode(ge::OpDescPtr &opDescPtr, L2FusionInfoPtr &l2FusionInfoPtr);
} // namespace tune
#endif //PROJECT_JSON_UTIL_H
