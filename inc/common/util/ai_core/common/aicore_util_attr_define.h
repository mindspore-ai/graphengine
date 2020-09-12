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

#ifndef INC_COMMON_UTILS_AI_CORE_COMMON_ATTR_DEFINE_H_
#define INC_COMMON_UTILS_AI_CORE_COMMON_ATTR_DEFINE_H_

#include <string>

namespace fe {
static const std::string SCOPE_ID_ATTR = "fusion_scope";

static const std::string FE_IMPLY_TYPE = "_fe_imply_type";

static const std::string PARENT_OP_TYPE = "parentOpType";

static const std::string ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR = "task_l2_fusion_info_extend_content";

static const std::string ATTR_DATA_DUMP_REF = "_datadump_ref";

static const std::string ATTR_NAME_L2_FUSION_EXTEND_PTR = "l2_fusion_extend_content";

static const std::string L1_OPTIMIZED = "l1_optimized";

static const std::string L2_OPTIMIZED = "l2_optimized";

static const std::string OP_SLICE_INFO = "_op_slice_info";
}  // namespace fe
#endif
