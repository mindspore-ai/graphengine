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

#ifndef INC_COMMON_UTILS_AI_CORE_COMMON_CONSTANTS_H_
#define INC_COMMON_UTILS_AI_CORE_COMMON_CONSTANTS_H_

#include <string>

namespace fe {
static const std::string CORE_TYPE = "_coretype";
/* engine name of AI core and vector core */
static const std::string AI_CORE_NAME = "AIcoreEngine";
static const std::string VECTOR_CORE_NAME = "VectorEngine";

static const int64_t IS_UNKNOWN_SHAPE_VALUE = 1;

static const int64_t SHAPE_UNKNOWN_DIM = -1;

static const int64_t SHAPE_UNKNOWN_DIM_NUM = -2;

static const std::string SOC_VERSION_ASCEND310 = "Ascend310";
static const std::string SOC_VERSION_ASCEND610 = "Ascend610";
static const std::string SOC_VERSION_ASCEND615 = "Ascend615";
static const std::string SOC_VERSION_ASCEND710 = "Ascend710";
static const std::string SOC_VERSION_ASCEND710P = "Ascend710Pro";
static const std::string SOC_VERSION_ASCEND910A = "Ascend910A";
static const std::string SOC_VERSION_ASCEND910B = "Ascend910B";
static const std::string SOC_VERSION_ASCEND910PROA = "Ascend910ProA";
static const std::string SOC_VERSION_ASCEND910PROB = "Ascend910ProB";
static const std::string SOC_VERSION_ASCEND910PREMIUMA = "Ascend910PremiumA";
static const std::string SOC_VERSION_HI3796CV300ES = "Hi3796CV300ES";
static const std::string SOC_VERSION_HI3796CV300CS = "Hi3796CV300CS";

static const std::vector<std::string> SOC_VERSION_CLOUD_LIST = {
        SOC_VERSION_ASCEND910A, SOC_VERSION_ASCEND910B, SOC_VERSION_ASCEND910PROA,
        SOC_VERSION_ASCEND910PROB, SOC_VERSION_ASCEND910PREMIUMA
};

static const std::vector<std::string> SOC_VERSION_DC_LIST = {SOC_VERSION_ASCEND610, SOC_VERSION_ASCEND615,
                                                             SOC_VERSION_ASCEND710, SOC_VERSION_ASCEND710P};
} // namespace fe
#endif
