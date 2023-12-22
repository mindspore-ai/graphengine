/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef ACLDVPP_SOLARIZE_H_
#define ACLDVPP_SOLARIZE_H_

#include "acldvpp_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief acldvppSolarizeGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
* @param [in] self: npu device侧的aclTensor，仅支持连续的Tensor，数据类型支持 UINT8 和 FLOAT，
*                   数据格式支持NCHW、NHWC，C轴支持1和3。
* @param [in] threshold：aclFloatArray类型，长度为2，反转的像素阈值范围。应该以（min，max）的格式提供，
                         min <= max，当数据类型为UINT8时，min和max取[0，255]范围内的整数值，
                         当数据类型为FLOAT时，min和max取[0，1]范围内的数值。
                         如果min = max，则反转大于等于min（max）的所有像素值。
* @param [in] out: npu device侧的aclTensor，仅支持连续的Tensor，数据类型支持 UINT8 和 FLOAT，
*                  数据格式支持NCHW、NHWC，且数据格式、数据类型、shape需要与self一致。
* @param [out] workspaceSize：返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor：返回op执行器，包含了算子计算流程。
* @return acldvppStatus：返回状态码。
*/
acldvppStatus acldvppSolarizeGetWorkspaceSize(const aclTensor *self, const aclFloatArray* threshold, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
* @brief acldvppSolarize 的第二段接口，用于执行计算。
* @param [in] workspace：在npu device侧申请的workspace内存起址。
* @param [in] workspaceSize：在npu device侧申请的workspace大小，由第一段接口acldvppSolarizeGetWorkspaceSize获取。
* @param [in] executor：op执行器，包含了算子计算流程。
* @param [in] stream：acl stream流。
* @return acldvppStatus：返回状态码。
*/
acldvppStatus acldvppSolarize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_SOLARIZE_H_