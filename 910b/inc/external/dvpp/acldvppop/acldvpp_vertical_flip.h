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

#ifndef ACLDVPP_VERTICAL_FLIP_H_
#define ACLDVPP_VERTICAL_FLIP_H_

#include "acldvpp_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief acldvppVerticalFlipGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
* @param [in] self: npu device侧的aclTensor，数据类型支持 UINT8 和 FLOAT，
*                   仅支持连续的Tensor，数据格式支持NCHW、NHWC，且数据格式需要与out一致。
* @param [out] out: npu device侧的aclTensor，数据类型支持 UINT8 和 FLOAT，
*                   且数据类型需要与self构成互相推导关系，shape需要与self满足broadcast关系，
*                   支持非连续的Tensor，数据格式支持NCHW、NHWC，且数据格式和shape需要与self一致。
* @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含了算子计算流程。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppVerticalFlipGetWorkspaceSize(const aclTensor *self, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
* @brief acldvppVerticalFlip 的第二段接口，用于执行计算。
* @param [in] workspace: 在npu device侧申请的workspace内存起址。
* @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口acldvppVerticalFlipGetWorkspaceSize获取。
* @param [in] executor: op执行器，包含了算子计算流程。
* @param [in] stream: acl stream流。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppVerticalFlip(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_VERTICAL_FLIP_H_