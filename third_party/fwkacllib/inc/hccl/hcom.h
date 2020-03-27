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

#ifndef __HCOM_H__
#define __HCOM_H__

#include <runtime/rt.h>

#include <hccl/base.h>

#ifdef __cplusplus
extern "C" {
#endif

extern hcclResult_t hcom_init(const char *rank_table, const char *identify);

extern hcclResult_t hcom_destroy(void);

extern hcclResult_t hcom_bind_model(rtModel_t model, rtStream_t stream);

extern hcclResult_t hcom_unbind_model(rtModel_t model);

extern hcclResult_t hcom_all_gather(const char *tag, void *inputPtr, void *outputPtr, u64 inputCount,
                                    hcclDataType_t dataType, const char *group, rtStream_t stream);

extern hcclResult_t hcom_all_reduce(const char *tag, void *inputPtr, void *outputPtr, u64 count,
                                    hcclDataType_t dataType, hcclRedOp_t op, const char *group, rtStream_t stream);

extern hcclResult_t hcom_broadcast(const char *tag, void *ptr, u64 count, hcclDataType_t dataType, u32 root,
                                   const char *group, rtStream_t stream);

extern hcclResult_t hcom_reduce_scatter(const char *tag, void *inputPtr, void *outputPtr, u64 count,
                                        hcclDataType_t dataType, hcclRedOp_t op, const char *group, rtStream_t stream);

hcclResult_t hcom_get_rank_size(const char *group, u32 *rankSize);

hcclResult_t hcom_get_local_rank_size(const char *group, u32 *localRankSize);

hcclResult_t hcom_get_rank_id(const char *group, u32 *rankId);

hcclResult_t hcom_get_local_rank_id(const char *group, u32 *localRankId);

hcclResult_t hcom_get_world_rank_from_group_rank(const char *group, u32 groupRank, u32 *worldRank);

hcclResult_t hcom_get_group_rank_from_world_rank(u32 worldRank, const char *group, u32 *groupRank);

hcclResult_t hcom_create_group(const char *group, u32 rankNum, u32 *rankIds);

hcclResult_t hcom_destroy_group(const char *group);

hcclResult_t hcom_send(const char *tag, void *inputPtr, u64 count, hcclDataType_t dataType, u32 destRank, u32 srTag,
                       const char *group, rtStream_t stream);

hcclResult_t hcom_receive(const char *tag, void *outputPtr, u64 count, hcclDataType_t dataType, u32 srcRank, u32 srTag,
                          const char *group, rtStream_t stream);

hcclResult_t hcom_get_split_strategy(const char *group, const struct model_feature *feature, u32 maxSegmentNum,
                                     u32 *segmentNum, u32 *segmentIdx);

extern hcclResult_t hcom_set_split_strategy_by_index(const char *group, u32 segmentNum, const u32 *IdxList);

extern hcclResult_t hcom_set_split_strategy_by_size(const char *group, u32 segmentNum, const float *sizeList);
#ifdef __cplusplus
}
#endif
#endif  // __HCOM_H__
