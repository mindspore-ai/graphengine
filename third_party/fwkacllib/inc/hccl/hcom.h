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

/**
 * @file hcom.h
 * @brief HCOM API
 */

#ifndef HCOM_H_
#define HCOM_H_

#include <hccl/base.h>
#include <hccl/hccl_types.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus



/**
 * @brief Get the rank number in the group.
 *
 * @param group A string identifying the group name.
 * @param rankSize A pointer identifying the rank number.
 * @return HcclResult 
 */
HcclResult hcom_get_rank_size(const char *group, u32 *rankSize);

/**
 * @brief Get the rank number of this rank's server within the group.
 *
 * @param group A string identifying the group name.
 * @param localRankSize A pointer identifying the rank number.
 * @return HcclResult 
 */
HcclResult hcom_get_local_rank_size(const char *group, u32 *localRankSize);

/**
 * @brief Get the rank id of this rank.
 *
 * @param group A string identifying the group name.
 * @param rankId A pointer identifying the rank id.
 * @return HcclResult 
 */
HcclResult hcom_get_rank_id(const char *group, u32 *rankId);

/**
 * @brief Get the local rank id of this rank's server within the group.
 *
 * @param group A string identifying the group name.
 * @param localRankId A pointer identifying the local rank id.
 * @return HcclResult 
 */
HcclResult hcom_get_local_rank_id(const char *group, u32 *localRankId);

/**
 * @brief Get the world rank id according to the group rank id.
 *
 * @param group A string identifying the group name.
 * @param groupRank An integer(u32) identifying the group rank id.
 * @param worldRank A pointer identifying the world rank id.
 * @return HcclResult 
 */
HcclResult hcom_get_world_rank_from_group_rank(const char *group, u32 groupRank, u32 *worldRank);

/**
 * @brief Get the group rank id according to the world rank id.
 *
 * @param worldRank An integer(u32) identifying the world rank id.
 * @param group A string identifying the group name.
 * @param groupRank A pointer identifying the group rank id.
 * @return HcclResult 
 */
HcclResult hcom_get_group_rank_from_world_rank(u32 worldRank, const char *group, u32 *groupRank);

/**
 * @brief Create group.
 *
 * @param group A string identifying the group name.
 * @param rankNum An integer(u32) identifying the number of ranks in the group.
 * @param rankIds A list identifying the ranks in the group.
 * @return HcclResult 
 */
HcclResult hcom_create_group(const char *group, u32 rankNum, u32 *rankIds);

/**
 * @brief Destroy group
 *
 * @param group A string identifying the group name.
 * @return HcclResult 
 */
HcclResult hcom_destroy_group(const char *group);

/**
 * @brief Set the gradient split strategy with in the group, according to gradient index.
 *
 * @param group A string identifying the group name.
 * @param segmentNum An integer(u32) identifying the segments number of gradients.
 * @param IdxList A list identifying the index of end gradient in each segment.
 * @return HcclResult
 */
extern HcclResult hcom_set_split_strategy_by_index(const char *group, u32 segmentNum, const u32 *IdxList);

/**
 * @brief Set the gradient split strategy with in the group, according to gradient data size.
 *
 * @param group A string identifying the group name.
 * @param segmentNum An integer(u32) identifying the segments number of gradients.
 * @param sizeList A list identifying the percent of each segment.
 * @return HcclResult
 */
extern HcclResult hcom_set_split_strategy_by_size(const char *group, u32 segmentNum, const float *sizeList);

/**
 * @brief Register memories and init resources for remote access.
 *
 * @param addrList memory addresses for remote access.
 * @param count number of remote memory addresses.
 * @return HcclResult
 */
extern HcclResult hcom_remote_access_mem_register(const MemRegisterAddr* addrList, u32 count);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCOM_H_
