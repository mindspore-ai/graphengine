/**
 * @file aoe_tuning_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 *
 */

#ifndef AOE_TUNING_API_H
#define AOE_TUNING_API_H

#include <memory>
#include "aoe_types.h"
#include "ge/ge_api.h"
#include "aoe_ascend_string.h"

namespace Aoe {
using SessionId = uint64_t;

// this set for global option key
const std::set<AscendString> GLOBAL_OPTION_SET = {
    AscendString(WORK_PATH),
    AscendString(SERVER_IP),
    AscendString(SERVER_PORT),
    AscendString(TUNING_PARALLEL_NUM),
    AscendString(DEVICE),
    AscendString(CORE_TYPE),
    AscendString(BUFFER_OPTIMIZE),
    AscendString(ENABLE_COMPRESS_WEIGHT),
    AscendString(COMPRESS_WEIGHT_CONF),
    AscendString(PRECISION_MODE),
    AscendString(DISABLE_REUSE_MEMORY),
    AscendString(ENABLE_SINGLE_STREAM),
    AscendString(AICORE_NUM),
    AscendString(FUSION_SWITCH_FILE),
    AscendString(ENABLE_SMALL_CHANNEL),
    AscendString(OP_SELECT_IMPL_MODE),
    AscendString(OPTYPELIST_FOR_IMPLMODE),
    AscendString(ENABLE_SCOPE_FUSION_PASSES),
    AscendString(OP_DEBUG_LEVEL),
    AscendString(VIRTUAL_TYPE),
    AscendString(SPARSITY),
    AscendString(MODIFY_MIXLIST),
    AscendString(CUSTOMIZE_DTYPES),
    AscendString(FRAMEWORK),

    AscendString(SOC_VERSION),
    AscendString(TUNE_DEVICE_IDS),
    AscendString(EXEC_DISABLE_REUSED_MEMORY),
    AscendString(AUTO_TUNE_MODE),
    AscendString(OP_COMPILER_CACHE_MODE),
    AscendString(OP_COMPILER_CACHE_DIR),
    AscendString(DEBUG_DIR),
    AscendString(EXTERNAL_WEIGHT),
    AscendString(DETERMINISTIC),
    AscendString(OPTION_HOST_ENV_OS),
    AscendString(OPTION_HOST_ENV_CPU),
};

// this set for session option key
const std::set<AscendString> SESSION_OPTION_SET = {
    AscendString(JOB_TYPE),
    AscendString(RUN_LOOP),
    AscendString(RESOURCE_CONFIG_PATH)
};

// this set for tuning option key
const std::set<AscendString> TUNING_OPTION_SET = {
    AscendString(INPUT_FORMAT),
    AscendString(INPUT_SHAPE),
    AscendString(INPUT_SHAPE_RANGE),
    AscendString(OP_NAME_MAP),
    AscendString(DYNAMIC_BATCH_SIZE),
    AscendString(DYNAMIC_IMAGE_SIZE),
    AscendString(DYNAMIC_DIMS),
    AscendString(PRECISION_MODE),
    AscendString(OUTPUT_TYPE),
    AscendString(OUT_NODES),
    AscendString(INPUT_FP16_NODES),
    AscendString(LOG_LEVEL),
    AscendString(OP_DEBUG_LEVEL),
    AscendString(INSERT_OP_FILE),
    AscendString(GE_INPUT_SHAPE_RANGE),
    AscendString(OUTPUT),
    AscendString(RELOAD),
    AscendString(TUNING_NAME),
    AscendString(FRAMEWORK),
    AscendString(MODEL_PATH),
    AscendString(TUNE_OPS_FILE),
    AscendString(RECOMPUTE),
    AscendString(AOE_CONFIG_FILE),
    AscendString(OP_PRECISION_MODE),
    AscendString(KEEP_DTYPE),
    AscendString(SINGLE_OP),
    AscendString(TUNE_OPTIMIZATION_LEVEL),
    AscendString(FEATURE_DEEPER_OPAT),
    AscendString(FEATURE_NONHOMO_SPLIT),
    AscendString(OUT_FILE_NAME),
    AscendString(HOST_ENV_OS),
    AscendString(HOST_ENV_CPU),
    AscendString(EXEC_DISABLE_REUSED_MEMORY),
    AscendString(AUTO_TUNE_MODE),
    AscendString(OP_COMPILER_CACHE_MODE),
    AscendString(OP_COMPILER_CACHE_DIR),
    AscendString(DEBUG_DIR),
    AscendString(MDL_BANK_PATH),
    AscendString(OP_BANK_PATH),
    AscendString(MODIFY_MIXLIST),
    AscendString(SHAPE_GENERALIZED_BUILD_MODE),
    AscendString(OP_DEBUG_CONFIG),
    AscendString(EXTERNAL_WEIGHT),
    AscendString(EXCLUDE_ENGINES),
};

/**
 * @brief       : initialize aoe tuning api
 * @param [in]  : map<AscendString, AscendString> &globalOptions          global options
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeInitialize(const std::map<AscendString, AscendString> &globalOptions);

/**
 * @brief       : finalize aoe tuning api
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeFinalize();

/**
 * @brief       : create aoe session
 * @param [in]  : map<AscendString, AscendString> &sessionOptions          session options
 * @param [out] : SessionId sessionId                                      session id
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeCreateSession(const std::map<AscendString, AscendString> &sessionOptions, SessionId &sessionId);

/**
 * @brief       : destroy aoe session
 * @param [in]  : SessionId sessionId                                      session id
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeDestroySession(SessionId sessionId);

/**
 * @brief       : set ge session for session id
 * @param [in]  : SessionId sessionId                                      session id
 * @param [in]  : ge::Session *geSession                                   ge session handle
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeSetGeSession(SessionId sessionId, ge::Session *geSession);

/**
 * @brief       : set depend graphs for session id
 * @param [in]  : SessionId sessionId                                      session id
 * @param [in]  : std::vector<ge::Graph> &dependGraphs                     depend graphs
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeSetDependGraphs(SessionId sessionId, std::vector<ge::Graph> &dependGraphs);

/**
 * @brief       : set inputs of depend graphs for session id
 * @param [in]  : SessionId sessionId                                      session id
 * @param [in]  : std::vector<std::vector<ge::Tensor>> &inputs             depend input tensor
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeSetDependGraphsInputs(SessionId sessionId, std::vector<std::vector<ge::Tensor>> &inputs);


/**
 * @brief       : set tuning graphs for session id
 * @param [in]  : SessionId sessionId                                      session id
 * @param [in]  : ge::Graph &tuningGraph                                   tuning graph
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeSetTuningGraph(SessionId sessionId, const ge::Graph &tuningGraph);

/**
 * @brief       : set input of tuning graph for session id
 * @param [in]  : SessionId sessionId                                      session id
 * @param [in]  : std::vector<ge::Tensor> &inputs                          input tensor
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeSetTuningGraphInput(SessionId sessionId, std::vector<ge::Tensor> &input);

/**
 * @brief       : tuning graph
 * @param [in]  : SessionId sessionId                                      session id
 * @param [in]  : map<AscendString, AscendString> &tuningOptions           tuning options
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeTuningGraph(SessionId sessionId, const std::map<AscendString, AscendString> &tuningOptions);
}  // namespace Aoe

#endif
