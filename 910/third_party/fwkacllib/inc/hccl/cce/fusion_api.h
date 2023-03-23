/**
 * @file fusion_api.h
 *
 * Copyright(C), 2017 - 2017, Huawei Tech. Co., Ltd. ALL RIGHTS RESERVED.
 *
 * @brief fusion api declare
 *
 * @version 1.0
 *
 */

#ifndef _FUSION_API_H_
#define _FUSION_API_H_

#include "cce.h"
using namespace cce;

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

/* value scope for addr Change Flag */
#define CCE_FUSION_ADDR_FLAG_UNCHANGED      (uint32_t(1))
#define CCE_FUSION_ADDR_FLAG_INOUT_CHANGED  (uint32_t(2))
#define CCE_FUSION_ADDR_FLAG_ALL_CHANGED    (uint32_t(3))

#define MAX_GRAPH_NUM_PER_STREAM           16

/**
  * @ingroup fusion
  * @brief fusion start
  * @param [in] streamId               stream number
  * @param [in] graphId                number of graph(network model op DAG) in a stream. value:0~15
  * @param [in] initFlag               init falg. value:0,valid;1,not valid
  * @param [in] memCfg.addrChangeFlag  op data addr change flag. value:0,valid;1,not valid
  * @param [in] memCfg.memAddr         memAddr
  * @param [in] memCfg.memSize         memSize
  * @return ccStatus_t                 CC_STATUS_SUCCESS, success; CC_STATUS_BAD_PARAM, fail
  */
ccStatus_t ccFusionStart(ccHandle_t handle, uint32_t graphId, uint32_t initFlag, CceFusionMemCfg_t memCfg);

/**
  * @ingroup fusion
  * @brief fusion end
  * @param [in] streamId        stream number
  * @param [in] graphId         number of graph(network model op DAG) in a stream. value:0~15
  * @return ccStatus_t          CC_STATUS_SUCCESS, success; CC_STATUS_BAD_PARAM, fail
  */
ccStatus_t ccFusionEnd(ccHandle_t handle, uint32_t graphId);

/**
  * @ingroup fusion
  * @brief fusion task end
  * @param [in] streamId        stream number
  * @param [in] graphId         number of graph(network model op DAG) in a stream. value:0~15
  * @return ccStatus_t          CC_STATUS_SUCCESS, success; CC_STATUS_BAD_PARAM, fail
  */
ccStatus_t ccFusionTaskEnd(ccHandle_t handle, uint32_t graphId);

/**
  * @ingroup fusion
  * @brief repeat launch last fusion graph
  * @param [in] handle          cce handler
  * @return ccStatus_t          CC_STATUS_SUCCESS, success; CC_STATUS_BAD_PARAM, fail
  */
ccStatus_t ccKernelLaunchRepeat(ccHandle_t handle);


/**
  * @ingroup fusion
  * @brief delete kernel for repeat mode
  * @param [in] handle          cce handler
  * @return ccStatus_t          CC_STATUS_SUCCESS, success; CC_STATUS_BAD_PARAM, fail
  */
ccStatus_t ccKernelDelete(ccHandle_t handle);

/**
* @ingroup op
* @brief op start
* @return ccStatus_t          CC_STATUS_SUCCESS, success;
*/
ccStatus_t ccOpStart();

/**
* @ingroup op
* @brief op end
* @param [in] handle          cce handler
* @return ccStatus_t          CC_STATUS_SUCCESS, success; CC_STATUS_BAD_PARAM, fail
*/
ccStatus_t ccOpEnd(ccHandle_t handle);

/**
  * @ingroup fusion
  * @brief fusion node start
  * @param [in] handle          cce handler
  * @return ccStatus_t          CC_STATUS_SUCCESS, success; CC_STATUS_BAD_PARAM, fail
  */
ccStatus_t ccFusionNodeStart(ccHandle_t handle);


/**
  * @ingroup fusion
  * @brief fusion node end
  * @param [in] handle          cce handler
  * @return ccStatus_t          CC_STATUS_SUCCESS, success; CC_STATUS_BAD_PARAM, fail
  */
ccStatus_t ccFusionNodeEnd(ccHandle_t handle);

/**
  * @ingroup fusion
  * @brief fusion node end
  * @param [in] handle          cce handler
  * @param [out] status         fusion status
  * @return ccStatus_t          CC_STATUS_SUCCESS, success; CC_STATUS_BAD_PARAM, fail
  */
ccStatus_t ccFusionNodeStatus(ccHandle_t handle, bool &status);

/**
  * @ingroup fusion
  * @brief fusion op invoke
  * @param [in] handle          cce handler
  * @return ccStatus_t          CC_STATUS_SUCCESS, success; CC_STATUS_BAD_PARAM, fail
  */
ccStatus_t ccFusionOpInvoke(ccHandle_t handle);


#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif

