/**
 * @file tune_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 * 描述：aoe调优接口头文件
 */
/** @defgroup aoe aoe调优接口 */
#ifndef TUNE_API_H
#define TUNE_API_H
#include <vector>
#include <map>
#include <string>
#include "graph/graph.h"
#include "ge/ge_api.h"
#include "aoe_types.h"

/**
 * @ingroup aoe
 *
 * aoe status
 */
enum MsTuneStatus {
    MSTUNE_SUCCESS,  /** tune success */
    MSTUNE_FAILED,   /** tune failed */
};

// Option key: for train options sets
const std::string MSTUNE_SELF_KEY = "mstune";
const std::string MSTUNE_GEINIT_KEY = "initialize";
const std::string MSTUNE_GESESS_KEY = "session";

#ifdef __cplusplus
extern "C" {
#endif

struct RunnerInitConfig {
    // onilne online
    std::string profPath;
    std::string parserPath;
    // ncs only
    std::vector<uint32_t> devList;
};

struct RunnerOpInfo {
    std::string opName;
    uint64_t opCostTime;
    uint64_t aicoreCostTime;
    // gradient_split only
    std::string modelName;
    std::string opType;
    std::vector<uint64_t> start;
    std::vector<uint64_t> end;
};

struct RunnerModelInfo {
    uint64_t totalCostTime;
};

struct RunnerRunResult {
    std::vector<RunnerModelInfo> modelInfo;
    std::vector<RunnerOpInfo> opInfo;
};

struct RunnerResult {
    uint64_t totalCostTime;
    std::map<std::string, uint64_t> opCostTime;
    std::map<std::string, uint64_t> aicoreCostTime;
};

struct RunnerDataBuf {
    void *ptr = nullptr;
    size_t size = 0;
};

struct AOEBufferData {
    std::shared_ptr<uint8_t> data = nullptr;
    uint64_t length;
};

struct RunnerConfig {
    bool isProf;
    uint32_t loop;
    // offline only
    std::vector<RunnerDataBuf> input;
    std::vector<RunnerDataBuf> output;
    std::string modelPath;
    RunnerDataBuf modelData;
    // online only
    uint32_t devId;
    std::vector<std::vector<ge::Tensor>> inputs;
    std::vector<ge::Graph> dependGraph; // run graph (for training)
};
#ifdef __cplusplus
}
#endif

/**
 * @ingroup aoe
 * @par 描述: 命令行调优
 *
 * @attention 无
 * @param  option [IN] 调优参数
 * @param  msg [OUT] 调优异常下返回信息
 * @retval #MSTUNE_SUCCESS 执行成功
 * @retval #MSTUNE_FAILED 执行失败
 * @par 依赖:
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
AoeStatus AoeOfflineTuning(const std::map<std::string, std::string> &option, std::string &msg);

/**
 * @ingroup aoe
 * @par 描述: 梯度调优
 *
 * @attention 无
 * @param  tuningGraph [IN] 调优图
 * @param  dependGraph [IN] 调优依赖图
 * @param  session [IN] ge连接会话
 * @param  option [IN] 参数集. 包含调优参数及ge参数
 * @retval #MSTUNE_SUCCESS 执行成功
 * @retval #MSTUNE_FAILED 执行失败
 * @par 依赖:
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" MsTuneStatus MsTrainTuning(ge::Graph &tuningGraph, std::vector<ge::Graph> &dependGraph,
    ge::Session *session, const std::map<std::string, std::map<std::string, std::string>> &option);

/**
 * @ingroup aoe
 * @par 描述: 梯度调优
 *
 * @attention 无
 * @param  tuningGraph [IN] 调优图
 * @param  dependGraph [IN] 调优依赖图
 * @param  session [IN] ge连接会话
 * @param  option [IN] 参数集. 包含调优参数及ge参数
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILED 执行失败
 * @par 依赖:
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeOnlineTuning(ge::Graph &tuningGraph, std::vector<ge::Graph> &dependGraph,
    ge::Session *session, const std::map<std::string, std::map<std::string, std::string>> &option);
#endif
