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

/** @defgroup aoe aoe调优接口 */
#ifndef TUNE_API_H
#define TUNE_API_H
#include <map>
#include <string>
#include "ge/ge_api.h"
#include "aoe_types.h"

/**
 * @ingroup aoe
 * @par 描述: 命令行调优
 *
 * @attention 无
 * @param  option [IN] 调优参数
 * @param  msg [OUT] 调优异常下返回信息
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILURE 执行失败
 * @par 依赖:
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
AoeStatus AoeOfflineTuning(const std::map<std::string, std::string> &option, std::string &msg);

/**
 * @ingroup aoe
 * @par 描述: 调优初始化
 *
 * @attention 无
 * @param  session [IN] ge连接会话
 * @param  option [IN] 参数集. 包含调优参数及ge参数
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILURE 执行失败
 * @par 依赖:
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeOnlineInitialize(ge::Session *session, const std::map<std::string, std::string> &option);

/**
 * @ingroup aoe
 * @par 描述: 调优去初始化
 *
 * @attention 无
 * @param  无
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILURE 执行失败
 * @par 依赖:
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeOnlineFinalize();

/**
 * @ingroup aoe
 * @par 描述: 调优处理
 *
 * @attention 无
 * @param  tuningGraph [IN] 调优图
 * @param  dependGraph [IN] 调优依赖图
 * @param  session [IN] ge连接会话
 * @param  option [IN] 参数集. 包含调优参数及ge参数
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILURE 执行失败
 * @par 依赖:
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeOnlineTuning(ge::Graph &tuningGraph, std::vector<ge::Graph> &dependGraph,
    ge::Session *session, const std::map<std::string, std::string> &option);
#endif
