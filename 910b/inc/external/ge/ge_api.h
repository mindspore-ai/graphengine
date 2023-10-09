/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef INC_EXTERNAL_GE_GE_API_H_
#define INC_EXTERNAL_GE_GE_API_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ge/ge_api_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge/ge_data_flow_api.h"
#include "ge/ge_continuous_tensor_list_api.h"
#include "ge/ge_graph_compile_summary.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "ge/ge_allocator.h"
namespace ge {
typedef uint32_t (*pCallBackFunc)(uint32_t graph_id, const std::map<std::string, ge::Tensor> &params_list);

namespace session {
typedef uint32_t (*pCallBackFunc)(uint32_t graph_id, const std::map<AscendString, ge::Tensor> &params_list);
}

// Initialize GE
ATTRIBUTED_DEPRECATED(GE_FUNC_VISIBILITY Status GEInitialize(const std::map<AscendString, AscendString> &))
GE_FUNC_VISIBILITY Status GEInitialize(const std::map<std::string, std::string> &options);

GE_FUNC_VISIBILITY Status GEInitialize(const std::map<AscendString, AscendString> &options);

// Finalize GE, release all resources
GE_FUNC_VISIBILITY Status GEFinalize();

GE_FUNC_VISIBILITY std::string GEGetErrorMsg();

GE_FUNC_VISIBILITY std::string GEGetWarningMsg();

GE_FUNC_VISIBILITY Status GetModelDistributeDesc(const void *data, const uint64_t length,
                                                 ModelDistibuteDesc &model_dist_desc);

class GE_FUNC_VISIBILITY Session {
 public:
  ATTRIBUTED_DEPRECATED(Session(const std::map<AscendString, AscendString> &))
  explicit Session(const std::map<std::string, std::string> &options);

  explicit Session(const std::map<AscendString, AscendString> &options);

  ~Session();

  ///
  /// @ingroup client
  /// @brief add a graph with a specific graph id
  /// @param [in] graph_id graph id
  /// @return Status result of function
  ///
  Status AddGraph(uint32_t graph_id, const Graph &graph);

  ///
  /// @ingroup client
  /// @brief add a graph with a specific graph id and graphOptions
  /// @param [in] graphId graph id
  /// @param [in] graph the graph
  /// @param [in] options graph options
  /// @return Status result of function
  ///
  ATTRIBUTED_DEPRECATED(Status AddGraph(uint32_t, const Graph &, const std::map<AscendString, AscendString> &))
  Status AddGraph(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

  ///
  /// @ingroup client
  /// @brief add a graph with a specific graphId and graphOptions
  /// @param [in] graphId graph id
  /// @param [in] graph the graph
  /// @param [in] options graph options
  /// @return Status result of function
  ///
  Status AddGraph(uint32_t graph_id, const Graph &graph, const std::map<AscendString, AscendString> &options);

  ///
  /// @ingroup client
  /// @brief add a copy graph with a specific graphId
  /// @param [in] graphId graph id
  /// @param [in] graph the graph
  /// @return Status result of function
  ///
  Status AddGraphWithCopy(uint32_t graph_id, const Graph &graph);

  ///
  /// @ingroup client
  /// @brief add a copy graph with a specific graphId and graphOptions
  /// @param [in] graphId graph id
  /// @param [in] graph the graph
  /// @param [in] options graph options
  /// @return Status result of function
  ///
  Status AddGraphWithCopy(uint32_t graph_id, const Graph &graph, const std::map<AscendString, AscendString> &options);

  ///
  /// @ingroup ge_graph
  /// @brief remove a graph of the session with specific session id
  /// @param [in] graph_d graph id
  /// @return Status result of function
  ///
  Status RemoveGraph(uint32_t graph_id);

  ///
  /// @ingroup ge_graph
  /// @brief run a graph of the session with specific session id
  /// @param [in] graphId graph id
  /// @param [in] inputs input data
  /// @param [out] outputs output data
  /// @return Status result of function
  ///
  Status RunGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs);

  ///
  /// @ingroup ge_graph
  /// @brief run a graph of the session with specific session id
  /// @param [in] graphId graph id
  /// @param [in] inputs input data list
  /// @param [out] outputs output data
  /// @return Status result of function
  ///
  Status RunGraph(uint32_t graph_id, const ContinuousTensorList &inputs, std::vector<Tensor> &outputs);

  ///
  /// @ingroup ge_graph
  /// @brief Load graph from om
  /// @param [in] graphId graph id
  /// @param [in] options graph options
  /// @param [in] om_file_path offline om path
  /// @return Status result of function
  ///
  /* 规避方案：规避acl系列接口不支持拉远环境；aclModelLoad接口当前不支持分布式的模型加载，不支持session管理权重变量复用
              根据20230615 SEG会议纪要可以在session中开接口作为临时方案
     方案详述：acl接口不支持拉远形态，不支持外置权重。使用该接口实现在session加载离线om。
              1.将om根据指定om_file_path直接加载到ModelManager中
              2.modelManager判断是异构模型，调用异构部署函数，生成deloyplan
              3.ModelManger提供接口返回modelid和flowModelPtr GraphPtr，InnerSession根据返回信息注册在GraphManager中生成graphnode
              4.当前调用进程可以通过RunGraph接口去执行加载的离线模型
     方案约束：只用于加载异构离线模型，即存储格式为flowmodel下包含一个或多个submodel和modelrelation的离线模型;
              不支持包含variable的离线模型，variable的值在device侧目前还没有保存到om中的方案
  */
  Status LoadGraph(const uint32_t graph_id, const std::map<std::string, std::string> options,
                   const std::string om_file_path);

  ///
  /// @ingroup ge_graph
  /// @brief run a graph of the session with specific session id and specific stream asynchronously
  /// @param [in] graph_id graph id
  /// @param [in] stream specific stream
  /// @param [in] inputs input data
  /// @param [out] outputs output data
  /// @return Status result of function
  ///
  Status RunGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<Tensor> &inputs,
                                 std::vector<Tensor> &outputs);

  ///
  /// @ingroup ge_graph
  /// @brief build graph in the session with specific session id
  /// @param [in] graphId: graph id
  /// @param [in] inputs: input data
  /// @return Status result of function
  ///
  Status BuildGraph(uint32_t graph_id, const std::vector<InputTensorInfo> &inputs);

  Status BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs);  /*lint !e148*/

  ///
  /// @ingroup ge_graph
  /// @brief run graph in the session with specific session id asynchronously
  /// @param [in] graphId: graph id
  /// @param [in] inputs: input data
  /// @param [out] callback: callback while runing graph has been finished.
  ///                        The callback function will not be checked.
  ///                        Please ensure that the implementation of the function is trusted.
  /// @return Status result of function
  ///
  Status RunGraphAsync(uint32_t graph_id, const std::vector<ge::Tensor> &inputs, RunAsyncCallback callback);

  ///
  /// @ingroup ge_graph
  /// @brief run graph in the session with specific session id asynchronously
  /// @param [in] graphId: graph id
  /// @param [in] inputs: input data list
  /// @param [out] callback: callback while runing graph has been finished.
  ///                        The callback function will not be checked.
  ///                        Please ensure that the implementation of the function is trusted.
  /// @return Status result of function
  ///
  Status RunGraphAsync(uint32_t graph_id, const ContinuousTensorList &inputs, RunAsyncCallback callback);

  ///
  /// @ingroup ge_graph
  /// @brief get variables in the session with specific session id
  /// @param [in] var_names: variable names
  /// @param [out] var_values: variable values
  /// @return Status result of function
  ///
  ATTRIBUTED_DEPRECATED(Status GetVariables(const std::vector<std::string> &, std::vector<Tensor> &))
  Status GetVariables(const std::vector<std::string> &var_names, std::vector<Tensor> &var_values);

  ///
  /// @ingroup ge_graph
  /// @brief get variables in the session with specific session id
  /// @param [in] var_names: variable names
  /// @param [out] var_values: variable values
  /// @return Status result of function
  ///
  Status GetVariables(const std::vector<AscendString> &var_names, std::vector<Tensor> &var_values);

  ///
  /// @ingroup ge_graph
  /// @brief register callback func with specific summary or checkpoint by users
  /// @param [in] key: func key
  /// @param [in] callback: callback  specific summary or checkpoint.
  ///                       The callback function will not be checked.
  ///                       Please ensure that the implementation of the function is trusted.
  /// @return Status result of function
  ///
  ATTRIBUTED_DEPRECATED(Status RegisterCallBackFunc(const char *, const session::pCallBackFunc &))
  Status RegisterCallBackFunc(const std::string &key, const pCallBackFunc &callback);

  Status RegisterCallBackFunc(const char *key, const session::pCallBackFunc &callback);

  bool IsGraphNeedRebuild(uint32_t graph_id);

  uint64_t GetSessionId() const;

  /// @ingroup ge_graph
  /// @brief Feed input data to graph.
  /// @param [in] graph_id graph id
  /// @param [in] inputs input data
  /// @param [in] info intput data flow flag
  /// @param [in] timeout data feed timeout(ms), -1 means never timeout
  /// @return Status result of function
  Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, const DataFlowInfo &info,
                           int32_t timeout);

  /// @ingroup ge_graph
  /// @brief Feed input data to graph.
  /// @param [in] graph_id graph id
  /// @param [in] indexes fetch output data order(index cannot be duplicated)
  /// @param [in] inputs input data
  /// @param [in] info intput data flow flag
  /// @param [in] timeout data feed timeout(ms), -1 means never timeout
  /// @return Status result of function
  Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes, const std::vector<Tensor> &inputs,
                           const DataFlowInfo &info, int32_t timeout);

  /// @ingroup ge_graph
  /// @brief Fetch graph output data in order.
  /// @param [in] graph_id graph id
  /// @param [out] outputs output data
  /// @param [out] info output data flow flag
  /// @param [in] timeout data fetch timeout(ms), -1 means never timeout
  /// @return Status result of function
  Status FetchDataFlowGraph(uint32_t graph_id, std::vector<Tensor> &outputs, DataFlowInfo &info, int32_t timeout);

  /// @ingroup ge_graph
  /// @brief Fetch graph output data in order.
  /// @param [in] graph_id graph id
  /// @param [in] indexes fetch output data order(index cannot be duplicated)
  /// @param [out] outputs output data
  /// @param [out] info output data flow flag
  /// @param [in] timeout data ftech timeout(ms), -1 means never timeout
  /// @return Status result of function
  Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes, std::vector<Tensor> &outputs,
                            DataFlowInfo &info, int32_t timeout);

  /// @ingroup ge_graph
  /// @brief compile graph in the session with specific session id
  /// @param [in] graphId: graph id
  /// @return Status result of function
  Status CompileGraph(uint32_t graph_id);

  ///
  /// @ingroup ge_graph
  /// @brief get graph resource summary after compiled
  /// @param [in] graphId: graph id
  /// @return share_ptr of CompiledGraphSummary
  ///
  CompiledGraphSummaryPtr GetCompiledGraphSummary(uint32_t graph_id);

  ///
  /// @ingroup ge_graph
  /// @brief set const memory base after compiled and before loaded, only allows setting once
  /// @param [in] graphId graph id
  /// @param [in] memory const memory base
  /// @param [out] size const memory size
  /// @return Status result of function
  ///
  Status SetGraphConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  ///
  /// @ingroup ge_graph
  /// @brief set or update fearture memory base after compiled
  /// @param [in] graphId graph id
  /// @param [in] memory feature map memory base, without input and output mem
  /// @param [out] size feature map memory size
  /// @return Status result of function
  ///
  Status UpdateGraphFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  /// @ingroup ge_graph
  /// @brief register external allocator to GE.
  /// @param [in] stream stream handle
  /// @param [in] allocator_obj allocator object handle
  /// @return Status result of function
  Status RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const;

  /// @ingroup ge_graph
  /// @brief unregister external allocator to GE.
  /// @param [in] stream stream handle
  /// @return Status result of function
  Status UnregisterExternalAllocator(const void *const stream) const;

 private:
  uint64_t sessionId_;
};
}  // namespace ge

#endif  // INC_EXTERNAL_GE_GE_API_H_
