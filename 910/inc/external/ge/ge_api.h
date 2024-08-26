/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

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
#include "exe_graph/runtime/tensor.h"
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

GE_FUNC_VISIBILITY ge::AscendString GEGetErrorMsgV2();

GE_FUNC_VISIBILITY std::string GEGetWarningMsg();

GE_FUNC_VISIBILITY ge::AscendString GEGetWarningMsgV2();

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
  /// @brief run a distributed graph of the session with specific session id
  /// @param [in] graphId graph id
  /// @param [in] device id respond to inputs input data
  /// @param [out] device id respond to outputs output data
  /// @return Status result of function
  ///
  Status RunGraphDistribute(uint32_t graph_id, const std::map<int32_t, std::vector<Tensor>> &device_to_inputs,
                            std::map<int32_t, std::vector<Tensor>> &device_to_outputs);

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
  Status LoadGraph(const uint32_t graph_id, const std::map<std::string, std::string> &options,
                   const std::string &om_file_path) const;

  ///
  /// @ingroup ge_graph
  /// @brief load a graph of the session with specific session id and specific stream, 
  /// @param [in] graph_id graph id
  /// @param [in] stream specific stream
  /// @param [in] options graph options
  /// @return Status result of function
  ///
  /* 
     方案约束：只用于加载已经完成CompileGraph的图，不支持重复加载图，
     使用方法：CompileGraph + LoadGraph + ExecuteGraphWithStreamAsync
  */
  Status LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options,
                   void *stream) const;

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
  
  Status ExecuteGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<gert::Tensor> &inputs,
                                 std::vector<gert::Tensor> &outputs);

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
  /// @brief Feed input data to graph.
  /// @param [in] graph_id graph id
  /// @param [in] raw_data_list can be 1 or n, feed will be combine n raw data automatically
  /// @param [in] indexes feed input index
  /// @param [in] info intput data flow flag
  /// @param [in] timeout data feed timeout(ms), -1 means never timeout
  /// @return Status result of function
  Status FeedRawData(uint32_t graph_id, const std::vector<RawData> &raw_data_list, const uint32_t index,
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

  ///
  /// @ingroup ge_graph
  /// @brief set fix feature memory base after compiled and before loaded, only allows setting once
  /// @param [in] graphId graph id
  /// @param [in] memory const memory base
  /// @param [out] size const memory size
  /// @return Status result of function
  ///
  Status SetGraphFixedFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  ///
  /// @ingroup ge_graph
  /// @brief set fix feature memory base with type after compiled and before loaded, one type only allows setting once
  /// @param [in] graph_id graph id
  /// @param [in] type memory type
  /// @param [in] memory const memory base
  /// @param [out] size const memory size
  /// @return Status result of function
  ///
  Status SetGraphFixedFeatureMemoryBaseWithType(uint32_t graph_id, MemoryType type, const void *const memory,
                                                size_t size);

  ///
  /// @ingroup ge_graph
  /// @brief set or update tefreshable fearture memory base after compiled, not include fix memory
  /// @param [in] graphId graph id
  /// @param [in] memory feature map memory base, without input and output mem
  /// @param [out] size feature map memory size
  /// @return Status result of function
  ///
  Status UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

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

  /// @ingroup ge_graph
  /// @brief shard graphs in the session according the add graph sequence
  /// @return Status result of function
  Status ShardGraphsToFile(const char_t *file_path = "./") const;

  /// @ingroup ge_graph
  /// @brief shard graphs in the session according the add graph sequence
  /// @return Status result of function
  Status ShardGraphs() const;

  /// @ingroup ge_graph
  /// @brief save graphs in the session with specific file path
  /// @return Status result of function
  Status SaveGraphsToPb(const char_t *file_path) const;

  /// @ingroup ge_graph
  /// @brief virtual memory need to remap
  /// @param [in] va virtual memory
  /// @param [in] new_pa new physical memory
  /// @param [in] len the lens of va to remap
  /// @return Status result of function
  Status PaRemapped(const uint64_t va, const uint64_t new_pa, const uint64_t len) const;

 private:
  uint64_t sessionId_{0};
};
}  // namespace ge
extern "C" {
ge::Status GeSessionLoadGraph(ge::Session &session, uint32_t graph_id,
                              const std::map<ge::AscendString, ge::AscendString> &options,
                              void *stream);

ge::Status GeSessionExecuteGraphWithStreamAsync(ge::Session &session, uint32_t graph_id, void *stream,
                                                const std::vector<gert::Tensor> &inputs,
                                                std::vector<gert::Tensor> &outputs);
} // extern "C"

#endif  // INC_EXTERNAL_GE_GE_API_H_
