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

#include "hybrid_model_executor.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"

namespace ge {
namespace hybrid {
namespace {
const int kIntBase = 10;
const char *const kEnvProfilingLevel = "HYBRID_PROFILING_LEVEL";
}  // namespace
HybridModelExecutor::HybridModelExecutor(HybridModel *model, uint32_t device_id, rtStream_t stream)
    : model_(model), device_id_(device_id), stream_(stream) {}

Status HybridModelExecutor::Init() {
  GELOGD("Start to init HybridGraphEngine.");
  GE_CHK_STATUS_RET_NOLOG(InitExecutionContext());
  GELOGD("HybridGraphEngine initialized successfully.");
  return SUCCESS;
}

Status HybridModelExecutor::Execute(HybridModelExecutor::ExecuteArgs &args) {
  GELOGD("Start to execute model.");
  auto root_graph_item = model_->GetRootGraphItem();
  GE_CHECK_NOTNULL(root_graph_item);

  SubgraphExecutor executor(model_->GetRootGraphItem(), &context_);
  auto ret = ExecuteGraphInternal(executor, args);
  Cleanup();
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Cleanup] End");
  GE_CHK_STATUS_RET(ret, "Failed to execute model");
  GELOGD("Model executed successfully.");

  if (context_.profiler != nullptr) {
    context_.profiler->Dump(std::cout);
    context_.profiler->Reset();
  }

  context_.iteration += 1;
  return SUCCESS;
}

Status HybridModelExecutor::ExecuteGraphInternal(SubgraphExecutor &executor, HybridModelExecutor::ExecuteArgs &args) {
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[InitContext] Start");
  GE_CHK_STATUS_RET_NOLOG(ResetExecutionContext(context_));
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[InitContext] End");

  GE_CHK_STATUS_RET(executor.ExecuteAsync(args.inputs, args.input_desc), "Failed to execute partitioned call.");
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[ExecuteAsync] End");

  GE_CHK_STATUS_RET(executor.Synchronize(), "Failed to sync root graph.");
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Synchronize] End");

  GE_CHK_STATUS_RET(executor.GetOutputs(args.outputs, args.output_desc), "Failed to get outputs");
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[GetOutput] End");
  return SUCCESS;
}

Status HybridModelExecutor::Cleanup() {
  GELOGD("Start to cleanup.");
  context_.callback_manager->Destroy();
  RuntimeInferenceContext::DestroyContext(to_string(context_.session_id));
  GELOGD("Cleanup successfully.");
  return SUCCESS;
}

Status HybridModelExecutor::InitExecutionContext() {
  GE_CHK_RT_RET(rtCtxGetCurrent(&context_.rt_context));
  GE_CHK_RT_RET(rtCtxCreate(&context_.rt_gen_context, RT_CTX_GEN_MODE, 0));
  GE_CHK_RT_RET(rtCtxSetCurrent(context_.rt_context));

  context_.stream = stream_;
  context_.model = model_;
  context_.session_id = ::ge::GetContext().SessionId();
  GELOGD("session id from model = %lu, from context = %lu", model_->GetSessionId(), context_.session_id);
  context_.allocator = NpuMemoryAllocator::GetAllocator(device_id_);
  GE_CHECK_NOTNULL(context_.allocator);
  context_.callback_manager = std::unique_ptr<CallbackManager>(new (std::nothrow) CallbackManager(stream_));
  GE_CHECK_NOTNULL(context_.callback_manager);
  context_.dump_properties = PropertiesManager::Instance().GetDumpProperties(context_.session_id);
  const char *profiling_level = std::getenv(kEnvProfilingLevel);
  if (profiling_level != nullptr) {
    context_.profiling_level = std::strtol(profiling_level, nullptr, kIntBase);
    GELOGD("Got profiling level = %d", context_.profiling_level);
    if (context_.profiling_level > 0) {
      context_.profiler.reset(new (std::nothrow) HybridProfiler());
      GE_CHECK_NOTNULL(context_.profiler);
    }
  }

  if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    context_.trace_enabled = true;
  }
  return SUCCESS;
}

Status HybridModelExecutor::ResetExecutionContext(GraphExecutionContext &context) {
  GE_CHK_STATUS_RET_NOLOG(context.callback_manager->Init());
  string ctx_id = std::to_string(context.session_id);
  RuntimeInferenceContext::DestroyContext(ctx_id);
  GE_CHK_GRAPH_STATUS_RET(RuntimeInferenceContext::CreateContext(ctx_id), "Failed to Destroy RuntimeInferenceContext");
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
