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
HybridModelExecutor::HybridModelExecutor(HybridModel *model, uint32_t device_id, rtStream_t stream)
    : model_(model), device_id_(device_id), stream_(stream) {}

Status HybridModelExecutor::Init() {
  GELOGD("Start to init HybridGraphEngine.");
  GE_CHK_STATUS_RET_NOLOG(InitExecutionContext());
  infer_shape_engine_.reset(new (std::nothrow) ShapeInferenceEngine(&context_));
  compile_engine_.reset(new (std::nothrow) TaskCompileEngine(&context_));
  execute_engine_.reset(new (std::nothrow) ExecutionEngine(&context_, context_.callback_manager.get()));
  GE_CHK_STATUS_RET_NOLOG(compile_engine_->Init());
  GELOGD("HybridGraphEngine initialized successfully.");
  return SUCCESS;
}

Status HybridModelExecutor::Execute(HybridModelExecutor::ExecuteArgs &args) {
  GELOGD("Start to execute model.");
  auto ret = ExecuteGraphInternal(args);
  Cleanup();
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Cleanup] End");
  GE_CHK_STATUS_RET(ret, "Failed to execute model");
  GELOGD("Model executed successfully.");

  if (context_.profiler != nullptr) {
    context_.profiler->Reset();
  }

  return SUCCESS;
}

Status HybridModelExecutor::ExecuteGraphInternal(HybridModelExecutor::ExecuteArgs &args) {
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[InitContext] Start");
  GE_CHK_STATUS_RET_NOLOG(ResetExecutionContext(context_));
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[InitContext] End");
  GE_CHK_STATUS_RET_NOLOG(InitInputsAndOutputs(args, context_));
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[InitInputsAndOutputs] End");
  GE_CHK_STATUS_RET_NOLOG(compile_engine_->Start(pool_));
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[CompileProcess] Started");
  GE_CHK_STATUS_RET_NOLOG(infer_shape_engine_->Start(pool_));
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[InferShapeProcess] Started");
  GE_CHK_STATUS_RET(execute_engine_->Start(), "Run execution engine failed.");
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[ExecutionProcess] End");
  GE_CHK_STATUS_RET_NOLOG(Synchronize());
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Synchronize] End");
  GE_CHK_STATUS_RET_NOLOG(GetOutput(args));
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[GetOutput] End");
  return SUCCESS;
}

Status HybridModelExecutor::Cleanup() {
  GELOGD("Start to cleanup.");
  context_.callback_manager->Destroy();
  context_.cv_manager.Reset();
  context_.node_states.clear();
  context_.all_inputs.clear();
  context_.all_outputs.clear();
  context_.compile_queue.Clear();
  context_.execution_queue.Clear();
  RuntimeInferenceContext::DestroyContext(to_string(context_.session_id));
  GELOGD("Cleanup successfully.");
  return SUCCESS;
}

Status HybridModelExecutor::InitExecutionContext() {
  context_.stream = stream_;
  context_.model = model_;
  context_.session_id = ::ge::GetContext().SessionId();
  GELOGD("session id from model = %lu, from context = %lu", model_->GetSessionId(), context_.session_id);
  context_.allocator = NpuMemoryAllocator::GetAllocator(device_id_);
  GE_CHECK_NOTNULL(context_.allocator);
  context_.callback_manager = std::unique_ptr<CallbackManager>(new (std::nothrow) CallbackManager(stream_));
  GE_CHECK_NOTNULL(context_.callback_manager);
  if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    context_.trace_enabled = true;
  }

  return SUCCESS;
}

Status HybridModelExecutor::ResetExecutionContext(GraphExecutionContext &context) {
  auto &model = *context.model;
  context.all_inputs.resize(model.TotalInputs());
  context.all_outputs.resize(model.TotalOutputs());
  context.compile_queue.Restart();
  context.execution_queue.Restart();
  GE_CHK_STATUS_RET_NOLOG(context.callback_manager->Init());

  // TODO do not re-assign Consts every run
  for (auto const_node : model.GetConstNodes()) {
    auto weight_tensor = model.GetWeight(const_node);
    GE_CHECK_NOTNULL(weight_tensor);
    for (auto &dst_aid_and_nid : const_node->outputs[0]) {
      auto *dst_node_item = dst_aid_and_nid.second;
      auto input_offset = dst_node_item->input_start + dst_aid_and_nid.first;
      context.all_inputs[input_offset] = *weight_tensor;
    }
  }

  string ctx_id = std::to_string(context.session_id);
  RuntimeInferenceContext::DestroyContext(ctx_id);
  GE_CHK_GRAPH_STATUS_RET(RuntimeInferenceContext::CreateContext(ctx_id), "Failed to Destroy RuntimeInferenceContext");
  return SUCCESS;
}

Status HybridModelExecutor::InitInputsAndOutputs(HybridModelExecutor::ExecuteArgs &args,
                                                 GraphExecutionContext &context) {
  for (const auto &it : model_->GetInputNodes()) {
    uint32_t input_index = it.first;
    if (input_index >= args.inputs.size()) {
      GELOGE(PARAM_INVALID, "Not enough inputs. NumInputs = %zu, but input index = %u", args.inputs.size(),
             input_index);
      return PARAM_INVALID;
    }

    auto node_item = it.second;
    auto &input_tensor = args.inputs[input_index];
    GELOGD("Set input tensor[%u] to inputs with index = %d, addr = %p, size = %zu", input_index, node_item->input_start,
           input_tensor.GetData(), input_tensor.GetSize());
    context.all_inputs[node_item->input_start] = input_tensor;
  }

  for (size_t i = 0; i < model_->GetOutputOffsets().size(); ++i) {
    auto offset = model_->GetOutputOffsets()[i];
    if (i < args.outputs.size() && args.outputs[i].GetData() != nullptr) {
      GELOGD("Use user allocated output memory. output index = %zu, output offset = %d", i, offset);
      context.all_outputs[offset] = args.outputs[i];
    }
  }

  return SUCCESS;
}

Status HybridModelExecutor::Synchronize() {
  GE_CHK_RT_RET(rtStreamSynchronize(stream_));
  return SUCCESS;
}

Status HybridModelExecutor::GetOutput(HybridModelExecutor::ExecuteArgs &args) {
  auto &net_output_input_offsets = model_->GetNetOutputInputOffsets();
  auto num_outputs = net_output_input_offsets.size();
  args.outputs.resize(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    auto offset = net_output_input_offsets[i];
    GELOGI("Get output[%zu] from offset %d", i, offset);
    args.outputs[i] = context_.all_inputs[offset];
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
