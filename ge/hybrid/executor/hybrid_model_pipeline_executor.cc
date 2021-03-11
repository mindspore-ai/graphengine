#include "hybrid_model_pipeline_executor.h"

#include "common/math/math_util.h"
#include "common/dump/dump_manager.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int kNumExecutors = 2;
const int kMinLoopCount = 2;
const int kIntBase = 10;
const char *const kEnvProfilingLevel = "HYBRID_PROFILING_LEVEL";
}

StageExecutor::StageExecutor(int id, HybridModel *model, PipeExecutionConfig *config)
    : id_(id), model_(model), pipe_config_(config) {}

StageExecutor::~StageExecutor() { GELOGD("~StageExecutor(), id = %d", id_); }

Status StageExecutor::Init() {
  GELOGD("[Executor: %d] Start to init StateExecutor", id_);
  context_.rt_context = pipe_config_->rt_context;
  GE_CHK_STATUS_RET_NOLOG(InitExecutionContext());
  GE_CHK_RT_RET(rtStreamCreate(&stream_, RT_STREAM_PRIORITY_DEFAULT));
  context_.stream = stream_;

  root_graph_executor_.reset(new (std::nothrow) SubgraphExecutor(model_->GetRootGraphItem(), &context_));
  GE_CHECK_NOTNULL(root_graph_executor_);

  GELOGD("[Executor: %d] Init stage executor successfully", id_);
  return SUCCESS;
}

Status StageExecutor::ResetExecutionContext(GraphExecutionContext &context) {
  GE_CHK_STATUS_RET_NOLOG(context.callback_manager->Init());
  string ctx_id = std::to_string(context.context_id);
  RuntimeInferenceContext::DestroyContext(ctx_id);
  GE_CHK_GRAPH_STATUS_RET(RuntimeInferenceContext::CreateContext(ctx_id), "Failed to Destroy RuntimeInferenceContext");
  return SUCCESS;
}

Status StageExecutor::Start(const std::vector<TensorValue> &inputs, const std::vector<ConstGeTensorDescPtr> &input_desc,
                            int iteration_count) {
  GELOGD("Start");
  GE_CHK_RT_RET(rtCtxSetCurrent(context_.rt_context));
  int num_loops = iteration_count / pipe_config_->num_executors;
  if (id_ < iteration_count % iteration_count) {
    num_loops += 1;
  }
  FMK_INT32_MULCHECK(num_loops, pipe_config_->num_stages);
  num_loops *= pipe_config_->num_stages;
  GELOGD("[Executor: %d] loop count = %d", id_, num_loops);

  for (int loop_idx = 0; loop_idx < num_loops; ++loop_idx) {
    GELOGD("[Executor: %d] Start to wait for task.", id_);
    StageTask task_info;
    task_queue_.Pop(task_info);
    GELOGD("[Executor: %d] Got task, stage = %d, iteration = %ld", id_, task_info.stage, task_info.iteration);
    if (task_info.iteration >= pipe_config_->iteration_end) {
      GELOGE(INTERNAL_ERROR, "[Executor: %d] Unexpected iteration: %d", id_, task_info.iteration);
      return INTERNAL_ERROR;
    }

    if (task_info.event != nullptr) {
      GELOGD("[%d] Add StreamWaitEvent", id_);
      GE_CHK_RT_RET(rtStreamWaitEvent(stream_, task_info.event));
      RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %d] [Stage = %d] End", task_info.iteration - 1,
                                   task_info.stage);
    }

    RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %d] [Stage = %d] Start", task_info.iteration,
                                 task_info.stage);

    if (task_info.stage == 0) {
      GELOGD("[Executor: %d] To ResetExecutionContext", id_);
      GE_CHK_STATUS_RET(ResetExecutionContext(context_), "[Executor: %d] Failed to reset context", id_);
      context_.iteration = task_info.iteration;
      GE_CHK_STATUS_RET_NOLOG(SetInputs(inputs, input_desc));
    }

    RECORD_MODEL_EXECUTION_EVENT(&context_, "[Stage = %d] PartialExecuteAsync Start", task_info.stage);
    GE_CHK_STATUS_RET(root_graph_executor_->PartialExecuteAsync(task_info.stage));
    RECORD_MODEL_EXECUTION_EVENT(&context_, "[Stage = %d] PartialExecuteAsync End", task_info.stage);
    GELOGD("[Executor: %d] PartialExecuteAsync successfully.", id_);

    // notify next execution unit
    StageTask next_task;
    next_task.stage = task_info.stage;
    next_task.iteration = task_info.iteration + 1;

    auto sync_result = Synchronize();
    if (sync_result != SUCCESS) {
      GELOGE(sync_result, "[Executor: %d] Failed to sync result. iteration = %d", id_, task_info.iteration);

      context_.profiler->Dump(std::cout);
      context_.callback_manager->Destroy();
      RuntimeInferenceContext::DestroyContext(std::to_string(context_.context_id));
      return sync_result;
    }

    RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %d] [Stage = %d] End", task_info.iteration, task_info.stage);

    // if not end stage
    if (task_info.stage >= pipe_config_->num_stages - 1) {
      RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %d] Schedule End", task_info.iteration);
      GELOGD("[Executor: %d] End of iteration [%ld]", id_, task_info.iteration);
      context_.callback_manager->Destroy();
      RuntimeInferenceContext::DestroyContext(std::to_string(context_.context_id));
    }
    next_executor_->ExecuteAsync(next_task);
    GELOGD("[Executor: %d] Push item successfully.", id_);
  }

  GELOGD("[Executor: %d] Process task ended.", id_);
  return SUCCESS;
}

Status StageExecutor::ExecuteAsync(const StageTask &args) {
  (void)task_queue_.Push(args);
  return SUCCESS;
}

Status StageExecutor::Synchronize() {
  auto ret = root_graph_executor_->Synchronize();
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Synchronize] End, ret = %u", ret);
  return ret;
}

HybridModelPipelineExecutor::HybridModelPipelineExecutor(HybridModel *model, uint32_t device_id)
    : model_(model), device_id_(device_id) {
  config_.num_executors = kNumExecutors;
  config_.num_stages = model_->GetRootGraphItem()->NumGroups();
  config_.device_id = device_id_;
}

Status StageExecutor::InitExecutionContext() {
  GE_CHK_RT_RET(rtCtxCreate(&context_.rt_gen_context, RT_CTX_GEN_MODE, 0));
  GE_CHK_RT_RET(rtCtxSetCurrent(context_.rt_context));

  context_.model = model_;
  context_.session_id = ::ge::GetContext().SessionId();
  GELOGD("session id from model = %lu, from context = %lu", model_->GetSessionId(), context_.session_id);
  context_.allocator = NpuMemoryAllocator::GetAllocator(pipe_config_->device_id);
  GE_CHECK_NOTNULL(context_.allocator);
  context_.callback_manager = std::unique_ptr<CallbackManager>(new (std::nothrow) CallbackManager());
  GE_CHECK_NOTNULL(context_.callback_manager);
  context_.dump_properties = DumpManager::GetInstance().GetDumpProperties(context_.session_id);
  if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    context_.trace_enabled = true;
  }
  return SUCCESS;
}

Status StageExecutor::SetInputs(const vector<TensorValue> &inputs, const vector<ConstGeTensorDescPtr> &input_desc) {
  root_graph_executor_->InitForPartialExecution(inputs, input_desc);
  return SUCCESS;
}

Status StageExecutor::GetOutputs(vector<TensorValue> &outputs, vector<ConstGeTensorDescPtr> &output_desc) {
  return root_graph_executor_->GetOutputs(outputs, output_desc);
}

void StageExecutor::Reset() {
  task_queue_.Stop();
  task_queue_.Clear();
  task_queue_.Restart();
}

Status HybridModelPipelineExecutor::Init() {
  const char *profiling_level = std::getenv(kEnvProfilingLevel);
  if (profiling_level != nullptr) {
    context_.profiling_level = std::strtol(profiling_level, nullptr, kIntBase);
    GELOGD("Got profiling level = %ld", context_.profiling_level);
    if (context_.profiling_level > 0) {
      context_.profiler.reset(new (std::nothrow) HybridProfiler());
      GE_CHECK_NOTNULL(context_.profiler);
    }
  }

  GELOGD("Number of stages = %d, number of executors = %d", config_.num_stages, config_.num_executors);
  GE_CHK_RT_RET(rtCtxGetCurrent(&config_.rt_context));
  GE_CHK_STATUS_RET_NOLOG(InitStageExecutors());
  return SUCCESS;
}

Status HybridModelPipelineExecutor::InitStageExecutors() {
  for (int i = 0; i < config_.num_executors; ++i) {
    auto stage_executor = std::unique_ptr<StageExecutor>(new (std::nothrow) StageExecutor(i, model_, &config_));
    GE_CHECK_NOTNULL(stage_executor);
    GE_CHK_STATUS_RET_NOLOG(stage_executor->Init());

    if (context_.profiler != nullptr) {
      // will call unique_ptr::release later
      stage_executor->context_.profiler.reset(context_.profiler.get());
      stage_executor->context_.profiling_level = context_.profiling_level;
    }

    stage_executors_.emplace_back(std::move(stage_executor));
  }

  // build propagation loop
  for (int i = 0; i < config_.num_executors - 1; ++i) {
    stage_executors_[i]->SetNext(stage_executors_[i + 1].get());
  }
  stage_executors_[config_.num_executors - 1]->SetNext(stage_executors_[0].get());
  return SUCCESS;
}

Status HybridModelPipelineExecutor::Execute(HybridModelExecutor::ExecuteArgs &args) {
  int loop_count = args.num_loops;
  GE_CHECK_GE(loop_count, kMinLoopCount);

  auto &inputs = args.inputs;
  auto &input_desc = args.input_desc;
  // Start schedulers
  std::vector<std::future<Status>> futures;
  for (size_t i = 0; i < stage_executors_.size(); ++i) {
    GELOGD("Starting executor %zu", i);
    auto executor = stage_executors_[i].get();
    executor->Reset();
    auto future = std::async(
        [loop_count, executor, inputs, input_desc]() { return executor->Start(inputs, input_desc, loop_count); });

    futures.emplace_back(std::move(future));
  }

  // Push initial tasks
  GELOGD("Start to execute with loops, loop count = %d", loop_count);
  config_.iteration_end = iteration_ + loop_count;
  for (int i = 0; i < config_.num_stages; ++i) {
    StageExecutor::StageTask task_info;
    task_info.stage = i;
    task_info.iteration = iteration_;
    stage_executors_[0]->ExecuteAsync(task_info);
  }

  // Wait for end of iterations
  bool has_error = false;
  for (size_t i = 0; i < stage_executors_.size(); ++i) {
    GELOGD("Start to sync result of executor[%zu]", i);
    auto ret = futures[i].get();
    if (ret != SUCCESS) {
      GELOGE(ret, "[Executor: %zu] Failed to schedule tasks.", i);
      has_error = true;
      continue;
    }

    ret = stage_executors_[i]->Synchronize();

    if (ret != SUCCESS) {
      GELOGE(ret, "[Executor: %zu] Failed to synchronize result.", i);
      has_error = true;
      continue;
    }
  }

  // record for profiling analyzer
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Cleanup] End");

  if (context_.profiler != nullptr) {
    context_.profiler->Dump(std::cout);
  }

  iteration_ = config_.iteration_end;

  if (has_error) {
    GELOGE(FAILED, "Error occurred while execution");
    return FAILED;
  }

  auto last_iter_executor_idx = loop_count % stage_executors_.size();
  GE_CHK_STATUS_RET(stage_executors_[last_iter_executor_idx]->GetOutputs(args.outputs, args.output_desc),
                    "Failed to get output from executor[%zu]", last_iter_executor_idx);
  return SUCCESS;
}

HybridModelPipelineExecutor::~HybridModelPipelineExecutor() {
  GELOGD("~HybridModelPipelineExecutor()");
  for (auto &executor : stage_executors_) {
    (void)executor->context_.profiler.release();
  }
}
}  // namespace hybrid
}  // namespace ge
