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

#include "aicore_task_builder.h"
#include <mutex>
#include "graph/op_desc.h"
#include "cce/taskdown_common.hpp"
#include "framework/common/debug/log.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace hybrid {
std::mutex g_reg_mutex;

AiCoreTaskBuilder::AiCoreTaskBuilder(const OpDescPtr &op_desc, const domi::KernelDef &kernel_def)
    : op_desc_(op_desc), kernel_def_(kernel_def) {
  std::string session_graph_id;
  GE_IF_BOOL_EXEC(AttrUtils::GetStr(*op_desc_, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                  GELOGD("Get original type of session_graph_id."));
  // get bin_file_key
  stub_name_ = (session_graph_id.empty()) ? op_desc_->GetName() : session_graph_id + "_" + op_desc_->GetName();
}

Status AiCoreTaskBuilder::SetKernelArgs(AiCoreOpTask &task) {
  const domi::KernelContext &context = kernel_def_.context();
  // get kernel_type
  auto kernel_type = static_cast<cce::ccKernelType>(context.kernel_type());
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(kernel_type != cce::ccKernelType::TE, return UNSUPPORTED,
                                 "Invalid kernel type[%d] in AiCore TaskDef.", static_cast<int>(kernel_type));

  task.args_size_ = kernel_def_.args_size();
  task.block_dim_ = kernel_def_.block_dim();

  // malloc args memory
  task.args_.reset(new (std::nothrow) uint8_t[task.args_size_]);
  // task.args_ = std::make_unique<uint8_t>(task.args_size_);
  GE_CHECK_NOTNULL(task.args_);
  errno_t err = memcpy_s(task.args_.get(), task.args_size_, kernel_def_.args().data(), task.args_size_);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(err != EOK, return INTERNAL_ERROR, "AiCoreTask memcpy failed.");

  const auto *args_offset_tmp = reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data()));
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(context.args_offset().size() / sizeof(uint16_t) < 1, return FAILED,
                                 "context.args_offset().size() / sizeof(uint16_t) less than 1");
  task.offset_ = *args_offset_tmp;
  return SUCCESS;
}

const char *AiCoreKernelRegistry::GetUnique(const string &stub_key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = unique_stubs_.find(stub_key);
  GE_IF_BOOL_EXEC(it != unique_stubs_.end(), return it->c_str());
  it = unique_stubs_.insert(unique_stubs_.end(), stub_key);
  return it->c_str();
}

Status AiCoreTaskBuilder::SetStub(AiCoreOpTask &task) {
  AiCoreKernelRegistry &registry = AiCoreKernelRegistry::GetInstance();
  std::lock_guard<std::mutex> lock(g_reg_mutex);
  const char *unique_key = registry.GetUnique(stub_name_);

  GE_CHK_RT_RET(rtGetFunctionByName(unique_key, &(task.stub_func_)));
  task.stub_name_ = stub_name_;

  return SUCCESS;
}

Status AiCoreTaskBuilder::BuildTask(AiCoreOpTask &task) {
  GE_CHECK_NOTNULL(op_desc_);
  GELOGI("AiCoreTaskBuilder[%s] BuildTask Start.", op_desc_->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(SetKernelArgs(task));
  GE_CHK_STATUS_RET_NOLOG(SetStub(task));
  GELOGI("AiCoreTaskBuilder[%s] BuildTask End.", op_desc_->GetName().c_str());
  return SUCCESS;
}

}  // namespace hybrid
}  // namespace ge
