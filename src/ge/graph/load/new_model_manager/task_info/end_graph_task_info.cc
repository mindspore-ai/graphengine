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

#include "graph/load/new_model_manager/task_info/end_graph_task_info.h"

#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/load/new_model_manager/davinci_model.h"

namespace {
const uint32_t kDumpFlag = 2;
}  // namespace
namespace ge {
Status EndGraphTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("InitEndGraphTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }
  davinci_model_ = davinci_model;
  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    GELOGE(ret, "SetStream fail, stream_id:%u", task_def.stream_id());
    return FAILED;
  }

  model_ = davinci_model->GetRtModelHandle();
  GELOGI("InitEndGraphTaskInfo Init Success, model:%p, stream:%p", model_, stream_);
  return SUCCESS;
}

Status EndGraphTaskInfo::Distribute() {
  GELOGI("EndGraphTaskInfo Distribute Start.");
  auto all_dump_model = PropertiesManager::Instance().GetAllDumpModel();
  if (all_dump_model.find(ge::DUMP_ALL_MODEL) != all_dump_model.end() ||
      all_dump_model.find(davinci_model_->Name()) != all_dump_model.end()) {
    GELOGI("Start to call rtEndGraphEx");
    rtError_t rt_ret = rtEndGraphEx(model_, stream_, kDumpFlag);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rtEndGraphEx failed, ret: 0x%x", rt_ret);
      return RT_FAILED;
    }
  } else {
    GELOGI("Start to call rtEndGraph");
    rtError_t rt_ret = rtEndGraph(model_, stream_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rtEndGraph failed, ret: 0x%x", rt_ret);
      return RT_FAILED;
    }
  }

  uint32_t task_id = 0;
  GE_CHECK_NOTNULL(davinci_model_);
  rtError_t rt_ret = rtModelGetTaskId(davinci_model_->GetRtModelHandle(), &task_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  task_id_ = task_id;

  GELOGI("EndGraphTaskInfo Distribute Success, task id is %u", task_id);
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_MODEL_END_GRAPH, EndGraphTaskInfo);

}  // namespace ge
