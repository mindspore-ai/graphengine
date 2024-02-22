/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GE_COMMON_ERROR_TRACKING_H_
#define GE_COMMON_ERROR_TRACKING_H_

#include <mutex>
#include "graph/op_desc.h"
#include "runtime/base.h"

namespace ge {
class  TaskKey {
public:
  TaskKey(uint32_t task_id, uint32_t stream_id, uint32_t  context_id, uint32_t  thread_id) :
    task_id_(task_id), stream_id_(stream_id), context_id_(context_id), thread_id_(thread_id) {
  }

  TaskKey(uint32_t task_id, uint32_t stream_id) : task_id_(task_id), stream_id_(stream_id) {
  }
  bool operator <(const TaskKey &other) const {
      if (this->task_id_ < other.GetTaskId()) {
          return true;
      } else if (this->task_id_ == other.GetTaskId()) {
          if (this->stream_id_ < other.GetStreamId()) {
              return true;
          } else if (this->stream_id_ == other.GetStreamId()) {
              if (this->thread_id_ < other.GetThreadId()) {
                  return true;
              } else if (this->thread_id_ == other.GetThreadId()) {
                  return this->context_id_ < other.GetContextId();
              }
          }
      }

      return false;
  }
  uint32_t GetTaskId() const{
    return task_id_;
  }
  uint32_t GetStreamId() const{
    return stream_id_;
  }
  uint32_t GetThreadId() const{
    return thread_id_;
  }
  uint32_t GetContextId() const{
    return context_id_;
  }
private:
    uint32_t  task_id_;
    uint32_t  stream_id_;
    uint32_t  context_id_{UINT32_MAX};
    uint32_t  thread_id_{UINT32_MAX};
};

class ErrorTracking {
public:
  ErrorTracking(const ErrorTracking &) = delete;
  ErrorTracking(ErrorTracking &&) = delete;
  ErrorTracking &operator=(const ErrorTracking &) = delete;
  ErrorTracking &operator=(ErrorTracking &&) = delete;

  static ErrorTracking &GetInstance() {
    static ErrorTracking instance;
    return instance;
  }

  void SaveGraphTaskOpdescInfo(const OpDescPtr &op, const uint32_t task_id, const uint32_t stream_id);
  void SaveGraphTaskOpdescInfo(const OpDescPtr &op, const TaskKey &key);
  void SaveSingleOpTaskOpdescInfo(const OpDescPtr &op, const uint32_t task_id, const uint32_t stream_id);

  void GetGraphTaskOpdescInfo(const uint32_t task_id, const uint32_t stream_id, OpDescPtr &op) {
    TaskKey key(task_id, stream_id);
    GetTaskOpdescInfo(op, key, graph_task_to_opdesc_);
  }

  void GetGraphTaskOpdescInfo(TaskKey key, OpDescPtr &op) {
    GetTaskOpdescInfo(op, key, graph_task_to_opdesc_);
  }

  void GetSingleOpTaskOpdescInfo(const uint32_t task_id, const uint32_t stream_id, OpDescPtr &op) {
    TaskKey key(task_id, stream_id);
    GetTaskOpdescInfo(op, key, single_op_task_to_opdesc_);
  }

  void Finalize() {
    graph_task_to_opdesc_.clear();
    single_op_task_to_opdesc_.clear();
  }

private:
ErrorTracking();
void AddTaskOpdescInfo(const OpDescPtr &op, const TaskKey &key, std::map<TaskKey, OpDescPtr> &map, uint32_t max_count);
void GetTaskOpdescInfo(OpDescPtr &op, const TaskKey &key, std::map<TaskKey, OpDescPtr> &map);
std::mutex mutex_;
uint32_t single_op_max_count_{4096U};
std::map<TaskKey, OpDescPtr> graph_task_to_opdesc_;
std::map<TaskKey, OpDescPtr> single_op_task_to_opdesc_;
};

  void ErrorTrackingCallback(rtExceptionInfo *const exception_data);

  uint32_t RegErrorTrackingCallBack();
}
#endif