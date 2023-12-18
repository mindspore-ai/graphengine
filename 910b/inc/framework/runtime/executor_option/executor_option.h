/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef AIR_CXX_EXECUTOR_OPTION_H
#define AIR_CXX_EXECUTOR_OPTION_H

namespace gert {
enum class ExecutorType {
  // 顺序优先级执行器，基于优先级完成拓扑排序后，使用基于拓扑排序后的结果执行，
  // 本执行器调度速度最快，但是不支持条件/循环节点
  kSequentialPriority,

  // 基于拓扑的执行器，执行时基于拓扑动态计算ready节点，并做调度执行
  kTopological,

  // 基于拓扑的优先级执行器，在`kTopological`的基础上，将ready节点做优先级排序，总是优先执行优先级高的节点
  kTopologicalPriority,

  // 基于host缓存的执行器，在`kTopologicalPriority`的基础上，支持按照冻结后的图执行，被冻结的节点不再执行
  kHostCache,

  // 基于拓扑的多线程执行器
  kTopologicalMultiThread,

  // 基于tprt的多线程执行器
  kTprt,

  kEnd
};

class VISIBILITY_EXPORT ExecutorOption {
 public:
  ExecutorOption() : executor_type_(ExecutorType::kEnd) {}
  explicit ExecutorOption(ExecutorType executor_type) : executor_type_(executor_type) {}
  ExecutorType GetExecutorType() const {
    return executor_type_;
  }
  virtual ~ExecutorOption() = default;

 private:
  ExecutorType executor_type_;
};
}  // namespace gert

#endif  // AIR_CXX_EXECUTOR_OPTION_H
