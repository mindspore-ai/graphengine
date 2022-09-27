/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBERS_SCHEDULER_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBERS_SCHEDULER_H_
#include <array>
#include <vector>
#include "built_in_subscriber_definitions.h"
#include "executor_subscriber_guarder.h"
#include "framework/common/ge_visibility.h"
#include "global_profiling.h"
#include "global_dumper.h"
#include "graph/any_value.h"
namespace gert {
class VISIBILITY_EXPORT ExecutorSubscribersScheduler {
 public:
  static void OnExecuteEvent(ExecutorSubscribersScheduler *ins, ExecutorEvent event, const void *node,
                             KernelStatus result);

  ExecutorSubscribersScheduler()
      : enabled_(false),
        built_in_subscribers_ptr_(),
        subscribers_(),
        subscriber_wrapper_({reinterpret_cast<::SubscriberFunc>(ExecutorSubscribersScheduler::OnExecuteEvent), this}) {}
#ifdef ONLY_COMPILE_OPEN_SRC
  ~ExecutorSubscribersScheduler();
#endif
  void Init(const SubscriberExtendInfo &extend_info);
  ExecutorSubscribersScheduler(const ExecutorSubscribersScheduler &) = delete;
  ExecutorSubscribersScheduler &operator=(const ExecutorSubscribersScheduler &) = delete;
  ExecutorSubscriber &GetSubscriber() {
    if (subscribers_.size() == 1UL) {
      return subscribers_[0].GetSubscriber();
    } else {
      return subscriber_wrapper_;
    }
  }

  /**
   * 设置订阅者，订阅者需要实现一个static方法，原型为：
   * ```c++
   * static void OnExecuteEvent(T *void_arg, ExecutorEvent event, const void *node, KernelStatus result);
   * ```
   *
   * 默认情况下，subscribers处于disable状态，在添加首个subscriber时，自动将状态切换到enable状态。
   *
   * @tparam T 订阅者类型
   * @tparam Args 订阅者初始化参数类型
   * @param args 订阅者初始化参数
   * @return 添加的subscriber指针，注意subscriber所有权归`ExecutorSubscribersScheduler`所有，外部使用者不可以释放此指针
   */
  template <typename T, typename... Args>
  T *AddSubscriber(Args... args) {
    auto ins = new (std::nothrow) T(args...);
    if (ins == nullptr) {
      return nullptr;
    }
    // profiler exists when ess init
    if (subscribers_.size() == kInitSubscriberSize) {
      enabled_ = true;
    }
    subscribers_.emplace_back(reinterpret_cast<::SubscriberFunc>(T::OnExecuteEvent), ins, ObjectDeleter<T>);
    return ins;
  }

  /**
   * 添加一个内置的subscriber
   * 内置subscriber较少，当前没有使用注册机制，后续如果需要扩展，那么可以考虑通过注册机制自动注册。
   * 为了易用性，在本类提供了获取内置subscriber的指针的接口。而自注册的subscriber将丢失此能力。
   * @param subscriber_type
   */
  void AddBuiltIn(BuiltInSubscriberType subscriber_type, uint64_t enable_flag, const SubscriberExtendInfo &extend_info);
  void RemoveSubscriber(void *subscriber_ptr) {
    for (auto iter = subscribers_.begin(); iter != subscribers_.end(); ++iter) {
      if (iter->GetSubscriber().arg == subscriber_ptr) {
        subscribers_.erase(iter);
        break;
      }
    }
    for (auto &built_in_subscriber : built_in_subscribers_ptr_) {
      if (built_in_subscriber == subscriber_ptr) {
        built_in_subscriber = nullptr;
      }
    }
    if (subscribers_.size() == kInitSubscriberSize) {
      enabled_ = false;
    }
  }

  template <typename T>
  inline T *MutableBuiltInSubscriber(const BuiltInSubscriberType type) {
    return static_cast<T *>(built_in_subscribers_ptr_[static_cast<size_t>(type)]);
  }

  template <typename T>
  inline const T *GetBuiltInSubscriber(const BuiltInSubscriberType type) {
    return static_cast<T *>(built_in_subscribers_ptr_[static_cast<size_t>(type)]);
  }

  bool IsEnable() const {
    return enabled_ || GlobalProfilingWrapper::GetInstance()->GetEnableFlags() ||
           GlobalDumper::GetInstance()->GetEnableFlags();
  }
  void SetEnable(bool enable_flag) {
    enabled_ = enable_flag;
  }
  void Clear() {
    subscribers_.clear();
    for (auto &built_in_subscriber : built_in_subscribers_ptr_) {
      built_in_subscriber = nullptr;
    }
    enabled_ = false;
  }
  size_t GetSize() const {
    return subscribers_.size();
  }

 private:
  bool enabled_{false};
  std::array<void *, static_cast<size_t>(BuiltInSubscriberType::kNum)> built_in_subscribers_ptr_;
  std::vector<ExecutorSubscriberGuarder> subscribers_;
  ExecutorSubscriber subscriber_wrapper_;
};
}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBERS_SCHEDULER_H_
