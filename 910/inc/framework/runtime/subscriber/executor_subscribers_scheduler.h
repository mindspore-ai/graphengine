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
#include "global_profiler.h"
#include "global_dumper.h"
#include "global_tracer.h"
#include "graph/any_value.h"
#include "framework/runtime/exe_graph_executor.h"
#include "common/util/mem_utils.h"

namespace gert {
class VISIBILITY_EXPORT ExecutorSubscribersScheduler {
 public:
  static void OnExecuteEvent(SubExeGraphType sub_exe_graph_type, const ExecutorSubscribersScheduler *ins,
                             ExecutorEvent event, const void *node, KernelStatus result);

  ExecutorSubscribersScheduler()
      : enabled_(false),
        built_in_subscribers_ptr_(),
        sub_exe_graph_subscribers_(),
        subscribers_holder_(),
        subscriber_wrapper_({reinterpret_cast<::SubscriberFunc>(ExecutorSubscribersScheduler::OnExecuteEvent), this}) {}
  void Init(const SubscriberExtendInfo &extend_info);
  ExecutorSubscribersScheduler(const ExecutorSubscribersScheduler &) = delete;
  ExecutorSubscribersScheduler &operator=(const ExecutorSubscribersScheduler &) = delete;
  ExecutorSubscriber &GetSubscriber(SubExeGraphType sub_exe_graph_type);
  const std::vector<ExecutorSubscriberGuarderPtr> &GetWorkingSubscribers() const {
    return working_sub_exe_graph_subscribers_;
  }
  /**
   * 为所有子图类型设置订阅者，订阅者需要实现一个static方法，原型为：
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
    auto ins = AddSubscriberGuarder<T>(args...);
    if (ins == nullptr) {
      return nullptr;
    }
    for (size_t i = 0U; i < kSubExeGraphTypeEnd; ++i) {
      sub_exe_graph_subscribers_[i].emplace_back(subscribers_holder_[subscribers_holder_.size() - 1U]);
    }
    return ins;
  }

  /**
   * 为指定子图类型设置订阅者，订阅者需要实现一个static方法，原型为：
   * ```c++
   * static void OnExecuteEvent(T *void_arg, ExecutorEvent event, const void *node, KernelStatus result);
   * ```
   *
   * 默认情况下，subscribers处于disable状态，在添加首个subscriber时，自动将状态切换到enable状态。
   *
   * @tparam T 订阅者类型
   * @param sub_exe_graph_type 子图类型
   * @tparam Args 订阅者初始化参数类型
   * @param args 订阅者初始化参数
   * @return 添加的subscriber指针，注意subscriber所有权归`ExecutorSubscribersScheduler`所有，外部使用者不可以释放此指针
   */
  template <typename T, typename... Args>
  T *AddSubscriber(SubExeGraphType sub_exe_graph_type, Args... args) {
    auto ins = AddSubscriberGuarder<T>(args...);
    if (ins == nullptr) {
      return nullptr;
    }
    sub_exe_graph_subscribers_[sub_exe_graph_type].emplace_back(subscribers_holder_[subscribers_holder_.size() - 1U]);
    return ins;
  }

  template <typename T, typename... Args>
  T *AddSubscriber(SubExeGraphType sub_exe_graph_type, const std::function<bool()> &enabled_func, Args... args) {
    auto ins = AddSubscriberGuarder<T>(args...);
    if (ins == nullptr) {
      return nullptr;
    }
    auto &subscriber_guarder = subscribers_holder_[subscribers_holder_.size() - 1U];
    subscriber_guarder->SetEnabledFunc(enabled_func);
    sub_exe_graph_subscribers_[sub_exe_graph_type].emplace_back(subscriber_guarder);
    return ins;
  }

  template <typename T, typename... Args>
  T *AddSubscriber(const std::function<bool()> &enabled_func, Args... args) {
    auto ins = AddSubscriberGuarder<T>(args...);
    if (ins == nullptr) {
      return nullptr;
    }
    for (size_t i = 0U; i < kSubExeGraphTypeEnd; ++i) {
      auto &subscriber_guarder = subscribers_holder_[subscribers_holder_.size() - 1U];
      subscriber_guarder->SetEnabledFunc(enabled_func);
      sub_exe_graph_subscribers_[i].emplace_back(subscriber_guarder);
    }
    return ins;
  }
  /**
   * 添加一个内置的subscriber
   * 内置subscriber较少，当前没有使用注册机制，后续如果需要扩展，那么可以考虑通过注册机制自动注册。
   * 为了易用性，在本类提供了获取内置subscriber的指针的接口。而自注册的subscriber将丢失此能力。
   * @param subscriber_type
   */
  template <typename T>
  void AddBuiltIn(BuiltInSubscriberType subscriber_type, uint64_t enable_flag, const SubscriberExtendInfo &extend_info,
                  SubExeGraphType sub_graph_type, const std::function<bool()> &enabled_func) {
    (void)enable_flag;
    if (subscriber_type >= BuiltInSubscriberType::kNum) {
      GELOGW("Unexpected built-in subscriber type %zu", static_cast<size_t>(subscriber_type));
      return;
    }

    auto subscriber_index = static_cast<size_t>(subscriber_type);
    if (built_in_subscribers_ptr_[subscriber_index] != nullptr) {
      GELOGW("The built in subscriber %zu already exists, ignore the add operation", subscriber_index);
      return;
    }

    void *ins;
    if (sub_graph_type == kSubExeGraphTypeEnd) {
      ins = AddSubscriber<T>(enabled_func, extend_info);
    } else {
      ins = AddSubscriber<T>(sub_graph_type, enabled_func, extend_info);
    }
    built_in_subscribers_ptr_[subscriber_index] = ins;
  }
  void RemoveSubscriber(const void *subscriber_ptr) {
    for (auto iter = subscribers_holder_.begin(); iter != subscribers_holder_.end(); ++iter) {
      if ((*iter)->GetSubscriber().arg == subscriber_ptr) {
        RemoveFromSubExeGraphSubscribers(subscriber_ptr);
        subscribers_holder_.erase(iter);
        break;
      }
    }
    for (auto &built_in_subscriber : built_in_subscribers_ptr_) {
      if (built_in_subscriber == subscriber_ptr) {
        built_in_subscriber = nullptr;
      }
    }
    if (subscribers_holder_.size() == static_cast<size_t>(BuiltInSubscriberType::kNum)) {
      enabled_ = false;
    }
  }

  template <typename T>
  inline T *MutableBuiltInSubscriber(const BuiltInSubscriberType type) {
    return static_cast<T *>(built_in_subscribers_ptr_[static_cast<size_t>(type)]);
  }

  template <typename T>
  inline const T *GetBuiltInSubscriber(const BuiltInSubscriberType type) const {
    return static_cast<T *>(built_in_subscribers_ptr_[static_cast<size_t>(type)]);
  }

  bool IsEnable() const {
    return enabled_ || static_cast<bool>(GlobalProfilingWrapper::GetInstance()->GetEnableFlags()) ||
           static_cast<bool>(GlobalDumper::GetInstance()->GetEnableFlags()) ||
           static_cast<bool>(GlobalTracer::GetInstance()->GetEnableFlags());
  }
  void SetEnable(bool enable_flag) {
    enabled_ = enable_flag;
  }
  void Clear() {
    subscribers_holder_.clear();
    for (auto &built_in_subscriber : built_in_subscribers_ptr_) {
      built_in_subscriber = nullptr;
    }
    for (auto &subscribers_vec : sub_exe_graph_subscribers_) {
      subscribers_vec.clear();
    }
    enabled_ = false;
  }
  size_t GetSize() const {
    return subscribers_holder_.size();
  }
 private:
  template <typename T, typename... Args>
  T *AddSubscriberGuarder(Args... args) {
    auto ins = new (std::nothrow) T(args...);
    if (ins == nullptr) {
      return nullptr;
    }
    // profiler exists when ess init
    if (subscribers_holder_.size() == static_cast<size_t>(BuiltInSubscriberType::kNum)) {
      enabled_ = true;
    }
    auto guarder = ge::MakeShared<ExecutorSubscriberGuarder>(reinterpret_cast<::SubscriberFunc>(T::OnExecuteEvent),
                                                         ins, ObjectDeleter<T>);
    if (guarder == nullptr) {
      delete ins;
      return nullptr;
    }
    subscribers_holder_.emplace_back(guarder);
    return ins;
  }
  void RemoveFromSubExeGraphSubscribers(const void *subscriber_ptr) {
    for (auto &subscribers_vec : sub_exe_graph_subscribers_) {
      for (auto iter = subscribers_vec.begin(); iter != subscribers_vec.end(); ++iter) {
        if (subscriber_ptr == (*iter)->GetSubscriber().arg) {
          subscribers_vec.erase(iter);
          return;
        }
      }
    }
  }
 private:
  bool enabled_{false};
  std::array<void *, static_cast<size_t>(BuiltInSubscriberType::kNum)>
      built_in_subscribers_ptr_;
  std::array<std::vector<ExecutorSubscriberGuarderPtr>, kSubExeGraphTypeEnd> sub_exe_graph_subscribers_;
  std::vector<ExecutorSubscriberGuarderPtr> working_sub_exe_graph_subscribers_{};
  std::vector<ExecutorSubscriberGuarderPtr> subscribers_holder_;
  ExecutorSubscriber subscriber_wrapper_;
};
}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBERS_SCHEDULER_H_
