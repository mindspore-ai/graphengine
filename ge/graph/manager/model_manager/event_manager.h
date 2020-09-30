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

#ifndef GE_GRAPH_MANAGER_MODEL_MANAGER_EVENT_MANAGER_H_
#define GE_GRAPH_MANAGER_MODEL_MANAGER_EVENT_MANAGER_H_


#include <vector>

#include "common/fmk_error_codes.h"
#include "common/fmk_types.h"
#include "common/util.h"
#include "runtime/event.h"

namespace ge {
class EventManager {
 public:
  ///
  /// @ingroup domi_ome
  /// @brief constructor
  ///
  EventManager() : inited_(false), current_idx_(0) {}
  ///
  /// @ingroup domi_ome
  /// @brief destructor
  ///
  ~EventManager() { this->Release(); }

  ///
  /// @ingroup domi_ome
  /// @brief init and create event list
  /// @param [in] event_num event number created
  /// @return exec result
  ///
  Status Init(size_t event_num);

  ///
  /// @ingroup domi_ome
  /// @brief event record
  /// @param [in] event_idx event index
  /// @param [in] stream related stream
  /// @return exec result
  ///
  Status EventRecord(size_t event_idx, rtStream_t stream);

  ///
  /// @ingroup domi_ome
  /// @brief time between start and end in ms
  /// @param [in] start_event_idx start event index
  /// @param [in] stop_event_idx stop event index
  /// @param [out] time
  /// @return exec result
  ///
  Status EventElapsedTime(size_t start_event_idx, size_t stop_event_idx, float &time);

  ///
  /// @ingroup domi_ome
  /// @brief current event index
  /// @return
  ///
  uint32_t CurrentIdx() const { return current_idx_; }

  ///
  /// @ingroup domi_ome
  /// @brief  get event at specific loc
  /// @param [in] index event index
  /// @return
  ///
  Status GetEvent(uint32_t index, rtEvent_t &event);

  ///
  /// @ingroup domi_ome
  /// @brief release event list
  /// @param [in]
  /// @return
  ///
  void Release() noexcept;

 private:
  std::vector<rtEvent_t> event_list_;
  bool inited_;
  uint32_t current_idx_;
};  // EventManager
}  // namespace ge
#endif  // GE_GRAPH_MANAGER_MODEL_MANAGER_EVENT_MANAGER_H_
