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

#ifndef INC_EXTERNAL_GE_GE_DATA_FLOW_API_H
#define INC_EXTERNAL_GE_GE_DATA_FLOW_API_H
#include <memory>
#include "ge_api_error_codes.h"

namespace ge {
enum class DataFlowFlag : uint32_t {
  DATA_FLOW_FLAG_EOS = (1U << 0U),  // data flag end
  DATA_FLOW_FLAG_SEG = (1U << 1U)   // segment flag for discontinuous
};

class DataFlowInfoImpl;
class GE_FUNC_VISIBILITY DataFlowInfo {
 public:
  DataFlowInfo();
  ~DataFlowInfo();

  DataFlowInfo(const DataFlowInfo &context) = delete;
  DataFlowInfo(const DataFlowInfo &&context) = delete;
  DataFlowInfo &operator=(const DataFlowInfo &context)& = delete;
  DataFlowInfo &operator=(const DataFlowInfo &&context)& = delete;

  void SetStartTime(const uint64_t start_time);
  uint64_t GetStartTime() const;

  void SetEndTime(const uint64_t end_time);
  uint64_t GetEndTime() const;

  /**
   * @brief Set the Flow Flags object.
   * @param flow_flags can use operate | to merge multi DataFlowFla to flags.
   */
  void SetFlowFlags(const uint32_t flow_flags);
  /**
   * @brief Get the Flow Flags object.
   * @return uint32_t flow flags, can use operate & with DataFlowFlag to check which bit is set.
   */
  uint32_t GetFlowFlags() const;

  /**
   * @brief set user data, max data size is 64.
   * @param data user data point, input.
   * @param size user data size, need in (0, 64].
   * @param offset user data offset, need in [0, 64), size + offset <= 64.
   * @return success:0, failed:others.
   */
  Status SetUserData(const void *data, size_t size, size_t offset = 0U);

  /**
   * @brief get user data, max data size is 64.
   * @param data user data point, output.
   * @param size user data size, need in (0, 64]. The value must be the same as that of SetUserData.
   * @param offset user data offset, need in [0, 64), size + offset <= 64. The value must be the same as that of
   * SetUserData.
   * @return success:0, failed:others.
   */
  Status GetUserData(void *data, size_t size, size_t offset = 0U) const;

 private:
  std::shared_ptr<DataFlowInfoImpl> impl_;
};
}  // namespace ge
#endif  // INC_EXTERNAL_GE_GE_DATA_FLOW_API_H
