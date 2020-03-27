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

#ifndef TDT_HOST_INNER_INC_TSD_CLIENT_H_
#define TDT_HOST_INNER_INC_TSD_CLIENT_H_

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include "tdt/status.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

namespace tdt {
constexpr uint32_t RANK_SIZE_DEFAULT_VALUE = 1;

class TsdClient {
 public:
  static TsdClient *GetInstance();

  ~TsdClient();

  TDT_StatusT Open(const uint32_t phyDeviceId, const uint32_t rankSize = RANK_SIZE_DEFAULT_VALUE);

  TDT_StatusT Close();

 private:
  TsdClient();
  TsdClient(const TsdClient &) = delete;
  TsdClient(TsdClient &&) = delete;
  TsdClient &operator=(const TsdClient &) = delete;
  TsdClient &operator=(TsdClient &&) = delete;
  uint32_t rankSize_;
};
}  // namespace tdt
#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  //  TDT_HOST_INNER_INC_TSD_CLIENT_H_
