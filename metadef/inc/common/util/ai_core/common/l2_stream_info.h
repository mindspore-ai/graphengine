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

#ifndef L2_STREAM_INFO_H_
#define L2_STREAM_INFO_H_

#include <map>
#include <string>
#include <mutex>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "runtime/base.h"
#include "cce/l2fusion_struct.hpp"

namespace fe {
class StreamL2Info {
 public:
  StreamL2Info(const StreamL2Info &) = delete;
  StreamL2Info &operator=(const StreamL2Info &) = delete;
  static StreamL2Info& Instance();
  Status GetStreamL2Info(rtStream_t stream_id, string node_name, fusion::TaskL2Info_t *&l2_data);
  Status SetStreamL2Info(const rtStream_t &stream_id, fusion::TaskL2InfoFEMap_t &l2_alloc_res);

 private:
  StreamL2Info();
  ~StreamL2Info();
  mutable std::mutex stream_l2_mutex_;
  std::map<rtStream_t, fusion::TaskL2InfoFEMap_t> stream_l2_map_;
};
}  // namespace fe

#endif  // L2_STREAM_INFO_H_