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
#ifndef GE_COMMON_MODEL_MODEL_DEPLOY_RESOURCE_H_
#define GE_COMMON_MODEL_MODEL_DEPLOY_RESOURCE_H_
#include <string>
#include <vector>
#include <map>

namespace ge {
struct ModelDeployResource {
  std::string resource_type;
  std::string processor_core_num;
  std::string memory;
  std::string share_memory;
  bool is_heavy_load = false;
};

struct HcomCommGroup {
  std::string group_name;
  std::vector<uint32_t> group_rank_list;
};

struct ModelCompileResource {
  std::string host_resource_type;
  std::map<std::string, std::string> logic_dev_id_to_res_type;
};
}  // namespace ge

#endif  // GE_COMMON_MODEL_MODEL_DEPLOY_RESOURCE_H_
