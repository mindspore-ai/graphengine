/**
* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTIL_H
#define INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTIL_H

#include <map>
#include <string>
#include <vector>
#include "ge/ge_api_error_codes.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "framework/pne/flow_model.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "nlohmann/json.hpp"

namespace ge {
struct FileConstantInfo {
    std::string value_bin_file_id;
    std::string value_bin_file_path;
};

struct OptionInfo {
    std::vector<FileConstantInfo> info;
};

void from_json(const nlohmann::json &j, FileConstantMeta &meta);
void to_json(nlohmann::json &j, const FileConstantMeta &meta);
void from_json(const nlohmann::json &j, FileConstantInfo &info);

Status ReplaceNode(const NodePtr &new_node, const NodePtr &old_node, const ComputeGraphPtr &compute_graph);

Status GetFilePathFromOption(std::map<std::string, std::string> &file_id_and_path_map);

Status CopyOneWeightFromFile(const void *const curr_dev_ptr, const std::string &value, const size_t offset,
                             const size_t file_constant_size, size_t &left_size);

Status GetFilePath(const OpDescPtr &op_desc, const std::map<std::string, std::string> &file_id_and_path_map,
                   std::string &file_path, size_t &offset, size_t &length);

void GetFileConstantPath(const OpDescPtr &op_desc, std::string &file_path, size_t &offset, size_t &length);

Status SetFileConstantPath(const OpDescPtr &op_desc, const std::string &file_path, const int64_t offset = 0,
                           const int64_t length = 0);

Status TransferOmPathToExternalWeightDir(const std::string &om_path, string &file_constant_weight_dir);
Status GetExternalWeightDirFromModelData(const ge::ModelData &model_data, std::string &file_constant_weight_dir);

Status SetExternalPath(const OpDescPtr &op_desc, const std::string &weight_dir);

Status SetExternalPath(const ComputeGraphPtr &compute_graph, const std::string &weight_dir);

std::string GetTmpWeightDir(const int32_t pid, const uint64_t session_id);

Status ConvertFileConstToConst(const ComputeGraphPtr &compute_graph);

Status ConvertConstToFileConstWithMeta(const ComputeGraphPtr &compute_graph,
                                       const ExternalWeightManagerPtr &external_weight_manager,
                                       const std::string &file_const_dir, FileConstantMeta &meta);

Status ConvertConstToFileConst(const ComputeGraphPtr &compute_graph);

Status ChangeFilePath(const ComputeGraphPtr &compute_graph, const std::string &om_path);

Status ChangeFilePath(const FlowModelPtr &flow_model, const std::string &om_path);

Status ChangeFilePathAttr(const ComputeGraphPtr &compute_graph, const std::string &om_path,
                          std::map<std::string, std::string> &old_file_to_new_file);

Status MoveFilePath(const std::map<std::string, std::string> &old_file_to_new_file);

Status ReadExternalWeightFromFile(const std::string &file_path,
                                  const size_t offset,
                                  const size_t file_length,
                                  char_t *const bin_buff);
}

#endif // INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTIL_H
