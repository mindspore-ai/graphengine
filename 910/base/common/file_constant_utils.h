/**
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
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

#ifndef INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTILS_H
#define INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTILS_H

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

void from_json(const nlohmann::json &j, FileConstantInfo &info);
void from_json(const nlohmann::json &j, OptionInfo &option_info);

/* FileConstant算子的权重文件路径允许通过三种方式保存和获取:
 * 1.通过可选IR属性file_path直接设置或者获取外置权重文件的路径；
 * 2.通过可选IR属性file_id设置外置权重文件的唯一标识，并通过option ge.exec.value_bins设置file id到file path的映射；
 * 3.通过私有属性location获取外置权重路径，该属性存在于两种场景：
 *   ① parser模块解析onnx模型的外置权重时，权重路径会被写在节点的location属性上；
 *   ② 开启权重外置功能（ge.externalWeight）时，生成的外置权重文件的路径会被写在location属性上。
 */
class FileConstantUtils {
 public:
  /// @brief get file id to file path map from option ge.exec.value_bins
  /// @param [out] file_id_to_path_map
  /// @return Status
  static Status GetFileIdToPathMapFromOption(std::map<std::string, std::string> &file_id_to_path_map);

  /// @brief load one weight from file to device memory
  /// @param [in] fileconstant memory addr on device
  /// @param [in] file path
  /// @param [in] offset
  /// @param [in] weight size
  /// @param [in] fileconstant memory size
  /// @return Status
  static Status CopyOneWeightFromFile(const void *const curr_dev_ptr, const std::string &file_path, const size_t offset,
                                      const size_t file_constant_size, size_t &left_size);

  /// @brief get weight file path
  /// @param [in] op_desc
  /// @param [in] file_id_to_path_map
  /// @param [out] file_path
  /// @param [out] offset
  /// @param [out] length
  /// @return Status
  static Status GetFilePath(const OpDescPtr &op_desc, const std::map<std::string, std::string> &file_id_to_path_map,
                            std::string &file_path, size_t &offset, size_t &length);

  /// @brief get weight file path from attr location(private attribute)
  /// @param [in] op_desc
  /// @param [out] file_path
  /// @param [out] offset
  /// @param [out] length
  /// @return void
  static void GetFileConstantPath(const OpDescPtr &op_desc, std::string &file_path, size_t &offset, size_t &length);

  /// @brief get dir name to save external weight from om path
  /// @param [in] om_path
  /// @param [out] file_constant_weight_dir
  /// @return Status
  static Status GetExternalWeightDirFromOmPath(const std::string &om_path, string &file_constant_weight_dir);

  /// @brief set absolute file path for one fileconstant node
  /// @param [in] op_desc
  /// @param [in] weight_dir
  /// @return Status
  static Status SetExternalPath(const OpDescPtr &op_desc, const std::string &weight_dir);

  /// @brief set absolute file path for all fileconstant nodes in graph
  /// @param [in] compute_graph
  /// @param [in] weight_dir
  /// @return Status
  static Status SetExternalPath(const ComputeGraphPtr &compute_graph, const std::string &weight_dir);

  /// @brief replace const and fileconstant with attrs retained
  /// @param [in] new_node (const/fileconstant)
  /// @param [in] old_node (fileconstant/const)
  /// @param [in] compute_graph
  /// @return Status
  static Status ReplaceNodeWithAttrs(const NodePtr &new_node, const NodePtr &old_node,
                                     const ComputeGraphPtr &compute_graph);

  /// @brief get hash value of one weight
  /// @param [in] data addr
  /// @param [in] data_length
  /// @return string
  static std::string GetHashValueOfWeight(const uint8_t *const data, const size_t data_length);

  /// @brief load one weight from file to host memory
  /// @param [in] file_path
  /// @param [in] offset
  /// @param [in] file_length
  /// @param [in] bin_buff
  /// @return Status
  static Status ReadExternalWeightFromFile(const std::string &file_path, const size_t offset, const size_t file_length,
                                           char_t *const bin_buff);

  /// @brief convert all fileconstant nodes to const nodes in graph
  /// @param [in] compute_graph
  /// @return Status
  static Status ConvertFileConstToConst(const ComputeGraphPtr &compute_graph);

  /// @brief convert all const nodes to fileconstant nodes in graph
  /// @param [in] compute_graph
  /// @return Status
  static Status ConvertConstToFileConst(const ComputeGraphPtr &compute_graph);

  /// @brief move weight files from tmp_weight to om_path/weight
  /// @param [in] compute_graph
  /// @param [in] om_path
  /// @return Status
  static Status ChangeFilePath(const ComputeGraphPtr &compute_graph, const std::string &om_path);

  /// @brief move weight files from tmp_weight to om_path/weight
  /// @param [in] flow_model
  /// @param [in] om_path
  /// @return Status
  static Status ChangeFilePath(const FlowModelPtr &flow_model, const std::string &om_path);

 private:
  friend class ExternalWeightManager;
  /// @brief get tmp weight dir
  /// @param [in] pid
  /// @param [in] session_id
  /// @return Status
  static std::string GetTmpWeightDir(const int32_t pid, const uint64_t session_id);

  /// @brief set weight file path to attr location(private attribute)
  /// @param [in] op_desc
  /// @param [in] file_path
  /// @param [in] offset
  /// @param [in] length
  /// @return Status
  static Status SetFileConstantPath(const OpDescPtr &op_desc, const std::string &file_path, const int64_t offset = 0,
                                    const int64_t length = 0);

  /// @brief convert all const nodes to fileconstant nodes with meta in graph
  /// @param [in] op_desc
  /// @param [in] file_path
  /// @param [in] offset
  /// @param [in] length
  /// @return Status
  static Status ConvertConstToFileConstWithMeta(const ComputeGraphPtr &compute_graph,
                                                const ExternalWeightManagerPtr &external_weight_manager,
                                                const std::string &file_const_dir, FileConstantMeta &meta);

  /// @brief change all fileconstant nodes attr location in graph
  /// @param [in] compute_graph
  /// @param [in] om_path
  /// @param [out] old_file_to_new_file
  /// @return Status
  static Status ChangeFilePathAttr(const ComputeGraphPtr &compute_graph, const std::string &om_path,
                                   std::map<std::string, std::string> &old_file_to_new_file);

  /// @brief move weight file from old path to new path
  /// @param [in] old_file_to_new_file
  /// @return Status
  static Status MoveFilePath(const std::map<std::string, std::string> &old_file_to_new_file);

  /// @brief check whether conversion is needed
  /// @param [in] compute_graph
  /// @return true/false
  static bool IsNeedConvert(const ComputeGraphPtr &compute_graph);

  /// @brief get dir_name + file_name
  /// @param [in] dir_name
  /// @param [in] file_name
  /// @param [out] full_name
  /// @return Status
  static void GetValidFullPath(const std::string &dir_name, const std::string &file_name, std::string &full_name);
};
}

#endif // INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTILS_H
