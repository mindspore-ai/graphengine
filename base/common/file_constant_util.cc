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

#include "common/file_constant_util.h"
#include <fstream>
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "common/helper/file_saver.h"
#include "common/plugin/ge_util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/file_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "runtime/mem.h"
#include "nlohmann/json.hpp"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace {
const int64_t kBlockSize = 10485760;
const std::string kBinFileValues = "value_bins";
const std::string kBinIdValue = "value_bin_id";
const std::string kBinFilePathValue = "value_bin_file";
const std::int32_t kFirstElementIndex = 0;
const std::int32_t kIndentWidth = 2;
const char_t *kTmpWeightDir = "tmp_weight/";

Status ReplaceNode(const NodePtr &new_node, const NodePtr &old_node, const ComputeGraphPtr &compute_graph) {
  GE_CHK_STATUS_RET(GraphUtils::ReplaceNodeAnchors(new_node, old_node, {}, {0}),
                    "Replace node:%s's anchor by node:%s failed",
                    old_node->GetName().c_str(), new_node->GetName().c_str());
  NodeUtils::UnlinkAll(*old_node);
  GE_CHK_STATUS_RET(GraphUtils::RemoveJustNode(compute_graph, old_node), "Remove node:%s in graph:%s failed",
                    old_node->GetName().c_str(), compute_graph->GetName().c_str());
  return SUCCESS;
}
}

void from_json(const nlohmann::json &j, FileConstantInfo &info) {
  const auto id = j.find(kBinIdValue);
  if (id != j.end()) {
    info.value_bin_file_id = id->get<std::string>();
  }

  const auto file_path = j.find(kBinFilePathValue);
  if (file_path != j.end()) {
    info.value_bin_file_path = file_path->get<std::string>();
  }
}

void from_json(const nlohmann::json &j, OptionInfo &option_info) {
  const auto it = j.find(kBinFileValues);
  if (it != j.end()) {
    option_info = it->get<OptionInfo>();
  }
}

Status GetFilePathFromOption(std::map<std::string, std::string> &file_id_and_path_map) {
  std::string opt;
  (void)GetContext().GetOption(FILE_CONSTANT_PATH, opt);
  if (opt.empty()) {
    GELOGW("[Check][Param] Failed to get file constant path.");
    return SUCCESS;
  }
  GELOGI("source string = %s.", opt.c_str());

  nlohmann::json options;
  try {
    options = nlohmann::json::parse(opt);
  } catch (nlohmann::json::exception &ex) {
    REPORT_CALL_ERROR("E19999", "Failed to parse option FILE_CONSTANT_PATH, which [%s] is invalid", opt.c_str());
    GELOGE(GRAPH_FAILED, "Failed to parse option FILE_CONSTANT_PATH, which [%s] is invalid", opt.c_str());
    return GRAPH_FAILED;
  }

  for (const nlohmann::json &single_json : options) {
    GELOGD("Parsing op[%d], jsonStr = %s.", kFirstElementIndex, single_json.dump(kIndentWidth).c_str());
    std::vector<FileConstantInfo> multi_info;
    multi_info = single_json.get<std::vector<FileConstantInfo>>();
    for (const auto &single_info : multi_info) {
      GELOGD("get single info, file id is %s, file path is %s.", single_info.value_bin_file_id.c_str(),
             single_info.value_bin_file_path.c_str());
      (void)file_id_and_path_map.insert(
          std::pair<std::string, std::string>(single_info.value_bin_file_id, single_info.value_bin_file_path));
    }
  }
  return SUCCESS;
}

Status CopyOneWeightFromFile(const void *const curr_dev_ptr, const std::string &value, const size_t offset,
                             const size_t file_constant_size, size_t &left_size) {
  if (left_size < file_constant_size) {
    GELOGE(GRAPH_FAILED, "Failed to copy data to device, free memory is %zu, need copy size = %ld.", left_size,
           file_constant_size);
    return GRAPH_FAILED;
  }
  const std::string real_path = RealPath(value.c_str());
  std::ifstream ifs(real_path, std::ifstream::binary);
  if (!ifs.is_open()) {
    GELOGE(GRAPH_FAILED, "[Open][File] %s failed.", real_path.c_str());
    REPORT_CALL_ERROR("E19999", "open file:%s failed.", real_path.c_str());
    return GRAPH_FAILED;
  }
  ifs.clear();
  ifs.seekg(offset, ifs.beg);
  size_t used_memory = 0U;
  std::string compress_nodes;
  compress_nodes.reserve(static_cast<size_t>(kBlockSize));
  Status ret = SUCCESS;
  while ((!ifs.eof()) && (used_memory != file_constant_size)) {
    (void) ifs.read(&compress_nodes[0U], kBlockSize);
    size_t copy_len_once = static_cast<size_t>(ifs.gcount());
    if ((file_constant_size - used_memory) < copy_len_once) {
      copy_len_once = file_constant_size - used_memory;
    }
    if (left_size < (used_memory + copy_len_once)) {
      GELOGE(GRAPH_FAILED, "copy failed for lack memory, free size is %zu, need memroy is %zu.", left_size,
             used_memory + copy_len_once);
      REPORT_CALL_ERROR("E19999", "copy failed for lack memory, free size is %zu, need memroy is %zu.", left_size,
                        used_memory + copy_len_once);
      ret = FAILED;
      break;
    }

    GELOGI("copy %zu bytes to memory.", copy_len_once);
    void *const cur_dev_ptr = reinterpret_cast<void *>(PtrToValue(curr_dev_ptr) + used_memory);
    const rtError_t rts_error =
        rtMemcpy(cur_dev_ptr, left_size - used_memory, &compress_nodes[0U], copy_len_once, RT_MEMCPY_HOST_TO_DEVICE);
    if (rts_error != RT_ERROR_NONE) {
      GELOGE(GRAPH_FAILED, "copy failed, result code = %d.", rts_error);
      REPORT_CALL_ERROR("E19999", "copy failed, result code = %d.", rts_error);
      ret = RT_ERROR_TO_GE_STATUS(rts_error);
      break;
    }
    used_memory += copy_len_once;
  }
  ifs.close();
  left_size -= used_memory;
  GELOGI("used memory is %zu.", used_memory);
  return ret;
}

Status GetFilePath(const OpDescPtr &op_desc, const std::map<std::string, std::string> &file_id_and_path_map,
                   std::string &file_path, size_t &offset, size_t &length) {
  std::string file_path_attr;
  GetFileConstantPath(op_desc, file_path_attr, offset, length);
  if (!file_path_attr.empty()) {
    file_path = std::move(file_path_attr);
    return SUCCESS;
  }
  offset = 0U;
  length = 0U;
  (void)AttrUtils::GetStr(op_desc, ATTR_NAME_FILE_PATH, file_path_attr);
  if (!file_path_attr.empty()) {
    file_path = file_path_attr;
    return SUCCESS;
  }
  std::string file_id;
  file_path = "";
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, file_id), FAILED,
                         "Failed to get filed id from attr");
  GE_CHK_BOOL_RET_STATUS(!file_id.empty(), FAILED, "The file path and file id are empty.");
  const auto it = file_id_and_path_map.find(file_id);
  if (it == file_id_and_path_map.end()) {
    GELOGW("Failed to get file path of file id:%s", file_id.c_str());
    return SUCCESS;
  }
  GE_CHK_BOOL_RET_STATUS(!(it->second.empty()), FAILED, "File path is empty.");
  file_path = it->second;
  return SUCCESS;
}

void GetFileConstantPath(const OpDescPtr &op_desc, std::string &file_path, size_t &offset, size_t &length) {
  (void)AttrUtils::GetStr(op_desc, ATTR_NAME_LOCATION, file_path);
  offset = 0U;
  length = 0U;
  int64_t attr_value = 0;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_OFFSET, attr_value);
  if (attr_value != 0) {
    offset = static_cast<size_t>(attr_value);
  }
  int64_t attr_length = 0;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_LENGTH, attr_length);
  if (attr_length != 0) {
    length = static_cast<size_t>(attr_length);
  }
}

Status SetFileConstantPath(const OpDescPtr &op_desc, const std::string &file_path, const int64_t offset,
                           const int64_t length) {
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, offset), FAILED,
                         "SetInt attribute of %s failed.", ATTR_NAME_OFFSET.c_str());
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, length), FAILED,
                         "SetInt attribute of %s failed.", ATTR_NAME_LENGTH.c_str());
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, file_path), FAILED,
                         "SetStr attribute of %s failed.", ATTR_NAME_LOCATION.c_str());
  return SUCCESS;
}

Status SetExternalPath(const OpDescPtr &op_desc, const std::string &om_path) {
  std::string path = om_path;
  const char *const om_dir = mmDirName(&path[0]);
  GE_CHECK_NOTNULL(om_dir);
  std::string weight_dir = std::string(om_dir) + "/weight";
  std::string file_name;
  size_t offset = 0U;
  size_t length = 0U;
  GetFileConstantPath(op_desc, file_name, offset, length);
  if (file_name.empty()) {
    return SUCCESS;
  }
  if (file_name.rfind('/') != std::string::npos) {
    return SUCCESS;
  }
  std::string file_path = weight_dir + "/" + file_name;
  GE_CHK_STATUS_RET(SetFileConstantPath(op_desc, file_path, static_cast<int64_t>(offset), static_cast<int64_t>(length)),
                    "Failed to set file constant path to op:%s", op_desc->GetName().c_str());
  GELOGD("Set external path success, file path:%s", file_path.c_str());
  return SUCCESS;
}

Status ConvertFileConstToConst(const ComputeGraphPtr &compute_graph) {
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() == FILECONSTANT) {
      auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      auto output_desc = op_desc->MutableOutputDesc(0U);
      GE_CHECK_NOTNULL(output_desc);
      DataType out_type = ge::DT_UNDEFINED;
      (void)AttrUtils::GetDataType(op_desc, "dtype", out_type);
      output_desc->SetDataType(out_type);

      int64_t weight_size = 0;
      GE_CHK_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*output_desc, weight_size),
                        "Failed to get file constant weight size.");
      std::string file_path;
      size_t offset = 0U;
      size_t length = 0U;
      GetFileConstantPath(op_desc, file_path, offset, length);
      if (file_path.empty()) {
        (void)AttrUtils::GetStr(op_desc, ATTR_NAME_FILE_PATH, file_path);
        if (file_path.empty()) {
          continue;
        }
      }
      const size_t file_length = (length == 0U ? static_cast<size_t>(weight_size) : length);
      const std::string real_path = RealPath(file_path.c_str());
      GE_CHK_BOOL_RET_STATUS(!real_path.empty(), FAILED, "Failed to get real path of %s", file_path.c_str());
      std::ifstream ifs(real_path, std::ifstream::binary);
      GE_CHK_BOOL_RET_STATUS(ifs.is_open(), FAILED, "Read file %s failed.", real_path.c_str());
      ifs.clear();
      ifs.seekg(offset, ifs.beg);
      const auto bin_buff = MakeUnique<char[]>(file_length);
      (void)ifs.read(static_cast<char *>(bin_buff.get()), static_cast<int64_t>(file_length));
      ifs.close();

      const GeTensorPtr &const_value = MakeShared<GeTensor>(op_desc->GetOutputDesc(0U),
                                                     reinterpret_cast<uint8_t *>(bin_buff.get()), file_length);
      GE_CHECK_NOTNULL(const_value);
      auto const_op = OpDescUtils::CreateConstOp(const_value);
      GE_CHECK_NOTNULL(const_op);
      const_op->SetName(op_desc->GetName() + "_" + CONSTANT);
      const auto &own_graph = node->GetOwnerComputeGraph();
      NodePtr const_node = own_graph->AddNode(const_op);
      GE_CHK_STATUS_RET(ReplaceNode(const_node, node, own_graph),
                        "Convert node:%s from file constant to const failed.", node->GetName().c_str());
      GELOGD("Convert node:%s from file constant to const success.", node->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status ConvertConstToFileConst(const ComputeGraphPtr &compute_graph) {
  std::string time_stamp = CurrentTimeInStr();
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (NodeUtils::IsConst(*node)) {
      const auto &weight = OpDescUtils::MutableWeights(node);
      if (weight.empty()) {
        GELOGW("Node:%s has empty tensor, skip conversion.", node->GetName().c_str());
        continue;
      }
      const auto &tensor = weight[0];
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);

      const std::string file_constant_name = op_desc->GetName() + "_" + FILECONSTANT;
      const auto &file_constant_op = MakeShared<OpDesc>(file_constant_name, FILECONSTANT);
      GE_CHECK_NOTNULL(file_constant_op);
      const auto &output_desc = op_desc->GetOutputDesc(0U);
      (void)file_constant_op->AddOutputDesc("y", output_desc);
      std::string file_path = kTmpWeightDir + compute_graph->GetName() + "_" + time_stamp + "/" + file_constant_name;
      GE_CHK_STATUS_RET(FileSaver::SaveToFile(file_path, tensor->GetData().GetData(), tensor->GetData().GetSize()),
                        "Failed to save the weight of node:%s to file:%s.", node->GetName().c_str(), file_path.c_str());

      int64_t offset = 0U;
      int64_t length = static_cast<int64_t>(tensor->GetData().GetSize());
      GE_CHK_STATUS_RET(SetFileConstantPath(file_constant_op, file_path, offset, length),
                        "Failed to set file constant path to op:%s", file_constant_op->GetName().c_str());
      GE_CHK_BOOL_RET_STATUS(AttrUtils::SetDataType(file_constant_op, "dtype", output_desc.GetDataType()), FAILED,
                             "Failed to set data type to op:%s", file_constant_op->GetName().c_str());
      GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListInt(file_constant_op, "shape", output_desc.GetShape().GetDims()), FAILED,
                             "Failed to set shape to op:%s", file_constant_op->GetName().c_str());

      const auto &own_graph = node->GetOwnerComputeGraph();
      NodePtr file_constant_node = own_graph->AddNode(file_constant_op);
      GE_CHK_STATUS_RET(ReplaceNode(file_constant_node, node, own_graph),
                        "Convert node:%s from const to file constant failed.", node->GetName().c_str());
      GELOGD("Convert node:%s from const to file constant success.", node->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status UnloadFileConstantWeights(const ComputeGraphPtr &compute_graph) {
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() == FILECONSTANT) {
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      std::string file_path;
      size_t offset = 0U;
      size_t length = 0U;
      GetFileConstantPath(op_desc, file_path, offset, length);
      if (file_path.empty()) {
        continue;
      }
      std::string real_path = RealPath(file_path.c_str());
      // Only unload weight files in directory "./tmp_weight/"
      auto pos = real_path.find(kTmpWeightDir);
      if (pos == std::string::npos) {
        continue;
      }
      // Find directory "./tmp_weight/graph_name/"
      pos = real_path.find('/', pos);
      pos = real_path.find('/', pos + 1);
      GE_CHK_BOOL_RET_STATUS(pos != std::string::npos, FAILED, "File path:%s is invalid.", real_path.c_str());
      std::string file_dir = real_path.substr(0, pos);
      GE_CHK_BOOL_RET_STATUS(mmRmdir(file_dir.c_str()) == 0, FAILED, "Failed to remove dir:%s.", file_dir.c_str());
      break;
    }
  }
  // Remove directory "./tmp_weight/" when it is empty
  (void)rmdir(kTmpWeightDir);
  GELOGD("Unload file constant weights success, graph name:%s.", compute_graph->GetName().c_str());
  return SUCCESS;
}

Status ChangeFilePath(const ComputeGraphPtr &compute_graph, const std::string &om_path) {
  std::string origin_dir;
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() == FILECONSTANT) {
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      std::string file_path;
      size_t offset = 0U;
      size_t length = 0U;
      GetFileConstantPath(op_desc, file_path, offset, length);
      if (file_path.empty()) {
        continue;
      }
      std::string real_path = RealPath(file_path.c_str());
      GE_CHK_BOOL_RET_STATUS(!real_path.empty(), FAILED, "Failed to get real path of %s", file_path.c_str());
      auto pos = real_path.find(kTmpWeightDir);
      if (pos == std::string::npos) {
        continue;
      }
      if (origin_dir.empty()) {
        pos = real_path.find('/', pos);
        pos = real_path.find('/', pos + 1);
        GE_CHK_BOOL_RET_STATUS(pos != std::string::npos, FAILED, "File path:%s is invalid.", real_path.c_str());
        origin_dir = real_path.substr(0, pos);
      }

      std::string file_name = StringUtils::GetFileName(real_path);
      GE_CHK_BOOL_RET_STATUS(!file_name.empty(), FAILED, "The file name is empty.");
      std::string path = om_path;
      const char *const om_dir = mmDirName(&path[0]);
      GE_CHECK_NOTNULL(om_dir);
      std::string om_weight_path = std::string(om_dir) + "/weight/";
      GE_CHK_BOOL_RET_STATUS(CreateDirectory(om_weight_path) == 0, FAILED,
                             "Failed to create directory:%s.", om_weight_path.c_str());
      om_weight_path += file_name;
      GE_CHK_BOOL_RET_STATUS(std::rename(real_path.c_str(), om_weight_path.c_str()) == 0, FAILED,
                             "Failed to change path from %s to %s.", real_path.c_str(), om_weight_path.c_str());
      GE_CHK_STATUS_RET(SetFileConstantPath(op_desc, file_name,
                                            static_cast<int64_t>(offset), static_cast<int64_t>(length)),
                        "Failed to set file constant path to op:%s", op_desc->GetName().c_str());
      GELOGD("Node:%s changes file path to %s success.", node->GetName().c_str(), om_weight_path.c_str());
    }
  }
  if (!origin_dir.empty()) {
    GE_CHK_BOOL_RET_STATUS(mmRmdir(origin_dir.c_str()) == 0, FAILED, "Failed to remove dir:%s.", origin_dir.c_str());
    (void)rmdir(kTmpWeightDir);
  }
  return SUCCESS;
}
}  // namespace ge
