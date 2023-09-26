/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef GE_COMMON_MODEL_MODEL_COMPRESS_MANAGER_H_
#define GE_COMMON_MODEL_MODEL_COMPRESS_MANAGER_H_

#include "common/model/ge_model.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "graph/any_value.h"

namespace ge {
class ModelCompressManager {
 public:
  static Status Compress(const GeModelPtr &ge_model);
  static Status Decompress(const GeModelPtr &ge_model);
  static Status CpyModelAttrs2Dst(const GeModelPtr &src_ge_model, const GeModelPtr &dst_ge_model);
  static void DeleteModelAttrs(const GeModelPtr &ge_model);

 private:
  static Status ProcessAttrsForOp(const OpDescPtr &op_desc, const bool is_compress);
  static Status ProcessKernelNameAttrsForOp(const OpDescPtr &op_desc);
  static Status ProcessKernelName(const OpDescPtr &op_desc);
  static Status ProcessAtomicKernelName(const OpDescPtr &op_desc);
  static Status ProcessAttrsForTensor(const OpDescPtr &op_desc, const bool is_compress);
  static Status ProcessForTensor(const GeTensorDescPtr &tensor, const bool is_compress);
  static Status AddEnumAttrsToModel(const GeModelPtr &ge_model);
  static Status GetEnumAttrsFromModel(const GeModelPtr &ge_model);
  static bool DeleteUnusedAttrs(const OpDescPtr &op_desc, const string &attr_name);
  static Status EnumAttrs(const pair<const string, GeAttrValue> &name_to_value, string &enum_attr_name,
                          GeAttrValue &enum_attr_value);
  static Status DenumAttrs(const pair<const string, GeAttrValue> &name_to_value, string &attr_name,
                           GeAttrValue &attr_value);
  static void UpdateStatus(const bool is_new_attr, const bool is_string_type);
  static bool CheckNeedCompress(const GeModelPtr &ge_model);
  static void PrintOpInfo(const OpDescPtr &op_desc);
  static void CacheClear();

 private:
  static int64_t om_compress_version_;
  static vector<string> enum_attr_names_;
  static vector<string> enum_attr_values_;
  static vector<bool> name_use_string_values_;
  static std::mutex mutex_;
};
}  // namespace ge
#endif  // GE_COMMON_MODEL_MODEL_COMPRESS_MANAGER_H_
