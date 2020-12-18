/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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


#ifndef COMMON_GRAPH_UTILS_TRANSFORMER_UTILS_H_
#define COMMON_GRAPH_UTILS_TRANSFORMER_UTILS_H_
#include <string>
#include <map>

#include "external/graph/types.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "transformer/inc/transfer_shape_according_to_format.h"

namespace ge {
class NodeShapeTransUtils {
 public:
  bool CatchFormatAndShape();
  bool UpdateFormatAndShape();

  explicit NodeShapeTransUtils(OpDescPtr op_desc) : op_desc_(op_desc) {
  }

  ~NodeShapeTransUtils() {
  }

 private:
  std::map<std::string, Format> map_format_in_;
  std::map<std::string, Format> map_ori_format_in_;
  std::map<std::string, DataType> map_dtype_in_;
  std::map<std::string, Format> map_format_out_;
  std::map<std::string, Format> map_ori_format_out_;
  std::map<std::string, DataType> map_dtype_out_;

  OpDescPtr op_desc_;
};
}  // namespace ge
#endif  // COMMON_GRAPH_UTILS_TRANSFORMER_UTILS_H_