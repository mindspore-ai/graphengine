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

#include <stdint.h>
#include <functional>
#include <vector>
#include "debug/ge_log.h"
#include "debug/ge_util.h"

using namespace std;

namespace ge {

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
BroadCastInfer(const function<vector<int64_t>()>& get_in1_shape, const function<vector<int64_t>()>& get_in2_shape,
               const function<void(const vector<int64_t>& outShape)>& set_out_shape) {
  auto x1_shape = get_in1_shape();
  auto x2_shape = get_in2_shape();
  vector<int64_t> y_shape;

  if (x1_shape.empty()) {
    y_shape = x2_shape;
    set_out_shape(y_shape);
    return GRAPH_SUCCESS;
  }
  if (x2_shape.empty()) {
    y_shape = x1_shape;
    set_out_shape(y_shape);
    return GRAPH_SUCCESS;
  }

  int len_diff = static_cast<int>(x1_shape.size() - x2_shape.size());
  if (len_diff >= 0) {
    for (int i = 0; i < len_diff; i++) {
      y_shape.push_back(x1_shape[i]);
    }
    int x2_shape_size = static_cast<int>(x2_shape.size());
    for (int i = 0; i < x2_shape_size; i++) {
      bool shapeFlag =
        ((x1_shape[i + len_diff] != x2_shape[i]) && (std::min(x1_shape[i + len_diff], x2_shape[i]) != 1));
      if (shapeFlag) {
        GE_LOGE("operands could not be broadcast together");
        return GRAPH_FAILED;
      }
      y_shape.push_back(std::max(x1_shape[i + len_diff], x2_shape[i]));
    }
  } else {
    for (int i = 0; i < -len_diff; i++) {
      y_shape.push_back(x2_shape[i]);
    }
    int x1_shape_size = static_cast<int>(x1_shape.size());
    for (int i = 0; i < x1_shape_size; i++) {
      bool shapeFlag =
        ((x1_shape[i] != x2_shape[i - len_diff]) && (std::min(x1_shape[i], x2_shape[i - len_diff]) != 1));
      if (shapeFlag) {
        GE_LOGE("operands could not be broadcast together");
        return GRAPH_FAILED;
      }
      y_shape.push_back(std::max(x1_shape[i], x2_shape[i - len_diff]));
    }
  }
  set_out_shape(y_shape);
  return GRAPH_SUCCESS;
}

}  // namespace ge
