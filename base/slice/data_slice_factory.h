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
#ifndef SLICE_DATASLICE_DATA_SLICE_FACTORY_H_
#define SLICE_DATASLICE_DATA_SLICE_FACTORY_H_

#include <map>
#include <functional>
#include "graph/axis_type_info.h"
#include "external/ge/ge_api_types.h"
#include "slice/data_slice_infer_base.h"
namespace ge {
// automatic register factory
class DataSliceFactory {
 public:
  static DataSliceFactory *GetInstance() {
    static DataSliceFactory factory;
    return &factory;
  }
  std::shared_ptr<DataSliceInferBase> GetClassByAxisType(ge::AxisType axis_type);
  void RegistClass(ge::AxisType axis_type,
                     std::function<DataSliceInferBase*(void)> infer_util_class);
  ~DataSliceFactory() {}
 private:
  DataSliceFactory() {}
  std::map<ge::AxisType, std::function<DataSliceInferBase*(void)>> class_map_;
};

class AxisInferRegister {
 public:
  AxisInferRegister(ge::AxisType axis_type, std::function<DataSliceInferBase*(void)> data_slice_infer_impl) {
    DataSliceFactory::GetInstance()->RegistClass(axis_type, data_slice_infer_impl);
  }
  ~AxisInferRegister() = default;
};
}

#endif // SLICE_DATASLICE_DATA_SLICE_FACTORY_H_
