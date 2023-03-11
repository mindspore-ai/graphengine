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
#include "slice/data_slice_default_impl.h"
#include "slice/data_slice_toolkit.h"
#include "slice/data_slice_factory.h"
#include "framework/common/debug/ge_log.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/type_utils.h"

namespace ge {
static AxisInferRegister registerDefault(ge::AxisType::UNSPLIT,
  [] (void) noexcept ->DataSliceInferBase* {return new (std::nothrow) DataSliceDefaultImpl();});

// Default
Status DataSliceDefaultImpl::InferAxisSlice(ge::Operator &op, const AxisTypeInfo &slice_info,
    const DataSliceType &out_data_slice, DataSliceType &in_data_slice)
{
  GELOGI("Default infer func, op:%s, type:%s, axis type:%d",
         DataSliceGetName(op).c_str(), DataSliceGetOpType(op).c_str(), static_cast<int8_t>(slice_info.GetAxisType()));
  in_data_slice = out_data_slice;
  return SUCCESS;
}
}
