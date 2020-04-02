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

#ifndef GE_OP_DVPP_OPS_H_
#define GE_OP_DVPP_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(DvppCreateChannel)
    .OUTPUT(dvpp_channel, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(DvppCreateChannel)

REG_OP(DvppDestroyChannel)
    .INPUT(dvpp_channel, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(DvppDestroyChannel)

REG_OP(DvppResize)
    .INPUT(dvpp_channel, TensorType({DT_INT64}))
    .INPUT(input_desc, TensorType({DT_UINT8}))
    .INPUT(output_desc, TensorType({DT_UINT8}))
    .INPUT(resize_config, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(DvppResize)

REG_OP(DvppCrop)
    .INPUT(dvpp_channel, TensorType({DT_INT64}))
    .INPUT(input_desc, TensorType({DT_UINT8}))
    .INPUT(output_desc, TensorType({DT_UINT8}))
    .INPUT(crop_area, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(DvppCrop)

REG_OP(DvppCropAndPaste)
    .INPUT(dvpp_channel, TensorType({DT_INT64}))
    .INPUT(input_desc, TensorType({DT_UINT8}))
    .INPUT(output_desc, TensorType({DT_UINT8}))
    .INPUT(crop_area, TensorType({DT_UINT8}))
    .INPUT(paste_area, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(DvppCropAndPaste)

REG_OP(DvppDecodeJpeg)
    .INPUT(dvpp_channel, TensorType({DT_INT64}))
    .INPUT(input_desc, TensorType({DT_UINT8}))
    .INPUT(output_desc, TensorType({DT_UINT8}))
    .INPUT(decode_area, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(DvppDecodeJpeg)
}  // namespace ge

#endif  // GE_OP_DVPP_OPS_H_
