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

#ifndef GE_OP_RPN_OPS_H
#define GE_OP_RPN_OPS_H

#include "../graph/operator_reg.h"
namespace ge {
REG_OP(NMSWithMask)
    .INPUT(box_scores, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(selected_boxes, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(selected_idx, TensorType({DT_INT32}))
    .OUTPUT(selected_mask, TensorType({DT_UINT8}))
    .ATTR(iou_threshold, Float, 0.5)
    .OP_END_FACTORY_REG(NMSWithMask)
}  // namespace ge

#endif // GE_OP_TRAINING_OPS_H
