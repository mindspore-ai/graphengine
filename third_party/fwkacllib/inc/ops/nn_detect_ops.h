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

#ifndef GE_OP_NN_DETECT_OPS_H_
#define GE_OP_NN_DETECT_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

REG_OP(BoundingBoxDecode)
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(deltas, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(means, ListFloat, {0.0, 0.0, 0.0, 0.0})
    .ATTR(stds, ListFloat, {1.0, 1.0, 1.0, 1.0})
    .REQUIRED_ATTR(max_shape, ListInt)
    .ATTR(wh_ratio_clip, Float, 0.016)
    .OP_END_FACTORY_REG(BoundingBoxDecode)

REG_OP(BoundingBoxEncode)
    .INPUT(anchor_box, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(ground_truth_box, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(delats, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(means, ListFloat, {0.0, 0.0, 0.0, 0.0})
    .ATTR(stds, ListFloat, {1.0, 1.0, 1.0, 1.0})
    .OP_END_FACTORY_REG(BoundingBoxEncode)

REG_OP(CheckValid)
    .INPUT(bbox_tensor, TensorType({DT_FLOAT16}))
    .INPUT(img_metas, TensorType({DT_FLOAT16}))
    .OUTPUT(valid_tensor, TensorType({DT_INT8}))
    .OP_END_FACTORY_REG(CheckValid)

REG_OP(Iou)
    .INPUT(bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gtboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(overlap, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(mode, String, "iou")
    .OP_END_FACTORY_REG(Iou)

REG_OP(ROIAlignGrad)
    .INPUT(ydiff, TensorType({DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(rois_n, TensorType({DT_INT32}))
    .OUTPUT(xdiff, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(xdiff_shape, ListInt)
    .REQUIRED_ATTR(pooled_width, Int)
    .REQUIRED_ATTR(pooled_height, Int)
    .REQUIRED_ATTR(spatial_scale, Float)
    .ATTR(sample_num, Int, 2)
    .OP_END_FACTORY_REG(ROIAlignGrad)

REG_OP(ROIAlign)
    .INPUT(features, TensorType({DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(rois_n, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(spatial_scale, Float)
    .REQUIRED_ATTR(pooled_height, Int)
    .REQUIRED_ATTR(pooled_width, Int)
    .ATTR(sample_num, Int, 2)
    .OP_END_FACTORY_REG(ROIAlign)

}  // namespace ge

#endif  // GE_OP_NN_DETECT_OPS_H_
