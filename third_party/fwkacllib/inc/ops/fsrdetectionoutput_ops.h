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

#ifndef GE_OP_FSRDETECTIONOUTPUT_OPS_H_
#define GE_OP_FSRDETECTIONOUTPUT_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
REG_OP(FSRDetectionOutput)
    .INPUT(rois, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(prior_box, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(score, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(actual_rois_num, TensorType({DT_INT32}))
    .OUTPUT(actual_bbox_num, TensorType({DT_INT32}))
    .OUTPUT(box, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(batch_rois, Int, 1024)
    .ATTR(im_info, ListInt, {375,1024})
    .ATTR(num_classes, Int, 80)
    .ATTR(max_rois_num, Int, 1024)
    .ATTR(score_thresh, Float, 0.45)
    .ATTR(nms_thresh, Float, 0.7)
    .ATTR(bbox_reg_weights, ListInt, {1,1,1,1})
    .ATTR(post_nms_topn, Int, 304)
    .OP_END_FACTORY_REG(FSRDetectionOutput)
}
#endif
