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
/**
*@brief Returns detection result.

*@par Inputs:
* Four inputs, including:
*@li rois: An NCHW tensor of type floa16 or float32, output from operator proposal_d at the preceding layer, used as the input of operator FSRDetectionOutput.
*@li prior_box: An NCHWC0 tensor of type floa16 or float32, specifying the prediction offset, used to update the coordinates [x1, y1, x2, y2] of each ROI.
*@li score: An NCHWC0 tensor of type floa16 or float32, specifying the probability of each class. Class 0 is the background class.
*@li actual_rois_num: An NCHW tensor of type int32, specifying the number of valid boxes per batch.
*@par Attributes:
*@li batch_rois: An optional int32, specifying the number of images to be predicted. Defaults to "1024". The value range is [1, 1024].
*@li im_info: An optional list of two ints. Defaults to (375, 1024). The value range is [1, 1024].
*@li num_classes: An optional int32, specifying the number of classes to be predicted. Defaults to "80". The value must be greater than 0.
*@li max_rois_num: An optional int32, specifying the maximum number of ROIs per batch. Defaults to "1024". The value must be a multiple of 16.
*@li score_thresh: An optional float32, specifying the threshold for box filtering. Defaults to 0.45. The value range is [0.0, 1.0].
*@li nms_thresh: An optional float32, specifying the confidence threshold for box filtering, which is the output "obj" of operator Region. Defaults to 0.7. The value range is (0.0, 1.0).
*@li bbox_reg_weights: An optional list of four ints. Defaults to (1, 1, 1, 1). Must not have value "0".
*@li post_nms_topn: An optional int, specifying the number of output boxes. Defaults to "304". The value must be less than or equal to 1024 and must be a multiple of 16.
*@li kernel_name: An optional string, specifying the operator name. Defaults to "fsr_detection_output".
*@par Outputs:
*box: An NCHW tensor of type float16, describing the information of each output box, including the coordinates, class, and confidence.
*actual_bbox_num: An NCHW tensor of type int32, specifying the number of output boxes.

*@attention Constraints:\n
*@li totalnum < max_rois_num * batch_rois.
*@li "score" must be with shape (total_num, (num_classes+15)//16, 1, 1, 16), where "total_num" indicates the number of valid input boxes of all images.
*@li "prior_box" must be with shape (total_num, (num_classes*4+15)//16, 1, 1, 16), where "total_num" indicates the number of valid input boxes of all images.
*/
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
