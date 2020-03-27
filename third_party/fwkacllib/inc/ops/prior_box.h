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

 #ifndef GE_OP_PRIORBOX_H
 #define GE_OP_PRIORBOX_H

 #include "graph/operator_reg.h"

 namespace ge {
/**
*@brief Performs SSD prior box detection.

*@par Inputs:
* Two inputs, including:
*@li feature: An NC1HWC0 or NCHW feature map of type is float32 or float16.
*@li img: source image. Has the same type and format as "feature".

*@par Attributes:
*@li min_size: A required float32, specifying the minimum edge length of a square prior box.
*@li max_size: A required float32, specifying the maximum edge length of a square prior box: sqrt(min_size * max_size)
*@li aspect_ratio: An optional float32, specifying the aspect ratio for generated rectangle boxes. The height is min_size/sqrt(aspect_ratio), the width is min_size*sqrt(aspect_ratio). Defaults to "1.0".
*@li img_size: An optional int32, specifying the source image size. Defaults to "0".
*@li img_h: An optional int32, specifying the source image height. Defaults to "0".
*@li img_w: An optional int32, specifying the source image width. Defaults to "0".
*@li step: An optional float32, specifying the step for mapping the center point from the feature map to the source image. Defaults to "0.0".
*@li step_h: An optional float32, specifying the height step for mapping the center point from the feature map to the source image. Defaults to "0.0".
*@li step_w: An optional float32, specifying the width step for mapping the center point from the feature map to the source image. Defaults to "0.0".
*@li flip: An optional bool. If "True", "aspect_ratio" will be flipped. Defaults to "True".
*@li clip: An optional bool. If "True", a prior box is clipped to within [0, 1]. Defaults to "False".
*@li offset: An optional float32, specifying the offset. Defaults to "0.5".
*@li variance: An optional float32, specifying the variance of a prior box, either one or four variances. Defaults to "0.1" (one value).

*@par Outputs:
*y: An ND tensor of type float32 or float16, specifying the prior box information, including its coordinates and variance.

*@attention Constraints:\n
* This operator applies only to SSD networks.
*@see SSDDetectionOutput()
*/
 REG_OP(PriorBox)
     .INPUT(feature, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(img, TensorType({DT_FLOAT16, DT_FLOAT}))
     .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
     .REQUIRED_ATTR(min_size, ListFloat)
     .REQUIRED_ATTR(max_size, ListFloat)
     .ATTR(aspect_ratio, ListFloat, {1.0})
     .ATTR(img_size, Int, 0)
     .ATTR(img_h, Int, 0)
     .ATTR(img_w, Int, 0)
     .ATTR(step, Float, 0.0)
     .ATTR(step_h, Float, 0.0)
     .ATTR(step_w, Float, 0.0)
     .ATTR(flip, Bool, true)
     .ATTR(clip, Bool, false)
     .ATTR(offset, Float, 0.5)
     .ATTR(variance, ListFloat, {0.1})
     .OP_END_FACTORY_REG(PriorBox);

/**
*@brief Performs SSD prior box detection, with four additional matrices and the "aspect_ratio" attribute deleted compared to PriorBox.

*@par Inputs:
* Six inputs, including:
*@li feature: An NC1HWC0 or NCHW feature map of type is float32 or float16.
*@li img: source image. Has the same type and format as "feature".
*@li data_h: An NC1HWC0 or NCHW tensor of type float32 or float16, specifying the matrix for indexing the feature map height.
*@li data_w: An NC1HWC0 or NCHW tensor of type float32 or float16, specifying the matrix for indexing the feature map width.
*@li box_height: An NC1HWC0 or NCHW tensor of type float32 or float16, specifying the height of each prior box.
*@li box_width: An NC1HWC0 or NCHW tensor of type float32 or float16, specifying the width of each prior box.

*@par Attributes:
*@li min_size: A required float32, specifying the minimum edge length of a square prior box.
*@li max_size: A required float32, specifying the maximum edge length of a square prior box: sqrt(min_size * max_size)
*@li img_size: An optional int32, specifying the size of the source image.
*@li img_h: An optional int32, specifying the height of the source image.
*@li img_w: An optional int32, specifying the width of the source image.
*@li step: An optional float32, specifying the step for mapping the center point from the feature map to the source image.
*@li step_h: An optional float32, specifying the height step for mapping the center point from the feature map to the source image.
*@li step_w: An optional float32, specifying the width step for mapping the center point from the feature map to the source image.
*@li flip: An optional bool. If "True", "aspect_ratio" will be flipped. Defaults to "True".
*@li clip: An optional bool. If "True", a prior box is clipped to within [0, 1]. Defaults to "False".
*@li offset: An optional float32, specifying the offset. Defaults to "0.5".
*@li variance: An optional float32, specifying the variance of a prior box, either one or four variances. Defaults to "0.1" (one value).

*@par Outputs:
*y: An ND tensor of type float32 or float16, specifying the prior box information, including its coordinates and variance.

*@attention Constraints:\n
* This operator applies only to SSD networks.
*@see SSDDetectionOutput()
*/
 REG_OP(PriorBoxD)
     .INPUT(feature, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(img, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(data_h, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(data_w, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(box_height, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(box_width, TensorType({DT_FLOAT16, DT_FLOAT}))
     .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
     .REQUIRED_ATTR(min_size, ListFloat)
     .REQUIRED_ATTR(max_size, ListFloat)
     .ATTR(img_size, Int, 0)
     .ATTR(img_h, Int, 0)
     .ATTR(img_w, Int, 0)
     .ATTR(step, Float, 0.0)
     .ATTR(step_h, Float, 0.0)
     .ATTR(step_w, Float, 0.0)
     .ATTR(flip, Bool, true)
     .ATTR(clip, Bool, false)
     .ATTR(offset, Float, 0.5)
     .ATTR(variance, ListFloat, {0.1})
     .OP_END_FACTORY_REG(PriorBoxD);

 } // namespace ge

 #endif // GE_OP_PRIORBOX_H
