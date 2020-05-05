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

#ifndef GE_OP_ROIPOOLING_OPS_H_
#define GE_OP_ROIPOOLING_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Performs Region of Interest (ROI) pooling.

*@par Inputs:
* Three inputs, including:
*@li x: An NC1HWC0 tensor of type float16 or float32, describing the feature map.
*@li rois: A tensor of type float16 or float32, with shape [batch, 5, roi_max_num], describing the RIOs.
*@li roi_actual_num: A tensor of type int32, with shape [batch, 8], specifying the number of ROIs per batch.

*@par Attributes:
*@li roi_max_num: An optional int32, specifying the maximum number of ROIs per batch, at most 6000. Defaults to "3008". The value must be a multiple of 16.
*@li pooled_h: A required int32, specifying the pooled H. Must be greater than 0.
*@li pooled_w: A required int32, specifying the pooled W. Must be greater than 0.
*@li spatial_scale: An optional scaling factor for mapping the input coordinates to the ROI coordinates. Defaults to "0.0625".

*@par Outputs:
*y: An NC1HWC0 tensor of type float16 or float32, describing the result feature map.

*@attention Constraints:\n
*@li For the feature map input: \n
(1) If pooled_h = pooled_w = 2, the feature map size must not exceed 50. \n
(2) If pooled_h = pooled_w = 3, the feature map size must not exceed 60. \n
(3) If pooled_h = pooled_w = 4, the feature map size must not exceed 70. \n
(4) If pooled_h = pooled_w = 5, the feature map size must not exceed 70. \n
(5) If pooled_h = pooled_w = 6, the feature map size must not exceed 80. \n
(6) If pooled_h = pooled_w = 7, the feature map size must not exceed 80. \n
(7) If pooled_h = pooled_w = 8, the feature map size must not exceed 80. \n
(8) If pooled_h = pooled_w = 9, the feature map size must not exceed 70. \n
(9) If pooled_h = pooled_w = 10, the feature map size must not exceed 70. \n
(10) If pooled_h = pooled_w = 11, the feature map size must not exceed 70. \n
(11) If pooled_h = pooled_w = 12, the feature map size must not exceed 70. \n
(12) If pooled_h = pooled_w = 13, the feature map size must not exceed 70. \n
(13) If pooled_h = pooled_w = 14, the feature map size must not exceed 70. \n
(14) If pooled_h = pooled_w = 15, the feature map size must not exceed 70. \n
(15) If pooled_h = pooled_w = 16, the feature map size must not exceed 70. \n
(16) If pooled_h = pooled_w = 17, the feature map size must not exceed 50. \n
(17) If pooled_h = pooled_w = 18, the feature map size must not exceed 40. \n
(18) If pooled_h = pooled_w = 19, the feature map size must not exceed 40. \n
(19) If pooled_h = pooled_w = 20, the feature map size must not exceed 40. \n
*/

REG_OP(RoiPooling)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(rois, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(roi_actual_num, TensorType({DT_INT32}))
    .ATTR(roi_max_num, Int,3008)
    .REQUIRED_ATTR(pooled_h, Int)
    .REQUIRED_ATTR(pooled_w, Int)
    .ATTR(spatial_scale, Float, 0.0625)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(RoiPooling)

}  // namespace ge

#endif  // GE_OP_BITWISE_OPS_H_
