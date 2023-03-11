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

/*!
 * \file image.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_IMAGE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_IMAGE_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief LUT3D
* calculating: img, lut_table, lut_img
*    c = img.shape[- 1]
*    n_h_w = reduce(lambda x, y: x * y, img.shape[:-1])
*    lut_table_n = lut_table.shape[0]
*    tensor_img = img.reshape(n_h_w, c)
*    tensor_lut = np.cast["float32"](lut_table.reshape(lut_table_n * lut_table_n * lut_table_n, 3))
*    img_fp32 = np.cast["float32"](tensor_img)
*    img_fp32 = img_fp32 * (lut_table_n - 1) / 255.
*
*    b_floor = np.cast["int32"](np.floor(img_fp32[:, 0]))
*    b_ceil = np.cast["int32"](np.ceil(img_fp32[:, 0]))
*    g_floor = np.cast["int32"](np.floor(img_fp32[:, 1]))
*    g_ceil = np.cast["int32"](np.ceil(img_fp32[:, 1]))
*    r_floor = np.cast["int32"](np.floor(img_fp32[:, 2]))
*    r_ceil = np.cast["int32"](np.ceil(img_fp32[:, 2]))
*
*    b_fl_idx = b_floor * lut_table_n * lut_table_n
*    b_cl_idx = b_ceil * lut_table_n * lut_table_n
*    g_fl_idx = g_floor * lut_table_n
*    g_cl_idx = g_ceil * lut_table_n
*    r_fl_idx = r_floor
*    r_cl_idx = r_ceil
*
*    tensor_index1 = b_fl_idx + g_fl_idx + r_fl_idx
*    tensor_index2 = b_cl_idx + g_fl_idx + r_fl_idx
*    tensor_index3 = b_fl_idx + g_cl_idx + r_fl_idx
*    tensor_index4 = b_fl_idx + g_fl_idx + r_cl_idx
*    tensor_index5 = b_fl_idx + g_cl_idx + r_cl_idx
*    tensor_index6 = b_cl_idx + g_fl_idx + r_cl_idx
*    tensor_index7 = b_cl_idx + g_cl_idx + r_fl_idx
*    tensor_index8 = b_cl_idx + g_cl_idx + r_cl_idx
*
*    lut_tensor1 = tensor_lut[tensor_index1]
*    lut_tensor2 = tensor_lut[tensor_index2]
*    lut_tensor3 = tensor_lut[tensor_index3]
*    lut_tensor4 = tensor_lut[tensor_index4]
*    lut_tensor5 = tensor_lut[tensor_index5]
*    lut_tensor6 = tensor_lut[tensor_index6]
*    lut_tensor7 = tensor_lut[tensor_index7]
*    lut_tensor8 = tensor_lut[tensor_index8]
*
*    fract_b = np.repeat((img_fp32[:, 0] - b_floor)[:, None], 3, 1)
*    fract_b_1 = 1 - fract_b
*    fract_g = np.repeat((img_fp32[:, 1] - g_floor)[:, None], 3, 1)
*    fract_g_1 = 1 - fract_g
*    fract_r = np.repeat((img_fp32[:, 2] - r_floor)[:, None], 3, 1)
*    fract_r_1 = 1 - fract_r
*
*    lut_img = ((lut_tensor1 * fract_r_1 + lut_tensor4 * fract_r) * fract_g_1 \
*              + (lut_tensor3 * fract_r_1 + lut_tensor5 * fract_r) * fract_g) * fract_b_1 \
*              + ((lut_tensor2 * fract_r_1 + lut_tensor6 * fract_r) * fract_g_1 \
*              + (lut_tensor7 * fract_r_1 + lut_tensor8 * fract_r) * fract_g) * fract_b
*
* @par Inputs:
* Two inputs, including:
* @li img: A Tensor. Must be one of the following types: uint8, float32.
* @li lut_table: A Tensor. Has the same type as "img" . \n

* @par Outputs:
* lut_img: A Tensor of type float32. Has the same shape as "img" . \n
*/
REG_OP(LUT3D)
    .INPUT(img, TensorType({DT_UINT8, DT_FLOAT}))
    .INPUT(lut_table, TensorType({DT_UINT8, DT_FLOAT}))
    .OUTPUT(lut_img, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(LUT3D)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_IMAGE_H_
