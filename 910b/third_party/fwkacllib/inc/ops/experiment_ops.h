/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file experiment_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Updates "var" according to the AdamW algorithm.
*
* @attention Constraints:
*  The input tensors must have the same shape.*
*
* @par Inputs:
* @li var: A mutable Tensor of the type TensorType::NumberType().
*     Should be from a Variable().
* @li m: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
* @li v: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
* @li beta1_power: A scalar of the same type as "var".
* @li beta2_power: A scalar of the same type as "var".
* @li lr: learning_rate. A scalar of the same type as "var".
* @li weight_decay: learning_rate. A scalar of the same type as "var".
* @li beta1: A scalar of the same type as "var".
* @li beta2: A scalar of the same type as "var".
* @li epsilon: A scalar of the same type as "var".
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li max_grad_norm: A mutable Tensor of the same type as "var", an optional input.
*     Should be from a Variable().
*
* @par Attributes:
* @li amsgrad: An optional bool. Defaults to "False".
*     If "True", max_grad_norm input and output must be entered.
* @li maximize: An optional bool. Defaults to "False".
*
* @par Outputs:
* @li var: A mutable tensor. Has the same type as input "var".
* @li m: A mutable tensor. Has the same type as input "m".
* @li v: A mutable tensor. Has the same type as input "v". \n
*/
REG_OP(ApplyAdamW)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(weight_decay, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OPTIONAL_INPUT(max_grad_norm, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .OUTPUT(v, TensorType::NumberType())
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamW)

/**
* @brief Calculate SQ distance. \n
*
* @par Inputs:
* @li ivf: A Tensor, dtype is uint8.
* @li query: A Tensor, dtype is float16 or float32.
* @li bucket_list: A Tensor, dtype is int32 or int64.
* @li bucket_limits: A Tensor, dtype is int32 or int64.
* @li bucket_offsets: A Tensor, dtype is int32 or int64.
* @li vmin: A Tensor, dtype is float16 or float32.
* @li vdiff: A Tensor, dtype is float16 or float32. \n
*
* @par Outputs:
* @li actual_count: A Tensor, dtype is int32 or int64, the actual number of sq_distance.
* @li sq_distance: A Tensor, dtype is float16 or float32.
* @li grouped_extreme_distance: A Tensor, dtype is float16 or float32, the extremum in each group of sq_distance.
* @li sq_ivf: A Tensor, dtype is int32 or int64.
* @li sq_index: A Tensor, dtype is int32 or int64. \n
*
* @par Attributes:
* @li total_limit: A Int, indicates the max length of the output sq_distance.
* @li group_size: A Int, indicates the group size of the extremum.
* @li extreme_mode: A Int, indicates the type of extremum, 0 means minimum, and 1 means maximum. \n
*
*/
REG_OP(ScanSQCodes)
    .INPUT(ivf, TensorType({DT_UINT8}))
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bucket_list, TensorType({DT_INT32, DT_INT64}))
    .INPUT(bucket_limits, TensorType({DT_INT32, DT_INT64}))
    .INPUT(bucket_offsets, TensorType({DT_INT32, DT_INT64}))
    .INPUT(vmin, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(vdiff, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(actual_count, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(sq_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grouped_extreme_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sq_ivf, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(sq_index, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(total_limit, Int)
    .ATTR(group_size, Int, 64)
    .ATTR(extreme_mode, Int, 0)
    .OP_END_FACTORY_REG(ScanSQCodes)

/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b". \n
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float32,
* float16, int32, int8, int4, bf16. 3D. Has format ND.
* @li x2: A matrix Tensor. Must be one of the following types: float32,
* float16, int32, int8, int4, bf16. 3D. Has format ND.
* @li bias: A optional Tensor. Must be one of the following types:
* float32, float16, int32, bf16. 1D. Has format ND.
* @li offset_w: A optional Tensor. Must be one of the following types:
* int8, int4. Has format ND. \n

* @par Attributes:
* Three attributes, including:
* @li perm_x1: A list int. "x1" is permuted to shape [B, M, K] before multiplication.
* @li perm_x2: A list int. "x2" is permuted to shape [B, K, N] before multiplication.
* @li perm_y: A list int. "y" is permuted after multiplication.
* @li offset_x: An optional integer for quantized TransposeBatchMatMul.
* The negative offset added to the input "x1" for int8, int4 type. Ensure offset_x
* within the effective range of input data type. Defaults to "0". \n

* @par Outputs:
* y: The result matrix Tensor. 3D. Must be one of the following
* types: float32, float16, int32, bf16. 3D. Has format ND. \n
*/
REG_OP(TransposeBatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .ATTR(perm_x1, ListInt, {})
    .ATTR(perm_x2, ListInt, {})
    .ATTR(perm_y, ListInt, {})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(TransposeBatchMatMul)

/**
* @brief Performs non-maximum suppression (NMS) on the rotated boxes according
* to their intersection-over-union (IoU). Rotated NMS interatively removes lower
* scoring rotated boxes which have an IoU greater than iou_threshold with
* another (higher scoring) rotated box.

* @par Inputs:
* Three inputs, including:
* @li boxes: A 2D Tensor of float16 or float32 with shape (N, 5). Rotated boxes to
* perform NMS on. They are expected to be in (x1, y1, x2, y2, angle_degress) format.
* @li scores: A 1D Tensor of float16 or float32 with shape (N). Scores for each one of
* the rotated boxes.
* @li labels: A 1D Tensor of int32 or int64 with shape (N). Labels for each one of
* the rotated boxes.

* @par Attributes:
* iou_threshold: A required float attribute. Discards all overlapping rotated
* boxes with IoU < iou_threshold.

* @par Outputs:
* Two outputs, including:
* @li selected_detections: A 2D Tensor of float16 or float32 with shape (N, 5).
* The selected boxes that kept by Rotated NMS, sorted in decreasing order of scores.
* @li keep_indices: A 1D Tensor of int32 or int64 with shape (N). The indices of
* selected_detections.

* @attention Constraints:
* Currently, the tensor type of input (boxes, scores) only support float.
* The tensor type of keep_indices only support int32.
*/
REG_OP(RotatedNMS)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(selected_detections, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(keep_indices, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(iou_threshold, Float)
    .OP_END_FACTORY_REG(RotatedNMS)

/**
* @brief According to the indices, return the value.

* @par Inputs:
* Four inputs, including:
* @li x: A ND Tensor.
* @li indexed_sizes: A 1D Tensor of int64 with shape (N). Sizes for each one of the indexed data.
* @li indexed_strides: A 1D Tensor of int64 with shape (N). Strides for each one of the indexed data.
* @li indices: Dynamic input. A ND Tensor of int64. return the value according to the indices.

* @par Outputs:
* y: The indexed output tensor. Has the same type and format as input "x".
*/
REG_OP(Index)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indexed_sizes, TensorType({DT_INT64}))
    .INPUT(indexed_strides, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Index)

/**
* @brief According to the index number of indexes, replace the value
* corresponding to X with the value.

* @par Inputs:
* Five inputs, including:
* @li x: A ND Tensor.
* @li value: A Tensor of the same type as "x".
* @li indexed_sizes: A 1D Tensor of int64 with shape (N). Sizes for each one of the indexed data.
* @li indexed_strides: A 1D Tensor of int64 with shape (N). Strides for each one of the indexed data.
* @li indices: Dynamic input. A Tensor of the indices.

* @par Attributes:
* @li accumulate: Does it support self accumulation. Defaults to false.

* @par Outputs:
* @li x: A Tensor.

* @par Third-party framework compatibility
* Compatible with the Pytorch operator index_put.

* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(IndexPutV2)
    .INPUT(x, TensorType::BasicType())
    .INPUT(value, TensorType::BasicType())
    .INPUT(indexed_sizes, TensorType({DT_INT64}))
    .INPUT(indexed_strides, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(x, TensorType::BasicType())
    .ATTR(accumulate, Bool, false)
    .OP_END_FACTORY_REG(IndexPutV2)

/**
* @brief Performs average pooling on the input. Used in the combination of conv + avgpoolupdate to replace avgpool
* @par Inputs:
* x1: Output of upstream Conv2d. A tensor of type float16, float32.
* x2: Input feature map of upstream Conv2d. A tensor of type int8, float16, float32.

* @par Attributes:
* @li ksize: A required list of 4 ints, specifying the size (N, C, H, and W) of the sliding window,
* where N = C = 1, and H and W are positive integers within the range [1, 255].
* @li strides: A required list of 4 ints, specifying the stride of the sliding window.
* The strides of the N and C dimensions are 1.
* The strides of the H and W dimensions are positive integers within the range [1, 63].
* @li padding_mode: A required string, specifying the padding algorithm,
* either "VALID", "SAME" and "CALCULATED".
* With "SAME" means that the outputs will have the same spatial dimensions as its inputs.
* With "VALID" means no padding.
* @li pads: Pad value when padding_mode is "CALCULATED".
* @li data_format: An optional string, specifying the data format of "ksize" and "strides",
* either "NCHW", or "NHWC" (default).
* @li ceil_mode: Use ceil or floor to calculate the output size when padding_mode is "CALCULATED".
* @li exclusive: Ignore padding area or not when calculating average.

* @par Outputs:
* y: The average pooled output tensor. Has the same type and format as input "x1".

* @attention Constraints:
* @li Only single input and single output are supported.
* @li "ksize_H" and "ksize_W" are positive integers within the range [1, 255]. ksize_H * ksize_W < 256
* @li Due to instruction restrictions,
* the values of "strides_h" and "strides_w" are positive integers within the range [1, 63].
* @par Third-party framework compatibility
* Compatible with the TensorFlow/Pytorch/Onnx operator AvgPoolV2.
*/
REG_OP(AvgPoolUpdate)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DA_INT4, DT_INT8, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NHWC")
    .ATTR(ceil_mode, Bool, false)
    .ATTR(exclusive, Bool, true)
    .OP_END_FACTORY_REG(AvgPoolUpdate)

/**
* @brief batch input by time
* @par Inputs:
* x: A list of input tensors. It's a dynamic input

* @par Attributes:
* @li window: time window, [-1, int64_max], if -1 will batch by input data flag,
* else will batch by input timestamp and data flag.
* @li batch_dim: [-1, input_shape_range), if -1 input shape:[x, ..., x] ---> output shape:[-1, x, ..., x],
* else output shape:[x, ..., -1(batch_dim), ..., x];
* @li drop_remainder: a bool flag, take effect when window > -1,
* if true when batch data window < window, will drop data.

* @par Outputs:
* y: A list of output tensors. It's a dynamic input, the same size as "x".

* @attention Constraints:
* @li Only support in helper udf
*/
REG_OP(TimeBatch)
    .DYNAMIC_INPUT(x, TensorType::RealNumberType())
    .DYNAMIC_OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(window, Int)
    .ATTR(batch_dim, Int, -1)
    .ATTR(drop_remainder, Bool, false)
    .OP_END_FACTORY_REG(TimeBatch)

/**
* @brief Auto Batch process. \n

* @par Inputs:
* @li x: A list of input tensor objects. It's a dynamic input. \n

* @par Outputs:
* @li y: A list of output tensor objects. It's a dynamic output. \n

* @par Attributes:
* @li batch_size: auto batch size.
* @li timeout: auto batch wait timeout(unit:ms).
* @li padding: weather to pad when batch is insufficient.
* @li slide_stride: sliding window step.
*/
REG_OP(CountBatch)
    .DYNAMIC_INPUT(x, TensorType::RealNumberType())
    .DYNAMIC_OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(batch_size, Int)
    .ATTR(timeout, Int, 0)
    .ATTR(padding, Bool, false)
    .ATTR(slide_stride, Int, 0)
    .OP_END_FACTORY_REG(CountBatch)

/**
* @brief YUVToRGB

* @par Inputs:
* @li x: A 4-D uint8 Tensor.
*        Must set the format, supported format list ["NYUV"].
* @li matrix: A 1-D float tensor of 2x3x3 elements

* @par Outputs:
* @li y: A 4-D uint8 Tensor.
*        Must set the format, supported format list ["NCHW, NHWC"].

* @par Attributes:
* @li matrix_type: An Int attr, Defaults to 0.
*                  support list [ 0: CSC_MATRIX_BT601_WIDE,
*                                 1: CSC_MATRIX_BT601_NARROW,
*                                 2: CSC_MATRIX_BT709_WIDE,
*                                 3: CSC_MATRIX_BT709_NARROW,
*                                 4: CSC_MATRIX_BT2020_WIDE,
*                                 5: CSC_MATRIX_BT2020_NARROW,
*                                 6: CSC_MATRIX_USR_DEFINE ]
* @li rb_swap: An Int attr, Defaults to 0.
*              support list [ 0: RGB, 1: BGR ]

* @attention Constraints:
* @li Only support in dvpp
*/

REG_OP(YUVToRGB)
    .INPUT(x, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(matrix, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .ATTR(matrix_type, Int, 0)
    .ATTR(rb_swap, Int, 0)
    .OP_END_FACTORY_REG(YUVToRGB)

/**
* @brief DecodeJpegPre

* @par Inputs:
* @li contents: A Tensor of type string. 0-D. The JPEG-encoded image.

* @par Outputs:
* @li dvpp_support: indicates if the dvpp support this jpeg image decode.

* @par Attributes:
* @li w_range: An required listInt contains width [min, max].
* @li h_range: An required listInt contains height [min, max].

* @attention Constraints:
* @li Only support in dvpp
*/

REG_OP(DecodeJpegPre)
    .INPUT(contents, TensorType({DT_STRING}))
    .OUTPUT(dvpp_support, BOOL)
    .REQUIRED_ATTR(w_range, ListInt)
    .REQUIRED_ATTR(h_range, ListInt)
    .OP_END_FACTORY_REG(DecodeJpegPre)

/**
* @brief init PartitionMap table. \n

* @par Inputs:
* @li ps_num: A Tensor, dtype is int32. 0-D. indicates ps number.
* @li ps_ids: A Tensor, dtype is int32. 1-D. indicates the id of ps. \n

* @par Attributes:
* @li partition_num: A Int, indicates the number of partition. \n
*/
REG_OP(InitPartitionMap)
    .INPUT(ps_num, TensorType({DT_INT32}))
    .INPUT(ps_ids, TensorType({DT_INT32}))
    .ATTR(partition_num, Int, 65537)
    .OP_END_FACTORY_REG(InitPartitionMap)

/**
* @brief uninit PartitionMap table. \n
*/
REG_OP(UninitPartitionMap)
    .OP_END_FACTORY_REG(UninitPartitionMap)

/**
* @brief init Embedding hashtable. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable. \n

* @par Attributes:
* @li value_total_len: A Int, indicates the length of hashtable value. \n
* @li embedding_dim: A Int, indicates the length of embedding. \n
* @li bucket_size: A Int, Defaults to "0". \n
* @li dtype: A Type for data, Defaults to "DT_FLOAT". \n
* @li random_alg: A String, "random_uniform" or "truncated_normal" or "", Defaults to "". \n
* @li seed: A Int, Defaults to "0". \n
* @li seed2: A Int, Defaults to "0". \n
*/
REG_OP(InitEmbeddingHashmap)
    .INPUT(table_id, TensorType({DT_INT32}))
    .ATTR(bucket_size, Int, 0)
    .REQUIRED_ATTR(value_total_len, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(dtype, Type, DT_FLOAT)
    .ATTR(random_alg, String, "")
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(InitEmbeddingHashmap)

/**
* @brief embedding hsahtable data import. \n

* @par Inputs:
* @li file_path: A Tensor, dtype is string. 0-D. indicates embedding filepath.
* @li file_name: A Tensor, dtype is string. 0-D. indicates embedding filename.
* @li ps_id: A Tensor, dtype is int32. 0-D. indicates the id of ps.
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int. indicates the hashtable value number.
* @li value_total_length: A Int. indicates the hashtable total length, inclue m+v or accum.
* @li only_var: A Bool. only import var.
* @li file_type: A String. indicates the import file . \n
*/
REG_OP(EmbeddingTableImport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(file_name, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .REQUIRED_ATTR(value_total_len, Int)
    .ATTR(only_var_flag, Bool, false)
    .ATTR(file_type, String, "bin")
    .OP_END_FACTORY_REG(EmbeddingTableImport)

/**
* @brief embedding hsahtable data lookup. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable.
* @li keys: A Tensor, dtype is uint32. 1-D. indicates the hashtable key. \n

* @par Outputs:
* @li values: indicates the hashtable value. \n

* @par Attributes:
* @li embedding_dim: A Int. indicates the hashtable value number. \n
*/
REG_OP(EmbeddingTableFind)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(EmbeddingTableFind)

/**
* @brief uninit embedding hsahtable. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable. \n
*/
REG_OP(UninitEmbeddingHashmap)
    .INPUT(table_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(UninitEmbeddingHashmap)

/**
* @brief embedding hashtable lookup or init. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is DT_INT32. 0-D. indicates the id of hashtable.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key. \n

* @par Outputs:
* @li values: indicates the hashtable value. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding var value in hashtable.
* @li value_total_len: A Int, indicates the dim of embedding var+m+v or var+accum values in hashtable
* @li random_alg: A String of "random_uniform" or "truncated_normal", indicates the algo of init method.
* Defaults to "random_uniform".
* @li seed: An Int, Used to create a random seed, Defaults to "0".
* @li seed2: An Int, Used to create a random seed, Defaults to "0". \n
*/
REG_OP(EmbeddingTableFindAndInit)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .REQUIRED_ATTR(value_total_len, Int)
    .ATTR(random_alg, String, "random_uniform")
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(EmbeddingTableFindAndInit)

/**
* @brief embedding hashtable embedding applyadam. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li beta1_power: A Scalar, dtype is DT_FLOAT16 or DT_FLOAT. 0-D. indicates the beta1's power.
* @li beta2_power: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta2's power.
* @li lr: A Scalar, dtype is same as "beta1_power". 0-D. indicates the learning rate.
* @li beta1: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta1 param.
* @li beta2: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta2 param.
* @li epsilon: A Scalar, dtype is same as "beta1_power". 0-D. indicates the small value param.
* @li grad: A Tensor, dtype is same as "beta1_power". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable. \n
*/
REG_OP(EmbeddingApplyAdam)
    .INPUT(var_handle, TensorType({DT_RESOURCE}))
    .INPUT(beta1_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(epsilon, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(EmbeddingApplyAdam)

/**
* @brief embedding hashtable embedding applyadamW. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li beta1_power: A Scalar, dtype is DT_FLOAT16 or DT_FLOAT. 0-D. indicates the beta1's power.
* @li beta2_power: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta2's power.
* @li lr: A Scalar, dtype is same as "beta1_power". 0-D. indicates the learning rate.
* @li weight_decay: A Scalar, dtype is same as "beta1_power". 0-D. indicates the weight decay.
* @li beta1: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta1 param.
* @li beta2: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta2 param.
* @li epsilon: A Scalar, dtype is same as "beta1_power". 0-D. indicates the small value param.
* @li grad: A Tensor, dtype is same as "beta1_power". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li max_grad_norm: A mutable Tensor of the same type as "beta1_power", an optional input. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable.
* @li amsgrad: An optional bool, indicates whether to use the AMSGrad variant of htis algorithm from
*     the paper On the Convergence of Adam and Beyond(default:False).
*     If "True", max_grad_norm input and output must be entered.
* @li maximize: An optional bool, maximize the params based on the objective(default:False). \n
*/
REG_OP(EmbeddingApplyAdamW)
    .INPUT(var_handle, TensorType({DT_RESOURCE}))
    .INPUT(beta1_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(weight_decay, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(epsilon, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(max_grad_norm, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingApplyAdamW)
/**
* @brief embedding hashtable export. \n

* @par Inputs:
* @li file_path: A String, indicates the export file path.
* @li file_name: A String, indicates the export file name.
* @li ps_id: A Int, dtype is DT_INT32, indicates the ps server id.
* @li table_id: A Int, dtype is DT_INT32, indicates the hashtable id.

* @par Attributes:
* @li embedding_dim: A Int. indicates the hashtable value number.
* @li value_total_length: A Int. indicates the hashtable total length, inclue m+v or accum.
* @li only_var: A Bool. only export var, Defaults to "false".
* @li file_type: A String. indicates the export file, Defaults to "bin". \n
*/
REG_OP(EmbeddingTableExport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(file_name, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .REQUIRED_ATTR(value_total_len, Int)
    .ATTR(only_var_flag, Bool, false)
    .ATTR(file_type, String, "bin")
    .OP_END_FACTORY_REG(EmbeddingTableExport)

/**
* @brief embedding tableid trans to resource. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable.

* @par Outputs:
* @li table_handle: indicates the resource_handle of tableid. \n
*/
REG_OP(TableToResource)
    .INPUT(table_id, TensorType({DT_INT32}))
    .OUTPUT(table_handle, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(TableToResource)

/**
* @brief embedding feature_id trans to offset_id. \n

* @par Inputs:
* @li feature_id: A Tensor, dtype is int64.

* @par Outputs:
* @li offset_id: A Tensor with same shape of feature_id, dtype is int32. \n
*/
REG_OP(EmbeddingFeatureMapping)
    .INPUT(feature_id, TensorType({DT_INT64}))
    .OUTPUT(offset_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(EmbeddingFeatureMapping)

/**
* @brief embedding hashtable resource applyadagrad. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li lr: A Scalar, dtype is DT_FLOAT/DT_FLOAT16. 0-D. indicates the learning rate.
* @li grad: A Tensor, dtype is DT_FLOAT/DT_FLOAT16. 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable. \n
*/
REG_OP(EmbeddingApplyAdaGrad)
    .INPUT(var_handle, TensorType({DT_RESOURCE}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(EmbeddingApplyAdaGrad)

/**
* @brief embedding compute var export. \n

* @par Inputs:
* @li file_path: A String, indicates the export file path.
* @li ps_id: A Int, dtype is int32, indicates the ps server id.
* @li table_id: A Int, dtype is int32, indicates the hashtable id.

*/
REG_OP(EmbeddingComputeVarExport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(EmbeddingComputeVarExport)

/**
* @brief embedding compute var import. \n

* @par Inputs:
* @li file_path: A String, indicates the import file path.
* @li ps_id: A Int, dtype is int32, indicates the ps server id.
* @li table_id: A Int, dtype is int32, indicates the hashtable id.

*/
REG_OP(EmbeddingComputeVarImport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(EmbeddingComputeVarImport)

/**
* @brief Computes the output as scale * (x + bias) if x+bias > 0 and scale * negative_slope * (x+bias)
* if x+bias <= 0 . \n

* @par Inputs:
* Two input:
* x: A Tensor. Must be one of the following types: float32, float16, double.
* bias: A Tensor. Must be one of the following types: float32, float16, double.
*
* @par Attributes:
* negative_slope: A float32. Defaults to "0.2".
* sacle: A float32. Defaults to "2**0.5".
*
* @par Outputs:
* y: A Tensor. Has the same type as "x".
* @par Third-party framework compatibility
* Compatible with the mmcv operator FusedBiasLeakyrelu.
*/
REG_OP(FusedBiasLeakyRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .ATTR(negative_slope, Float, 0.2)
    .ATTR(scale, Float, 1.414213562373)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FusedBiasLeakyRelu)

/**
* @brief Computes the output as scale * gradients if features > 0 and
* negative_slope * gradients * scale if features <= 0 . \n

* @par Inputs:
* Two inputs, including:
* @li y_grad: A Tensor. Must be one of the following types: float16, float32, double.
* @li features: A Tensor. Has the same type as "gradients" . \n

* @par Attributes:
* negative_slope: A float32. Defaults to "0.2" . \n
* scale : A float32. Defaults to "2**0.5"

* @par Outputs:
* x_grad: A Tensor. Has the same type as "y_grad" . \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator FusedBiasLeakyReluGrad.
*/
REG_OP(FusedBiasLeakyReluGrad)
    .INPUT(y_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(negative_slope, Float, 0.2)
    .ATTR(scale, Float, 1.414213562373)
    .OUTPUT(x_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FusedBiasLeakyReluGrad)


/**
* @brief Set initial values for memory of sizes list . \n

* @par Attributes:
* @li sizes: sizes of workspaces. \n
* @li dtypes: data types of initial values. \n
* @li values_int: integer values to be set. \n
* @li values_float: float values to be set. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(MemSet)
    .REQUIRED_ATTR(sizes, ListInt)
    .ATTR(dtypes, ListType, {})
    .ATTR(values_int, ListInt, {})
    .ATTR(values_float, ListFloat, {})
    .OP_END_FACTORY_REG(MemSet)

/**
* @brief Performs the backpropagation of DeformableRoiPool for training scenarios . \n

* @par Inputs:
* Four inputs, including:
* @li grad_output: A 5HD gradient input of type float32
* @li feature_map: A 5HD Tensor of type float32.
* @li rois: ROI position. A 2D Tensor of float32 with shape (N, 5). "N" indicates the number of ROIs,
* the value "5" indicates the indexes of images where the ROIs are located, "x0", "x1", "y0" and "y1".
* @li offset: An optional 5HD Tensor input, specifying the offset of sampled points . \n

* @par Attributes:
* Four attributes, including:
* @li output_size: A required list of 2 ints, obtained based on the shape of "output" of DeformableRoiPool.
* @li spatial_scale: A optional attribute of type float, specifying the scaling ratio of "feature_map"
* to the original image.
* @li sample_ratio: An optional attribute of type int, specifying the horizontal and vertical sampling
* frequency of each output.
* If this attribute is set to "0", the sampling frequency is equal to the rounded up value of "rois",
* which is a floating point number. Defaults to "0".
* @li gamma: An optional attribute of type float, specfying the scaling factor of offset . \n

* @par Outputs:
* @li grad_fm: Gradient added to input "features". Has the same 5HD shape as input "features".
* @li grad_offset: Gradient added to input "offset". Has the same 4D shape as input "offset".
*/
REG_OP(DeformableRoiPoolGrad)
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT}))
    .OUTPUT(grad_offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(spatial_scale, Float, 1.0)
    .ATTR(sampling_ratio, Int, 0)
    .ATTR(gamma, Float, 0.1)
    .OP_END_FACTORY_REG(DeformableRoiPoolGrad)

/**
* @brief find an optimal n for shift-n. \n

* @par Inputs:
* @li x: A Tensor. indicates the output of quantizable layers.
* @li scale_d: A Tensor, one number. indicates the scale of data.
* @li scale_w: A Tensor, must be one number or the same size as dim-C when x is NHWC/NCHW.
*              indicates the scale of weight. \n

* @par Outputs:
* @li n: A Tensor, has the same shape as scale_w. indicates the optimal n. \n
*/
REG_OP(SearchN)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scale_d, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scale_w, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(n, TensorType({DT_INT8}))
    .OP_END_FACTORY_REG(SearchN)

/**
* @brief The operator generates three assist matrixs which will be used in AdaptiveAvgPool2d. \n

* @par Input:
* input_size: A Tensor of type int64.  \n

* @par Outputs:
* three inputs, including:
* @li left_matrix: A Tensor of type float32.  \n
* @li right_matrix: A Tensor of type float32.  \n
* @li weight_matrix: A Tensor of type float32.  \n

* @par Attributes:
* output_size: A required attribute.  \n
*/
REG_OP(AdaptiveAvgPool2dAssistMatrix)
    .INPUT(input_size, TensorType({DT_INT64}))
    .OUTPUT(left_matrix, TensorType({DT_FLOAT}))
    .OUTPUT(right_matrix, TensorType({DT_FLOAT}))
    .OUTPUT(weight_matrix, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(output_size, ListInt)
    .OP_END_FACTORY_REG(AdaptiveAvgPool2dAssistMatrix)

/**
* @brief Compute correct bounding box.

* @par Inputs:
* Three inputs, including:
* @li x: A 5D Tensor of type float16 with shape (N, na, no, H, W), na indicates the number of anchors,
* no indicates the number of outputs per anchor, including [xywh, class_num, conf_score].
* @li grid: A 5D Tensor of type float16 with shape (1, na, 2, H, W) for V3/V5 and (1, 1, 2, H, W) for V7,
* the value "2" indicates offsets of coordinates.
* @li anchor_grid: A 5D Tensor of type float16 with shape (1, na, 2, H, W) for V3/V5 and (1, 1, 2, 1, 1) for V7,
* the value "2" indicates anchors relative to the original image.

* @par Attributes:
* @li stride: A required int32, scale for each box.
* @li yolo_version: A required string, specifying the YOLO version, optional [V3, V5, V7].

* @par Outputs:
* @li y: A 5D Tensor of type float16 with shape (N, na, no, H, W), same as the input x.

* @par attention Constraints:
* @li This operator applies to YOLO V3, V5 and V7 networks.
* @par Third-party framework compatibility
* It is a custom operator.
*/
REG_OP(CorrectBBox)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(grid, TensorType({DT_FLOAT16}))
    .INPUT(anchor_grid, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(stride, Int)
    .REQUIRED_ATTR(yolo_version, String)
    .OP_END_FACTORY_REG(CorrectBBox)

/**
* @brief Obtains the ROI feature matrix from the feature map. It is a customized FasterRcnn operator . \n

* @par Inputs:
* Three inputs, including:
* @li features: A 5HD Tensor of type float32 or float16.
* @li rois: ROI position. A 2D Tensor of float32 or float16 with shape (N, 5). "N" indicates the number of ROIs,
*     the value "5" indicates the indexes of images where the ROIs are located, "x0", "y0", "x1", and "y1".
* @li offset: An optional input of type float32 or float16, offset of height and width defaults to a Tensor of zero . \n

* @par Attributes:
* @li spatial_scale: A required attribute of type float32, specifying the scaling ratio of "features"
*     to the original image.
* @li pooled_height: A required attribute of type int32, specifying the H dimension.
* @li pooled_width: A required attribute of type int32, specifying the W dimension.
* @li sampling_ratio: An optional attribute of type int32, specifying the horizontal and vertical sampling frequency
*     of each output. If this attribute is set to "0",
* the sampling frequency is equal to the rounded up value of "rois", which is a floating point number. Defaults to "0".
* @li gamma: An optional attribute of type float32. Defaults to "0.1" . \n
* @par Outputs:
* output: Outputs the feature sample of each ROI position. The format is 5HD Tensor of type float32 or float16.
  The axis N is the number of input ROIs. Axes H, W, and C are consistent
* with the values of "pooled_height",
* "pooled_width", and "features", respectively.
*/
REG_OP(DeformableRoiPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(spatial_scale, Float, 1.0)
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(sampling_ratio, Int, 0)
    .ATTR(gamma, Float, 0.1)
    .OP_END_FACTORY_REG(DeformableRoiPool)

/**
 * @brief Generate the attention map of Point-wise Spatial Attention(PSA) \n

 * @par Inputs:
 * x: A Tensor of BasicType that indicates the global attention map from upstream computing. \n

 * @par Outputs:
 * y: A Tensor of BasicType that indicates the generated pixel-wise global attention map. \n

 * @par Attributes:
 * @li psa_type: An Int value of 1 or 2 that indicates the method used to generate pixel-wise global attention map.
 * @li num: An Int value that indicates the batch_size of input x.
 * @li h_feature: An Int value that indicates the hight of input feature map.
 * @li w_feature: An Int value that indicates the width of input feature map.
 * @li h_mask: An Int value that indicates the hight of the over-completed map.
 * @li w_mask: An Int value that indicates the width of the over-completed map.
 * @li half_h_mask: An Int value that indicates half of the hight of input feature map.
 * @li half_w_mask: An Int value that indicates half of the width of the over-completed map. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator PSAMask.\n
 */
REG_OP(PSAMask)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(psa_type, Int)
    .REQUIRED_ATTR(num, Int)
    .REQUIRED_ATTR(h_feature, Int)
    .REQUIRED_ATTR(w_feature, Int)
    .REQUIRED_ATTR(h_mask, Int)
    .REQUIRED_ATTR(w_mask, Int)
    .REQUIRED_ATTR(half_h_mask, Int)
    .REQUIRED_ATTR(half_w_mask, Int)
    .OP_END_FACTORY_REG(PSAMask)

/**
 * @brief Calculate the gradient of operator PSAMask \n

 * @par Inputs:
 * y_grad: A Tensor of BasicType that indicates the passed gradient. \n

 * @par Outputs:
 * x_grad: A Tensor of BasicType that indicates the calculated gradient. \n

 * @par Attributes:
 * @li psa_type: An Int value of 1 or 2 that indicates the method used to generate pixel-wise global attention map.
 * @li num: An Int value that indicates the batch_size of input x.
 * @li h_feature: An Int value that indicates the hight of input feature map.
 * @li w_feature: An Int value that indicates the width of input feature map.
 * @li h_mask: An Int value that indicates the hight of the over-completed map.
 * @li w_mask: An Int value that indicates the width of the over-completed map.
 * @li half_h_mask: An Int value that indicates half of the hight of input feature map.
 * @li half_w_mask: An Int value that indicates half of the width of the over-completed map. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator PSAMask.\n
 */
REG_OP(PSAMaskGrad)
    .INPUT(y_grad, TensorType::BasicType())
    .OUTPUT(x_grad, TensorType::BasicType())
    .REQUIRED_ATTR(psa_type, Int)
    .REQUIRED_ATTR(num, Int)
    .REQUIRED_ATTR(h_feature, Int)
    .REQUIRED_ATTR(w_feature, Int)
    .REQUIRED_ATTR(h_mask, Int)
    .REQUIRED_ATTR(w_mask, Int)
    .REQUIRED_ATTR(half_h_mask, Int)
    .REQUIRED_ATTR(half_w_mask, Int)
    .OP_END_FACTORY_REG(PSAMaskGrad)

/**
* @brief Find nearby points in spherical space or spherical layer. \n

* @par Inputs:
* Two inputs, including:
* @li xyz: A 3D Tensor of type float16 or float32, xyz coordinates of the features.
* @li center_xyz: A 3D Tensor of type float16 or float32. centers coordinates of the ball query. \n

* @par Attributes:
* @li min_radius: A required float, minimum radius of the balls.
* @li max_radius: A required float, maximum radius of the balls.
* @li sample_num: A required int, maximum number of features in the balls. \n

* @par Outputs:
* One outputs:
* @li idx: A 3D(B, M, sample_num) Tensor of type int32 with the indices of the features that form the query balls. \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator BallQuery(BallQuery branch).
*/
REG_OP(BallQuery)
    .INPUT(xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(center_xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(min_radius, Float)
    .REQUIRED_ATTR(max_radius, Float)
    .REQUIRED_ATTR(sample_num, Int)
    .OP_END_FACTORY_REG(BallQuery)

/**
* @brief Find nearby points in spherical space. \n

* @par Inputs:
* Four inputs, including:
* @li xyz: A 2D Tensor of type float16 or float32, xyz coordinates of the features.
* @li center_xyz: A 2D Tensor of type float16 or float32. Centers coordinates of the ball query.
* @li xyz_batch_cnt: A 1D Tensor of type int32 or int64, Stacked input xyz coordinates nums in
     each batch, just like (N1, N2, ...).
* @li center_xyz_batch_cnt: A 1D Tensor of type int32 or int64. Stacked input centers coordinates nums in
     each batch, just like (M1, M2, ...). \n

* @par Attributes:
* @li max_radius: A required float, maximum radius of the balls.
* @li sample_num: A required int, maximum number of features in the balls. \n

* @par Outputs:
* One outputs:
* @li idx: A 2D(M, sample_num) Tensor of type int32 with the indices of the features that form the query balls. \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator BallQuery(StackBallQuery branch).
*/
REG_OP(StackBallQuery)
    .INPUT(xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(center_xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(xyz_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .INPUT(center_xyz_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(max_radius, Float)
    .REQUIRED_ATTR(sample_num, Int)
    .OP_END_FACTORY_REG(StackBallQuery)

/**
 * @brief Find and get the corresponding value from the corresponding ps according to the keys
 * @par Inputs:
 * @li kyes: A tensor. Must be int64 type,
 * @li table_id: A tensor. Must be int32 type.
 * @par Outputs:
 * @li values: A Tensor. Must be float32 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li insert_option: Indicates whether lookup supports new value. Defaults to "0".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: Apply memory usage for output or infer shape.
 */
REG_OP(HcomRemoteLookup)
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(table_id, Int)
    .OUTPUT(values, TensorType({DT_FP32}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(insert_option, Int, 0)
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(HcomRemoteLookup)

/**
 * @brief Workers all find and get the corresponding value from the corresponding ps according to the keys
 * @par Inputs:
 * @li kyes: A tensor. Must be int64 type.
 * @li table_id: A tensor. Must be int32 type.
 * @par Outputs:
 * @li values: A Tensor. Must be float32 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li insert_option: Indicates whether lookup supports new value. Defaults to "0".
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: A required integer identifying Apply memory usage for output or infer shape.
 */
REG_OP(HcomCollRemoteLookup)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FP32}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(insert_option, Int, 0)
    .ATTR(group, String, "hccl_world_group")
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(HcomCollRemoteLookup)

/**
 * @brief Workers send the keys and values to ps according to keys
 * @par Inputs:
 * @li kyes: A tensor. Must be int64 type.
 * @li table_id: A tensor. Must be int32 type.
 * @li values: A Tensor. Must be float32 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: Apply memory usage for output or infer shape.
 */
REG_OP(HcomCollRemoteUpdate)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FP32}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(group, String, "hccl_world_group")
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(HcomCollRemoteUpdate)

/**
* @brief Find a min polygon from the point set in the operator MinAreaPolygons. \n

* @par Inputs:
* @li pointsets: A 2D Tensor with shape (N, 18), format ND, dtype must be one
 of the following types: float16, float32, double. \n

* @par Outputs:
* @li polygons: A 2D Tensor with shape (N, 8), format ND, dtype must be one of
 the following types: float16, float32, double.  \n
*/
REG_OP(MinAreaPolygons)
    .INPUT(pointsets, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(polygons, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(MinAreaPolygons)

/**
* @brief Calculate the index and distance of the nearest three point to the target point.
* @par Inputs:
* Two input:
* xyz1: The set of target points.
* xyz2: The set of compare points. \n

* @par Outputs:
* dist: A Tensor, the distance of the nearest point to the target point.
* idx: A Tensor, the index of the nearest point to the target point. \n
*/
REG_OP(ThreeNN)
    .INPUT(xyz1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(xyz2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(dist, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(ThreeNN)

/**
 * @brief Calculate the voxels of cloud points \n

 * @par Inputs:
 * Three inputs, including:
 * @li points: the shape is [M, C], points[:3] contain xyz points and points[3:] contain other information. 
 * @li voxel_size: the size of voxel with the shape of [3]. 
 * @li coors_range:the coordinate range of voxel with the shape of [6]. \n

 * @par Outputs:
 * Four outputs, including:
 * @li voxels: the output voxels with the shape of [M, max_points, C].
 * @li coors: the voxel coordinates with shape of [M, 3].
 * @li num_points_per_voxel: the number of points per voxel with the shape of [M].
 * @li voxel_num: the number of voxels. \n

 * @par Attributes:
 * Three attrs, including:
 * @li max_points: maximum points contained in a voxel.
 * @li max_voxels: maximum voxels this op create. 
 * @li deterministic: An optional attr, only support true now, false is faster. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator Voxelization.\n
 */
REG_OP(Voxelization)
    .INPUT(points, TensorType({DT_DOUBLE,DT_FLOAT,DT_FLOAT16}))
    .INPUT(voxel_size, TensorType({DT_DOUBLE,DT_FLOAT,DT_FLOAT16}))
    .INPUT(coors_range, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(voxels, TensorType({DT_DOUBLE,DT_FLOAT,DT_FLOAT16}))
    .OUTPUT(coors, TensorType({DT_INT32}))
    .OUTPUT(num_points_per_voxel, TensorType({DT_INT32}))
    .OUTPUT(voxel_num, TensorType({DT_INT32}))
    .ATTR(max_points, Int, 35)
    .ATTR(max_voxels, Int, 20000)
    .ATTR(deterministic, Bool, true)
    .OP_END_FACTORY_REG(Voxelization)

/**
 * @brief Encoding the orientation information and generating orientation-sensitive features. \n

 * @par Inputs:
 * Two inputs, including:
 * @li x: Input features with shape [num_output_planes, num_input_planes, num_orientations, H, W]. 
 * @li indices: Indices with shape [num_orientations, H, W, num_rotations]. \n

 * @par Outputs:
 * One output, including:
 * @li y: Refined features with shape [num_output_planes * num_rotations, num_input_planes * num_orientations, H, W]. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator ActiveRotatedFilter.\n
 */
REG_OP(ActiveRotatedFilter)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OP_END_FACTORY_REG(ActiveRotatedFilter)

/**
 * @brief The backward of ActiveRotatedFilter. \n

 * @par Inputs:
 * Two inputs, including:
 * @li y_grad: Input features with shape [num_output_planes * num_rotations, num_input_planes * num_orientations, H, W].
 * @li indices: Indices with shape [num_orientations, H, W, num_rotations]. \n

 * @par Outputs:
 * One output, including:
 * @li x_grad: Refined features with shape [num_output_planes, num_input_planes, num_orientations, H, W]. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator ActiveRotatedFilterGrad.\n
 */
REG_OP(ActiveRotatedFilterGrad)
    .INPUT(y_grad, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OP_END_FACTORY_REG(ActiveRotatedFilterGrad)

/**
* @brief Blend face iamge to the backgroud.
*
* @par Inputs:
* @li face_img: A 3D Tensor, dtype is uint8 or float32, shape is (h, w, 3). The input face image.
* @li face_rect: A 1D Tensor, dtype is int32, shape is (4,). The coordinates of the face image in the backgroud.
* @li face_mask: A 3D Tensor, dtype is float32, shape is (h, w, 1).
* @li acc_face: A 3D Tensor, dtype is float32, shape is (H, W, 3).
* @li acc_mask: A 3D Tensor, dtype is float32, shape is (H, W, 3).
* @li max_mask: A 3D Tensor, dtype is float32, shape is (H, W, 3).
*
* @par Outputs:
* @li acc_face: A 3D Tensor, Has the same type and shape as input "acc_face".
* @li acc_mask: A 3D Tensor, Has the same type and shape as input "acc_mask".
* @li max_mask: A 3D Tensor, Has the same type and shape as input "max_mask". \n
*/
REG_OP(BlendFaceBgPartOne)
    .INPUT(face_img, TensorType({DT_UINT8, DT_FLOAT}))
    .INPUT(face_rect, TensorType({DT_INT32}))
    .INPUT(face_mask, TensorType({DT_FLOAT}))
    .INPUT(acc_face, TensorType({DT_FLOAT}))
    .INPUT(acc_mask, TensorType({DT_FLOAT}))
    .INPUT(max_mask, TensorType({DT_FLOAT}))
    .OUTPUT(acc_face, TensorType({DT_FLOAT}))
    .OUTPUT(acc_mask, TensorType({DT_FLOAT}))
    .OUTPUT(max_mask, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BlendFaceBgPartOne)

/**
* @brief Convert the image from YUV to Raw.
*
* @par Inputs:
* @li img_channel_0: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 0.
* @li img_channel_1: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 1.
* @li img_channel_2: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 2.
* @li img_channel_3: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 3.
* @li img_size: A 1D Tensor, dtype is int32, shape is (2,).
*     The data is h_out and w_out, which indicates the output height and width.
* @li gamma: A 1D Tensor, dtype is int32, shape is (4,).
*
* @par Outputs:
* @li raw_img: A 2D Tensor, dtype is uint16, shape is (h_out, w_out). the output raw image. \n
*/
REG_OP(ImgRawDecodePostHandle)
    .INPUT(img_channel_0, TensorType({DT_UINT16}))
    .INPUT(img_channel_1, TensorType({DT_UINT16}))
    .INPUT(img_channel_2, TensorType({DT_UINT16}))
    .INPUT(img_channel_3, TensorType({DT_UINT16}))
    .INPUT(img_size, TensorType({DT_INT32}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .OUTPUT(raw_img, TensorType({DT_UINT16}))
    .OP_END_FACTORY_REG(ImgRawDecodePostHandle)

/**
* @brief RGB2YUV422. Convert the image from rgb to yuv422. \n

* @par Inputs:
* rgb: A Tensor of type uint8. \n
* @par Outputs:
* yuv: A Tensor of type uint8. \n

* @attention Constraints:
* Input images is a tensor of at least 3 dimensions. The last dimension is
* interpretted as channels, and must be three . \n
*/
REG_OP(RGB2YUV422)
    .INPUT(rgb, TensorType({DT_UINT8}))
    .OUTPUT(yuv, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(RGB2YUV422)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
