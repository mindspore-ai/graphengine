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

#ifndef GE_OP_MAGE_OPS_H_
#define GE_OP_MAGE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(AdjustHue)
    .INPUT(images, TensorType({DT_FLOAT}))
    .INPUT(delta, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustHue)

REG_OP(AdjustSaturation)
    .INPUT(images, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustSaturation)

REG_OP(AdjustContrast)
    .INPUT(images, TensorType({DT_FLOAT}))
    .INPUT(contrast_factor, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustContrast)

REG_OP(CropAndResize)
    .INPUT(images, TensorType({DT_UINT8, DT_UINT16, DT_INT8, \
        DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .INPUT(crop_size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(extrapolation_value, Float, 0)
    .ATTR(method, String, "bilinear")
    .OP_END_FACTORY_REG(CropAndResize)

REG_OP(CropAndResizeGradBoxes)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(images, TensorType({DT_UINT8, DT_UINT16, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(method, String, "bilinear")
    .OP_END_FACTORY_REG(CropAndResizeGradBoxes)

REG_OP(CropAndResizeGradImage)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .INPUT(image_size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(method, String, "bilinear")
    .REQUIRED_ATTR(T, Type)
    .OP_END_FACTORY_REG(CropAndResizeGradImage)

REG_OP(ExtractGlimpse)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(size, TensorType({DT_INT32}))
    .INPUT(offsets, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(centered, Bool, true)
    .ATTR(normalized, Bool, true)
    .ATTR(uniform_noise, Bool, true)
    .ATTR(noise, String, "uniform")
    .OP_END_FACTORY_REG(ExtractGlimpse)

REG_OP(HSVToRGB)
    .INPUT(images, TensorType({ DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_DOUBLE }))
    .OP_END_FACTORY_REG(HSVToRGB)

REG_OP(QuantizedResizeBilinear)
    .INPUT(images, TensorType({ DT_FLOAT }))
    .INPUT(size, TensorType({ DT_INT32 }))
    .INPUT(min, TensorType({ DT_FLOAT }))
    .INPUT(max, TensorType({ DT_FLOAT }))
    .OUTPUT(resized_images, TensorType({ DT_FLOAT }))
    .OUTPUT(y_min, TensorType({ DT_FLOAT }))
    .OUTPUT(y_max, TensorType({ DT_FLOAT }))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(QuantizedResizeBilinear)

REG_OP(ResizeArea)
    .INPUT(images, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(ResizeArea)

REG_OP(ResizeBicubicGrad)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(original_image, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeBicubicGrad)

REG_OP(ResizeBicubic)
    .INPUT(images, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeBicubic)

/**
*@brief Performs the backpropagation of ResizeNearestNeighbor for training scenarios.

*@par Inputs:
* Two inputs, including:
*@li grads: A 4D Tensor, specifying the backpropagation gradients. Must be one of the following types: int8, uint8, int16, uint16, int32, int64, float16, float32, float64.
*@li size: A 1D Tensor of type int32, specifying the source image size (orig_height, orig_width).

*@par Attributes: \n
*align_corners: An optional bool. If "True", the centers of the corner pixels of the input and gradient tensors are aligned. Defaults to "False".

*@par Outputs: \n
*y: A 4D Tensor, specifying the backpropagation gradient after computation. Has the same type as "grads".

*@attention Constraints:
* When the inputs are of type float32, the execution performance is high.

*@see ResizeNearestNeighbor
*/
REG_OP(ResizeNearestNeighborGrad)
    .INPUT(grads, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                              DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                           DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeNearestNeighborGrad)

REG_OP(ResizeNearestNeighborGradD)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(size, ListInt)
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(ResizeNearestNeighborGradD)

/**
*@brief Performs the backpropagation of ResizeBilinear, which is used to resize an image\n to a specified size, while this operator is used to restore the resized image to the original image.
*@par Inputs:
* Two inputs, including:
* @li grads: A float32 input in NC1HWC0 format, describing the image information after resizing,\n including the image height, width, number of channels, and number of images.
* @li original_image: A float32 input in NC1HWC0 format, describing the image information before resizing,\n including the image height, width, number of channels, and number of images.


*@par Attributes:
*align_corners: An optional bool. If "True", the centers of the corner pixels of the input and\n gradient tensors are aligned. Defaults to "False".

*@par Outputs:
*y: A float32 output in NC1HWC0 format, specifying the image information before resizing, including the image height,\n
width, number of channels, and number of images.
*/
REG_OP(ResizeBilinearGrad)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(original_image, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(ResizeBilinearGrad)

/**
*@brief Resizes "images" to "size" using bilinear interpolation.

*@par Inputs:
* Two inputs, including:
*@li images: An NC1HWC0 Tensor.
* Must be one of the following types: int8, uint8, int16, uint16, int32, int64, float16, float32, double
*@li size: An ND Tensor of type int32.

*@par Attributes:
*align_corners: An optional bool. If "true", the centers of the corner pixels of the input and output tensors are aligned. Defaults to "false".

*@par Outputs:
*y: A Tensor with the same format as input "images".
*/
REG_OP(ResizeBilinear)
    .INPUT(images, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                               DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                           DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(ResizeBilinear)

REG_OP(RGBToHSV)
    .INPUT(images, TensorType({ DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_DOUBLE }))
    .OP_END_FACTORY_REG(RGBToHSV)

REG_OP(SampleDistortedBoundingBoxExt2)
    .INPUT(image_size, TensorType({ DT_UINT8, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64 }))
    .INPUT(bounding_boxes, TensorType({ DT_FLAOT }))
    .INPUT(min_object_covered, TensorType({ DT_FLOAT }))
    .OUTPUT(begin, TensorType({ DT_UINT8, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64 }))
    .OUTPUT(size, TensorType({ DT_UINT8, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64 }))
    .OUTPUT(bboxes, TensorType({ DT_FLOAT }))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(aspect_ratio_range, ListFloat, { 0.75f, 1.33f })
    .ATTR(area_range, ListFloat, { 0.05f, 1.0f })
    .ATTR(max_attempts, Int, 100)
    .ATTR(use_image_if_no_bounding_boxes, Bool, false)
    .OP_END_FACTORY_REG(SampleDistortedBoundingBoxExt2)

/**
*@brief Resizes "images" to "size" using nearest neighbor interpolation.

*@par Inputs:
* Two inputs, including:
*@li images: An NC1HWC0 Tensor.
* Must be one of the following types: int8, uint8, int16, uint16, int32, int64, float16, float32, double
*@li size: An ND Tensor of type int32.

*@par Attributes:
*align_corners: An optional bool. If "true", the centers of the corner pixels of the input and output tensors are aligned. Defaults to "false".

*@par Outputs:
*y: A Tensor with the same type and format as input "images".
*/
REG_OP(ResizeNearestNeighbor)
    .INPUT(images, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                               DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                           DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(ResizeNearestNeighbor)

REG_OP(DrawBoundingBoxes)
    .INPUT(images, TensorType({DT_FLOAT}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(DrawBoundingBoxes)

REG_OP(NonMaxSuppression)
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .ATTR(iou_threshold, Float, 0.5f)
    .OP_END_FACTORY_REG(NonMaxSuppression)

REG_OP(NonMaxSuppressionV2)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(NonMaxSuppressionV2)

REG_OP(NonMaxSuppressionV3)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(NonMaxSuppressionV3)

REG_OP(NonMaxSuppressionV4)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OUTPUT(valid_outputs, TensorType({DT_INT32}))
    .ATTR(pad_to_max_output_size, Bool, false)
    .OP_END_FACTORY_REG(NonMaxSuppressionV4)

REG_OP(NonMaxSuppressionWithOverlaps)
    .INPUT(overlaps, TensorType({DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(overlap_threshold, TensorType({DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(NonMaxSuppressionWithOverlaps)

REG_OP(EncodeJpeg)
    .INPUT(image, TensorType({DT_UINT8}))
    .OUTPUT(contents, TensorType({DT_STRING}))
    .ATTR(format, String, "")
    .ATTR(quality, Int, 95)
    .ATTR(progressive, Bool, false)
    .ATTR(optimize_size, Bool, false)
    .ATTR(chroma_downsampling, Bool, true)
    .ATTR(density_unit, String, "in")
    .ATTR(x_density, Int, 300)
    .ATTR(y_density, Int, 300)
    .ATTR(xmp_metadata, String, "")
    .OP_END_FACTORY_REG(EncodeJpeg)

REG_OP(EncodePng)
    .INPUT(image, TensorType({DT_UINT8, DT_UINT16}))
    .OUTPUT(contents, TensorType({DT_STRING}))
    .ATTR(compression, Int, -1)
    .OP_END_FACTORY_REG(EncodePng)

/**
*@brief Resizes "images" to "size" using bilinear interpolation.

*@par Inputs:
* One input:
*images: An NC1HWC0 Tensor. \n
* Must be one of the following types: float16, float32.

*@par Attributes:
*@li size: A required int32 Tensor specifying the new size for the images. No default value.
*@li align_corners: An optional bool. If "true", the centers of the corner pixels of the input and output tensors are aligned. Defaults to "false".

*@par Outputs:
*y: A Tensor with type float32 and the same format as input "images".

*@attention Constraints:
*@li The input "size" must be a tensor of 2 elements: size[0] <= 2048, size[1] <= 2048.
*@li The input "images" must be a tensor of 5 elements: images[2] <= 2048, images[3] <= 2048.
*/
REG_OP(ResizeBilinearD)
    .INPUT(images, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .REQUIRED_ATTR(size, ListInt)
    .OP_END_FACTORY_REG(ResizeBilinearD)

/**
*@brief Resizes "images" to "size" using nearest neighbor interpolation.

*@par Inputs:
* One input:
*images: An NC1HWC0 Tensor. \n
* Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
*@li size: A required int32 Tensor specifying the new size for the images. No default value.
*@li align_corners: An optional bool. If "true", the centers of the corner pixels of the input and output tensors are aligned. Defaults to "false".

*@par Outputs:
*y: A Tensor with the same type and format as input "images".

*@attention Constraints:
* The input "size" must be a tensor of 2 elements: size[0] <= 7680, size[1] <= 4320
*/
REG_OP(ResizeNearestNeighborD)
    .INPUT(images, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .REQUIRED_ATTR(size, ListInt)
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(ResizeNearestNeighborD)

REG_OP(ExtractJpegShape)
    .INPUT(contents, TensorType({DT_STRING}))
    .OUTPUT(image_shape, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(output_type, Type)
    .OP_END_FACTORY_REG(ExtractJpegShape)
}  // namespace ge

#endif  // GE_OP_MAGE_OPS_H_
