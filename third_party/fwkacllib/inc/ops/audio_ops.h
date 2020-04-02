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

#ifndef GE_OP_AUDIO_OPS_H_
#define GE_OP_AUDIO_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Mel-Frequency Cepstral Coefficient (MFCC) calculation consists of taking the DCT-II of a log-magnitude mel-scale spectrogram.

*@par Inputs:
*The input spectrogram must be three-dimensional tensor, sample_rate must be a scalar. Inputs include: \n
* @li spectrogram:3D float tensor of mel-frequency cepstral coefficient.
* @li sample_rate:Mel-Frequency Cepstral Coefficient (MFCC) calculation sample rate.

*@par Attributes:
*@li upper_frequency_limit:Upper limit of the mfcc calculation frequency.
*@li lower_frequency_limit:Lower limit of the mfcc calculation frequency.
*@li filterbank_channel_count:Count of the channel filterbank.
*@li dct_coefficient_count:Count of the dct coefficient.

*@par Outputs:
*y:A float32 Tensor of the MFCCs of spectrogram.

*@attention Constraints:\n
*-The implementation for Mfcc on Ascend uses AI CPU, with bad performance.\n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(Mfcc)
    .INPUT(spectrogram, TensorType({DT_FLOAT}))
    .INPUT(sample_rate, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(upper_frequency_limit, Float, 4000)
    .ATTR(lower_frequency_limit, Float, 20)
    .ATTR(filterbank_channel_count, Int, 40)
    .ATTR(dct_coefficient_count, Int, 13)
    .OP_END_FACTORY_REG(Mfcc)

/**
*@brief Decode and generate spectrogram using wav float tensor.

*@par Inputs:
*The input x must be two-dimensional matrices. Inputs include: \n
* x:float tensor of the wav audio contents. contains length and channel

*@par Attributes:
*@li window_size:Size of the spectrogram window.
*@li stride:Size of the spectrogram stride.
*@li magnitude_squared:If true, using magnitude squared.

*@par Outputs:
*spectrogram:3-D float Tensor with the image contents.

*@attention Constraints:\n
*-The implementation for AudioSpectrogram on Ascend uses AI CPU, with bad performance.\n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(AudioSpectrogram)
    .INPUT(x, TensorType({DT_FLOAT}))
    .OUTPUT(spectrogram, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(window_size, Int)
    .REQUIRED_ATTR(stride, Int)
    .ATTR(magnitude_squared, Bool, false)
    .OP_END_FACTORY_REG(AudioSpectrogram)

/**
*@brief Decode a 16-bit WAV file into a float tensor.

*@par Inputs:
*The input contents must be string tensor. Inputs include: \n
* @li contents:A Tensor of type string. The WAV-encoded audio, usually from a file.

*@par Attributes:
*@li desired_channels:An optional int. Defaults to -1. Number of sample channels wanted.
*@li desired_samples:An optional int. Defaults to -1. Length of audio requested.

*@par Outputs:
*@li *audio:A Tensor of type float32.
*@li *sample_rate:A Tensor of type int32.

*@attention Constraints: \n
*-The implementation for DecodeWav on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(DecodeWav)
    .INPUT(contents, TensorType({DT_STRING}))
    .OUTPUT(audio, TensorType({DT_FLOAT}))
    .OUTPUT(sample_rate, TensorType({DT_INT32}))
    .ATTR(desired_channels, Int, -1)
    .ATTR(desired_samples, Int, -1)
    .OP_END_FACTORY_REG(DecodeWav)

REG_OP(EncodeWav)
    .INPUT(audio, TensorType({DT_FLOAT}))
    .INPUT(sample_rate, TensorType({DT_INT32}))
    .OUTPUT(contents, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(EncodeWav)
}   // namespace ge

#endif  // GE_OP_AUDIO_OPS_H_
