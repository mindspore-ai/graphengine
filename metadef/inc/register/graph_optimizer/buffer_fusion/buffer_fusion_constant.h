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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_CONSTANT_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_CONSTANT_H_
#include <string>
#include <vector>

namespace fe {
// add the op pattern
static const std::string TBE_PATTERN_INPUT_NODE = "InputData";
static const std::string TBE_PATTERN_OP_TYPE_ANY = "OpTypeAny";
static const std::string TBE_PATTERN_OUTPUT_NODE = "OutputData";
static const std::string OP_PATTERN_ELEMWISE = "ElemWise";
static const std::string OP_PATTERN_COMMONREDUCE = "CommReduce";
static const std::string OP_PATTERN_SEGMENT = "Segment";
static const std::string OP_PATTERN_MAXPOOL = "MaxPool";
static const std::string OP_PATTERN_CONV = "Convolution";
static const std::string OP_PATTERN_MATMUL = "Matmul";
static const std::string OP_PATTERN_BNUPDATE = "bn_update";
static const std::string OP_PATTERN_BNREDUCE = "bn_reduce";
static const std::string OP_PATTERN_CONV_BACKPROP_INPUT = "Conv2d_backprop_input";
static const std::string OP_PATTERN_DEPTHWISE_CONV = "DepthwiseConvolution";
static const std::string OP_PATTERN_QUANT = "quant";
static const std::string OP_PATTERN_DEQUANT = "dequant";
static const std::string OP_PATTERN_REQUANT = "requant";
static const std::string OP_PATTERN_POOL2D = "Pool2d";
static const std::string OP_PATTERN_ANTIQUANT = "anti_quant";
static const std::string OP_PATTERN_STRIDED_WRITE = "strided_write";
static const std::string OP_PATTERN_STRIDED_READ = "strided_read";
static const std::string OP_PATTERN_AIPP = "aipp";
static const std::string OP_PATTERN_CONFUSION_TRANSPOSE = "confusiontranspose";
static const std::string OP_PATTERN_DEQUANTS16 = "dequant_s16";
static const std::string OP_PATTERN_REQUANTS16 = "requant_s16";
static const std::string OP_PATTERN_READ_SELECT = "read_select";
static const std::string OP_PATTERN_WRITE_SELECT = "write_select";
static const std::string OP_PATTERN_BATCH_MATMUL = "BatchMatmul";
static const std::string OP_PATTERN_CONV3D = "Conv3d";

static const std::vector<std::string> OP_PATTERN_VEC{OP_PATTERN_ELEMWISE,
                                                     OP_PATTERN_COMMONREDUCE,
                                                     OP_PATTERN_SEGMENT,
                                                     OP_PATTERN_MAXPOOL,
                                                     OP_PATTERN_CONV,
                                                     OP_PATTERN_MATMUL,
                                                     OP_PATTERN_BNUPDATE,
                                                     OP_PATTERN_BNREDUCE,
                                                     OP_PATTERN_CONV_BACKPROP_INPUT,
                                                     OP_PATTERN_DEPTHWISE_CONV,
                                                     OP_PATTERN_QUANT,
                                                     OP_PATTERN_DEQUANT,
                                                     OP_PATTERN_REQUANT,
                                                     OP_PATTERN_POOL2D,
                                                     OP_PATTERN_ANTIQUANT,
                                                     OP_PATTERN_STRIDED_WRITE,
                                                     OP_PATTERN_STRIDED_READ,
                                                     OP_PATTERN_AIPP,
                                                     OP_PATTERN_CONFUSION_TRANSPOSE,
                                                     OP_PATTERN_DEQUANTS16,
                                                     OP_PATTERN_REQUANTS16,
                                                     OP_PATTERN_READ_SELECT,
                                                     OP_PATTERN_WRITE_SELECT,
                                                     OP_PATTERN_BATCH_MATMUL,
                                                     OP_PATTERN_CONV3D};
}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_CONSTANT_H_
