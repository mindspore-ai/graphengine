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

#ifndef COMMON_GRAPH_DEBUG_GE_OP_TYPES_H_
#define COMMON_GRAPH_DEBUG_GE_OP_TYPES_H_
#include <limits.h>
#include <stdint.h>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ge {
#define GE_REGISTER_OPTYPE(var_name, str_name) static const char* var_name __attribute__((unused)) = str_name

GE_REGISTER_OPTYPE(DATA, "Data");
GE_REGISTER_OPTYPE(AIPPDATA, "AippData");
GE_REGISTER_OPTYPE(CONVOLUTION, "Convolution");
GE_REGISTER_OPTYPE(CORRELATION, "Correlation");
GE_REGISTER_OPTYPE(CORRELATIONV2, "Correlation_V2");
GE_REGISTER_OPTYPE(DECONVOLUTION, "Deconvolution");
GE_REGISTER_OPTYPE(POOLING, "Pooling");
GE_REGISTER_OPTYPE(ELTWISE, "Eltwise");
GE_REGISTER_OPTYPE(RELU, "ReLU");
GE_REGISTER_OPTYPE(RELU6, "ReLU6");
GE_REGISTER_OPTYPE(SIGMOID, "Sigmoid");
GE_REGISTER_OPTYPE(ABSVAL, "AbsVal");
GE_REGISTER_OPTYPE(TANH, "TanH");
GE_REGISTER_OPTYPE(PRELU, "PReLU");
GE_REGISTER_OPTYPE(BATCHNORM, "BatchNorm");
GE_REGISTER_OPTYPE(FUSIONBATCHNORM, "FusionBatchNorm");
GE_REGISTER_OPTYPE(SCALE, "Scale");
GE_REGISTER_OPTYPE(FULL_CONNECTION, "FullConnection");
GE_REGISTER_OPTYPE(SOFTMAX, "Softmax");
GE_REGISTER_OPTYPE(PLUS, "Plus");
GE_REGISTER_OPTYPE(ACTIVATION, "Activation");
GE_REGISTER_OPTYPE(FLATTEN, "Flatten");
GE_REGISTER_OPTYPE(ADD, "Add");
GE_REGISTER_OPTYPE(SUB, "Sub");
GE_REGISTER_OPTYPE(MUL, "Mul");
GE_REGISTER_OPTYPE(MATMUL, "MatMul");
GE_REGISTER_OPTYPE(RSQRT, "Rsqrt");
GE_REGISTER_OPTYPE(BIASADD, "BiasAdd");
GE_REGISTER_OPTYPE(RESHAPE, "Reshape");
GE_REGISTER_OPTYPE(DEPCONVOLUTION, "ConvolutionDepthwise");
GE_REGISTER_OPTYPE(DROPOUT, "Dropout");
GE_REGISTER_OPTYPE(CONCAT, "Concat");
GE_REGISTER_OPTYPE(ROIPOOLING, "ROIPooling");
GE_REGISTER_OPTYPE(PROPOSAL, "Proposal");
GE_REGISTER_OPTYPE(FSRDETECTIONOUTPUT, "FSRDetectionOutput");
GE_REGISTER_OPTYPE(DETECTIONPOSTPROCESS, "Detectpostprocess");
GE_REGISTER_OPTYPE(LRN, "LRN");
GE_REGISTER_OPTYPE(TRANSDATA, "TransData");
GE_REGISTER_OPTYPE(PERMUTE, "Permute");
GE_REGISTER_OPTYPE(SSDNORMALIZE, "SSDNormalize");
GE_REGISTER_OPTYPE(SSDPRIORBOX, "SSDPriorBox");
GE_REGISTER_OPTYPE(NETOUTPUT, "NetOutput");
GE_REGISTER_OPTYPE(SSDDETECTIONOUTPUT, "SSDDetectionOutput");
GE_REGISTER_OPTYPE(CHANNELAXPY, "ChannelAxpy");
GE_REGISTER_OPTYPE(PSROIPOOLING, "PSROIPooling");
GE_REGISTER_OPTYPE(POWER, "Power");
GE_REGISTER_OPTYPE(ROIALIGN, "ROIAlign");
GE_REGISTER_OPTYPE(PYTHON, "Python");
GE_REGISTER_OPTYPE(FREESPACEEXTRACT, "FreespaceExtract");
GE_REGISTER_OPTYPE(SPATIALTF, "SpatialTransform");
GE_REGISTER_OPTYPE(SHAPE, "Shape");
GE_REGISTER_OPTYPE(ARGMAX, "ArgMax");
GE_REGISTER_OPTYPE(GATHERND, "GatherNd");
GE_REGISTER_OPTYPE(GATHER, "Gather");
GE_REGISTER_OPTYPE(REALDIV, "RealDiv");
GE_REGISTER_OPTYPE(PACK, "Pack");
GE_REGISTER_OPTYPE(SLICE, "Slice");
GE_REGISTER_OPTYPE(FLOORDIV, "FloorDiv");
GE_REGISTER_OPTYPE(SQUEEZE, "Squeeze");
GE_REGISTER_OPTYPE(STRIDEDSLICE, "StridedSlice");
GE_REGISTER_OPTYPE(RANGE, "Range");
GE_REGISTER_OPTYPE(RPNPROPOSALS, "GenerateRpnProposals");
GE_REGISTER_OPTYPE(DECODEBBOX, "DecodeBBox");
GE_REGISTER_OPTYPE(PAD, "Pad");
GE_REGISTER_OPTYPE(TILE, "Tile");
GE_REGISTER_OPTYPE(SIZE, "Size");
GE_REGISTER_OPTYPE(CLIPBOXES, "Clipboxes");
GE_REGISTER_OPTYPE(FASTRCNNPREDICTIONS, "FastrcnnPredictions");
GE_REGISTER_OPTYPE(SPLIT, "Split");
GE_REGISTER_OPTYPE(EXPANDDIMS, "ExpandDims");
GE_REGISTER_OPTYPE(MEAN, "Mean");
GE_REGISTER_OPTYPE(GREATER, "Greater");
GE_REGISTER_OPTYPE(SWITCH, "Switch");
GE_REGISTER_OPTYPE(REFSWITCH, "RefSwitch");
GE_REGISTER_OPTYPE(MERGE, "Merge");
GE_REGISTER_OPTYPE(REFMERGE, "RefMerge");
GE_REGISTER_OPTYPE(ENTER, "Enter");
GE_REGISTER_OPTYPE(REFENTER, "RefEnter");
GE_REGISTER_OPTYPE(LOOPCOND, "LoopCond");
GE_REGISTER_OPTYPE(NEXTITERATION, "NextIteration");
GE_REGISTER_OPTYPE(REFNEXTITERATION, "RefNextIteration");
GE_REGISTER_OPTYPE(EXIT, "Exit");
GE_REGISTER_OPTYPE(REFEXIT, "RefExit");
GE_REGISTER_OPTYPE(CONTROLTRIGGER, "ControlTrigger");
GE_REGISTER_OPTYPE(TRANSPOSE, "Transpose");
GE_REGISTER_OPTYPE(CAST, "Cast");
GE_REGISTER_OPTYPE(REGION, "Region");
GE_REGISTER_OPTYPE(YOLO, "Yolo");
GE_REGISTER_OPTYPE(YOLODETECTIONOUTPUT, "YoloDetectionOutput");
GE_REGISTER_OPTYPE(FILL, "Fill");
GE_REGISTER_OPTYPE(REVERSE, "Reverse");
GE_REGISTER_OPTYPE(UNPACK, "Unpack");
GE_REGISTER_OPTYPE(YOLO2REORG, "Yolo2Reorg");
GE_REGISTER_OPTYPE(REDUCESUM, "ReduceSum");
GE_REGISTER_OPTYPE(CONSTANT, "Const");
GE_REGISTER_OPTYPE(RESIZEBILINEAR, "ResizeBilinear");
GE_REGISTER_OPTYPE(MAXIMUM, "Maximum");
GE_REGISTER_OPTYPE(FRAMEWORKOP, "FrameworkOp");
GE_REGISTER_OPTYPE(ARG, "_Arg");
GE_REGISTER_OPTYPE(FUSEDBATCHNORMGRAD, "FusedBatchNormGrad");
GE_REGISTER_OPTYPE(LSTM, "LSTM");
GE_REGISTER_OPTYPE(HIGHWAY, "HighWay");
GE_REGISTER_OPTYPE(RNN, "RNN");
GE_REGISTER_OPTYPE(ATTENTIONDECODER, "AttentionDecoder");
GE_REGISTER_OPTYPE(LOGICAL_NOT, "LogicalNot");
GE_REGISTER_OPTYPE(LOGICAL_AND, "LogicalAnd");
GE_REGISTER_OPTYPE(EQUAL, "Equal");
GE_REGISTER_OPTYPE(INTERP, "Interp");
GE_REGISTER_OPTYPE(SHUFFLECHANNEL, "ShuffleChannel");
GE_REGISTER_OPTYPE(AIPP, "Aipp");

GE_REGISTER_OPTYPE(CROPANDRESIZE, "CropAndResize");
GE_REGISTER_OPTYPE(UNUSEDCONST, "UnusedConst");
GE_REGISTER_OPTYPE(BROADCASTGRADIENTARGS, "BroadcastGradientArgs");
GE_REGISTER_OPTYPE(BROADCASTARGS, "BroadcastArgs");
GE_REGISTER_OPTYPE(STOPGRADIENT, "StopGradient");
GE_REGISTER_OPTYPE(PPREVENTGRADIENT, "PreventGradient");
GE_REGISTER_OPTYPE(GUARANTEECONST, "GuaranteeConst");
GE_REGISTER_OPTYPE(SPARSETODENSE, "SparseToDense");
GE_REGISTER_OPTYPE(NONMAXSUPPRESSION, "NonMaxSuppression");
GE_REGISTER_OPTYPE(TOPKV2, "TopKV2");
GE_REGISTER_OPTYPE(INVERTPERMUTATION, "InvertPermutation");
GE_REGISTER_OPTYPE(MULTINOMIAL, "Multinomial");
GE_REGISTER_OPTYPE(REVERSESEQUENCE, "ReverseSequence");
GE_REGISTER_OPTYPE(GETNEXT, "GetNext");
GE_REGISTER_OPTYPE(INITDATA, "InitData");

// ANN specific operator
GE_REGISTER_OPTYPE(ANN_MEAN, "AnnMean");
GE_REGISTER_OPTYPE(ANN_CONVOLUTION, "AnnConvolution");
GE_REGISTER_OPTYPE(ANN_DEPCONVOLUTION, "AnnDepthConv");
GE_REGISTER_OPTYPE(DIV, "Div");
GE_REGISTER_OPTYPE(ANN_FULLCONNECTION, "AnnFullConnection");
GE_REGISTER_OPTYPE(ANN_NETOUTPUT, "AnnNetOutput");
GE_REGISTER_OPTYPE(ANN_DATA, "AnnData");

// Training operator
GE_REGISTER_OPTYPE(CONVGRADFILTER, "Conv2DBackpropFilter");
GE_REGISTER_OPTYPE(CONV2D, "Conv2D");
GE_REGISTER_OPTYPE(CONV2DBACKPROPINPUT, "Conv2DBackpropInput");
GE_REGISTER_OPTYPE(ACTIVATIONGRAD, "ReluGrad");
GE_REGISTER_OPTYPE(CONSTANTOP, "Constant");
GE_REGISTER_OPTYPE(AVGPOOLGRAD, "AvgPoolGrad");
GE_REGISTER_OPTYPE(SQUARE, "Square");
GE_REGISTER_OPTYPE(PLACEHOLDER, "PlaceHolder");
GE_REGISTER_OPTYPE(END, "End");
GE_REGISTER_OPTYPE(VARIABLE, "Variable");

/// @ingroup domi_omg
/// @brief INPUT node type
static const char* const kInputType = "Input";

///
/// @ingroup domi_omg
/// @brief AIPP tag, tag for aipp conv operator
///
static const char* const kAippConvFlag = "Aipp_Conv_Flag";

///
/// @ingroup domi_omg
/// @brief AIPP tag, tag for aipp data operator
///
static const char* const kAippDataFlag = "Aipp_Data_Flag";

///
/// @ingroup domi_omg
/// @brief AIPP tag, tag for aipp data operator
///
static const char* const kAippDataType = "AippData";

///
/// @ingroup domi_omg
/// @brief DATA node type
///
static const char* const kDataType = "Data";

///
/// @ingroup domi_omg
/// @brief Frame operator type
///
static const char* const kFrameworkOpType = "FrameworkOp";

///
/// @ingroup domi_omg
/// @brief Data node type
///
static const char* const kAnnDataType = "AnnData";
static const char* const kAnnNetoutputType = "AnnNetOutput";
///
/// @ingroup domi_omg
/// @brief Convolution node type
///
static const char* const kNodeNameNetOutput = "Node_Output";

///
/// @ingroup domi_omg
/// @brief RECV node type
///
static const char* const kRecvType = "Recv";

///
/// @ingroup domi_omg
/// @brief SEND node type
///
static const char* const kSendType = "Send";

///
/// @ingroup domi_omg
/// @brief Convolution node type
///
static const char* const kOpTypeConvolution = "Convolution";
///
/// @ingroup domi_omg
/// @brief Add convolution node name to hard AIPP
///
static const char* const kAippConvOpNmae = "aipp_conv_op";
///
/// @ingroup domi_omg
/// @brief Operator configuration item separator
///
static const char* const kOpConfDelimiter = ":";
};      // namespace ge
#endif  // COMMON_GRAPH_DEBUG_GE_OP_TYPES_H_
