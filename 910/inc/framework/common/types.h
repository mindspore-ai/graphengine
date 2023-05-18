/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef INC_FRAMEWORK_COMMON_TYPES_H_
#define INC_FRAMEWORK_COMMON_TYPES_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "framework/common/fmk_error_codes.h"
#include "framework/common/fmk_types.h"
#include "framework/common/op_types.h"
#include "register/register_types.h"

namespace ge {
// dump
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string DUMP_MODEL;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string DUMP_ALL_MODEL;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string DUMP_LAYER_OP_MODEL;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string DUMP_STATUS;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string DUMP_LAYER;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string DUMP_FILE_PATH;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string DUMP_MODE;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string OP_DEBUG_AICORE;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string OP_DEBUG_ATOMIC;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string OP_DEBUG_ALL;

// Profile-related constants
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string PROFILE_MODEL_ID;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string MODEL_ATTR_TASKS;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string MODEL_ATTR_TASK_GEN_BASE_ADDR;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string MODEL_ATTR_HOST_MEMORY_SIZE;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string MODEL_ATTR_HOST_SVM_SIZE;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string MODEL_ATTR_TASK_GEN_WEIGHT_ADDR;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string MODEL_ATTR_FUSION_MODEL_DEF;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint64_t ALLOC_MEMORY_MAX_SIZE;  // Max size of 8 GB.

REGISTER_OPTYPE_DECLARE(DATA, "Data");
REGISTER_OPTYPE_DECLARE(REFDATA, "RefData");
REGISTER_OPTYPE_DECLARE(AIPPDATA, "AippData");
REGISTER_OPTYPE_DECLARE(QUEUE_DATA, "QueueData");
REGISTER_OPTYPE_DECLARE(CONVOLUTION, "Convolution");
REGISTER_OPTYPE_DECLARE(CORRELATION, "Correlation");
REGISTER_OPTYPE_DECLARE(CORRELATIONV2, "Correlation_V2");
REGISTER_OPTYPE_DECLARE(DECONVOLUTION, "Deconvolution");
REGISTER_OPTYPE_DECLARE(POOLING, "Pooling");
REGISTER_OPTYPE_DECLARE(ELTWISE, "Eltwise");
REGISTER_OPTYPE_DECLARE(RELU, "ReLU");
REGISTER_OPTYPE_DECLARE(RELU6, "ReLU6");
REGISTER_OPTYPE_DECLARE(SIGMOID, "Sigmoid");
REGISTER_OPTYPE_DECLARE(ABSVAL, "AbsVal");
REGISTER_OPTYPE_DECLARE(TANH, "TanH");
REGISTER_OPTYPE_DECLARE(PRELU, "PReLU");
REGISTER_OPTYPE_DECLARE(BATCHNORM, "BatchNorm");
REGISTER_OPTYPE_DECLARE(FUSIONBATCHNORM, "FusionBatchNorm");
REGISTER_OPTYPE_DECLARE(SCALE, "Scale");
REGISTER_OPTYPE_DECLARE(FULL_CONNECTION, "FullConnection");
REGISTER_OPTYPE_DECLARE(SOFTMAX, "Softmax");
REGISTER_OPTYPE_DECLARE(PLUS, "Plus");
REGISTER_OPTYPE_DECLARE(ACTIVATION, "Activation");
REGISTER_OPTYPE_DECLARE(FLATTEN, "Flatten");
REGISTER_OPTYPE_DECLARE(ADD, "Add");
REGISTER_OPTYPE_DECLARE(SUB, "Sub");
REGISTER_OPTYPE_DECLARE(MUL, "Mul");
REGISTER_OPTYPE_DECLARE(MATMUL, "MatMul");
REGISTER_OPTYPE_DECLARE(RSQRT, "Rsqrt");
REGISTER_OPTYPE_DECLARE(BIASADD, "BiasAdd");
REGISTER_OPTYPE_DECLARE(RESHAPE, "Reshape");
REGISTER_OPTYPE_DECLARE(REFORMAT, "ReFormat");
REGISTER_OPTYPE_DECLARE(DEPCONVOLUTION, "ConvolutionDepthwise");
REGISTER_OPTYPE_DECLARE(DROPOUT, "Dropout");
REGISTER_OPTYPE_DECLARE(DROPOUTDOMASK, "DropOutDoMask");
REGISTER_OPTYPE_DECLARE(DROPOUTDOMASKV3, "DropOutDoMaskV3");
REGISTER_OPTYPE_DECLARE(DROPOUTDOMASKV3D, "DropOutDoMaskV3D");
REGISTER_OPTYPE_DECLARE(SOFTMAXV2WITHDROPOUTDOMASKV3D, "SoftmaxV2WithDropOutDoMaskV3D");
REGISTER_OPTYPE_DECLARE(ATTENTIONSCORE, "AttentionScore");
REGISTER_OPTYPE_DECLARE(ATTENTIONSCOREGRAD, "AttentionScoreGrad");
REGISTER_OPTYPE_DECLARE(DROPOUTGENMASK, "DropOutGenMask");
REGISTER_OPTYPE_DECLARE(AXPYWITHSOFTMAXANDDROPOUTDOMASK, "AxpyWithSoftmaxAndDropOutDoMask");
REGISTER_OPTYPE_DECLARE(CONCAT, "Concat");
REGISTER_OPTYPE_DECLARE(ROIPOOLING, "ROIPooling");
REGISTER_OPTYPE_DECLARE(PROPOSAL, "Proposal");
REGISTER_OPTYPE_DECLARE(FSRDETECTIONOUTPUT, "FSRDetectionOutput");
REGISTER_OPTYPE_DECLARE(DETECTIONPOSTPROCESS, "Detectpostprocess");
REGISTER_OPTYPE_DECLARE(LRN, "LRN");
REGISTER_OPTYPE_DECLARE(TRANSDATA, "TransData");
REGISTER_OPTYPE_DECLARE(PERMUTE, "Permute");
REGISTER_OPTYPE_DECLARE(SSDNORMALIZE, "SSDNormalize");
REGISTER_OPTYPE_DECLARE(SSDPRIORBOX, "SSDPriorBox");
REGISTER_OPTYPE_DECLARE(NETOUTPUT, "NetOutput");
REGISTER_OPTYPE_DECLARE(SSDDETECTIONOUTPUT, "SSDDetectionOutput");
REGISTER_OPTYPE_DECLARE(REFINEDETDETECTIONOUTPUT, "RefinedetDetectionOutput");
REGISTER_OPTYPE_DECLARE(CHANNELAXPY, "ChannelAxpy");
REGISTER_OPTYPE_DECLARE(PSROIPOOLING, "PSROIPooling");
REGISTER_OPTYPE_DECLARE(POWER, "Power");
REGISTER_OPTYPE_DECLARE(POW, "Pow");
REGISTER_OPTYPE_DECLARE(ROIALIGN, "ROIAlign");
REGISTER_OPTYPE_DECLARE(PYTHON, "Python");
REGISTER_OPTYPE_DECLARE(FREESPACEEXTRACT, "FreespaceExtract");
REGISTER_OPTYPE_DECLARE(SPATIALTF, "SpatialTransform");
REGISTER_OPTYPE_DECLARE(SHAPE, "Shape");
REGISTER_OPTYPE_DECLARE(SHAPEN, "ShapeN");
REGISTER_OPTYPE_DECLARE(ARGMAX, "ArgMax");
REGISTER_OPTYPE_DECLARE(GATHERND, "GatherNd");
REGISTER_OPTYPE_DECLARE(GATHER, "Gather");
REGISTER_OPTYPE_DECLARE(REALDIV, "RealDiv");
REGISTER_OPTYPE_DECLARE(PACK, "Pack");
REGISTER_OPTYPE_DECLARE(SLICE, "Slice");
REGISTER_OPTYPE_DECLARE(SLICED, "SliceD");
REGISTER_OPTYPE_DECLARE(FLOORDIV, "FloorDiv");
REGISTER_OPTYPE_DECLARE(SQUEEZE, "Squeeze");
REGISTER_OPTYPE_DECLARE(UNSQUEEZE, "Unsqueeze");
REGISTER_OPTYPE_DECLARE(SQUEEZEV2, "SqueezeV2");
REGISTER_OPTYPE_DECLARE(UNSQUEEZEV2, "UnsqueezeV2");
REGISTER_OPTYPE_DECLARE(SQUEEZEV3, "SqueezeV3");
REGISTER_OPTYPE_DECLARE(UNSQUEEZEV3, "UnsqueezeV3");
REGISTER_OPTYPE_DECLARE(STRIDEDSLICE, "StridedSlice");
REGISTER_OPTYPE_DECLARE(RANGE, "Range");
REGISTER_OPTYPE_DECLARE(RPNPROPOSALS, "GenerateRpnProposals");
REGISTER_OPTYPE_DECLARE(DECODEBBOX, "DecodeBBox");
REGISTER_OPTYPE_DECLARE(PAD, "Pad");
REGISTER_OPTYPE_DECLARE(PADV2, "PadV2");
REGISTER_OPTYPE_DECLARE(MIRRORPAD, "MirrorPad");
REGISTER_OPTYPE_DECLARE(TILE, "Tile");
REGISTER_OPTYPE_DECLARE(SIZE, "Size");
REGISTER_OPTYPE_DECLARE(CLIPBOXES, "Clipboxes");
REGISTER_OPTYPE_DECLARE(FASTRCNNPREDICTIONS, "FastrcnnPredictions");
REGISTER_OPTYPE_DECLARE(SPLIT, "Split");
REGISTER_OPTYPE_DECLARE(SPLITV, "SplitV");
REGISTER_OPTYPE_DECLARE(EXPANDDIMS, "ExpandDims");
REGISTER_OPTYPE_DECLARE(EMPTY, "Empty");
REGISTER_OPTYPE_DECLARE(MEAN, "Mean");
REGISTER_OPTYPE_DECLARE(GREATER, "Greater");
REGISTER_OPTYPE_DECLARE(SWITCH, "Switch");
REGISTER_OPTYPE_DECLARE(SWITCHN, "SwitchN");
REGISTER_OPTYPE_DECLARE(REFSWITCH, "RefSwitch");
REGISTER_OPTYPE_DECLARE(MERGE, "Merge");
REGISTER_OPTYPE_DECLARE(REFMERGE, "RefMerge");
REGISTER_OPTYPE_DECLARE(ENTER, "Enter");
REGISTER_OPTYPE_DECLARE(REFENTER, "RefEnter");
REGISTER_OPTYPE_DECLARE(LOOPCOND, "LoopCond");
REGISTER_OPTYPE_DECLARE(NEXTITERATION, "NextIteration");
REGISTER_OPTYPE_DECLARE(REFNEXTITERATION, "RefNextIteration");
REGISTER_OPTYPE_DECLARE(EXIT, "Exit");
REGISTER_OPTYPE_DECLARE(REFEXIT, "RefExit");
REGISTER_OPTYPE_DECLARE(CONTROLTRIGGER, "ControlTrigger");
REGISTER_OPTYPE_DECLARE(SYMBOLICGRADIENT, "SymbolicGradient");
REGISTER_OPTYPE_DECLARE(REMOTECALL, "RemoteCall");
REGISTER_OPTYPE_DECLARE(_IF, "_If");
REGISTER_OPTYPE_DECLARE(STATELESSIF, "StatelessIf");
REGISTER_OPTYPE_DECLARE(IF, "If");
REGISTER_OPTYPE_DECLARE(CASE, "Case");
REGISTER_OPTYPE_DECLARE(STATELESSCASE, "StatelessCase");
REGISTER_OPTYPE_DECLARE(_WHILE, "_While");
REGISTER_OPTYPE_DECLARE(WHILE, "While");
REGISTER_OPTYPE_DECLARE(STATELESSWHILE, "StatelessWhile");
REGISTER_OPTYPE_DECLARE(FOR, "For");
REGISTER_OPTYPE_DECLARE(PARTITIONEDCALL, "PartitionedCall");
REGISTER_OPTYPE_DECLARE(STATEFULPARTITIONEDCALL, "StatefulPartitionedCall");
REGISTER_OPTYPE_DECLARE(FAKEPARAM, "FakeParam");
REGISTER_OPTYPE_DECLARE(TRANSPOSE, "Transpose");
REGISTER_OPTYPE_DECLARE(TRANSPOSED, "TransposeD");
REGISTER_OPTYPE_DECLARE(CAST, "Cast");
REGISTER_OPTYPE_DECLARE(REGION, "Region");
REGISTER_OPTYPE_DECLARE(YOLO, "Yolo");
REGISTER_OPTYPE_DECLARE(YOLODETECTIONOUTPUT, "YoloDetectionOutput");
REGISTER_OPTYPE_DECLARE(FILL, "Fill");
REGISTER_OPTYPE_DECLARE(RANK, "Rank");
REGISTER_OPTYPE_DECLARE(REVERSE, "Reverse");
REGISTER_OPTYPE_DECLARE(UNPACK, "Unpack");
REGISTER_OPTYPE_DECLARE(YOLO2REORG, "Yolo2Reorg");
REGISTER_OPTYPE_DECLARE(REDUCESUM, "ReduceSum");
REGISTER_OPTYPE_DECLARE(SUM, "Sum");
REGISTER_OPTYPE_DECLARE(CONSTANT, "Const");
REGISTER_OPTYPE_DECLARE(RESIZEBILINEAR, "ResizeBilinear");
REGISTER_OPTYPE_DECLARE(RESIZEBILINEARGRAD, "ResizeBilinearGrad");
REGISTER_OPTYPE_DECLARE(MAXIMUM, "Maximum");
REGISTER_OPTYPE_DECLARE(FRAMEWORKOP, "FrameworkOp");
REGISTER_OPTYPE_DECLARE(ARG, "_Arg");
REGISTER_OPTYPE_DECLARE(FUSEDBATCHNORMGRAD, "FusedBatchNormGrad");
REGISTER_OPTYPE_DECLARE(LSTM, "LSTM");
REGISTER_OPTYPE_DECLARE(HIGHWAY, "HighWay");
REGISTER_OPTYPE_DECLARE(RNN, "RNN");
REGISTER_OPTYPE_DECLARE(ATTENTIONDECODER, "AttentionDecoder");
REGISTER_OPTYPE_DECLARE(LOGICAL_NOT, "LogicalNot");
REGISTER_OPTYPE_DECLARE(LOGICAL_AND, "LogicalAnd");
REGISTER_OPTYPE_DECLARE(LOGICAL_OR, "LogicalOr");
REGISTER_OPTYPE_DECLARE(EQUAL, "Equal");
REGISTER_OPTYPE_DECLARE(NOTEQUAL, "NotEqual");
REGISTER_OPTYPE_DECLARE(INTERP, "Interp");
REGISTER_OPTYPE_DECLARE(SHUFFLECHANNEL, "ShuffleChannel");
REGISTER_OPTYPE_DECLARE(AIPP, "Aipp");
REGISTER_OPTYPE_DECLARE(MULTISHAPE, "MultiShape");
REGISTER_OPTYPE_DECLARE(RECIPROCAL, "Reciprocal");
REGISTER_OPTYPE_DECLARE(SELU, "Selu");
REGISTER_OPTYPE_DECLARE(ELU, "Elu");
REGISTER_OPTYPE_DECLARE(ACOSH, "Acosh");
REGISTER_OPTYPE_DECLARE(ASINH, "Asinh");
REGISTER_OPTYPE_DECLARE(MINIMUM, "Minimum");
REGISTER_OPTYPE_DECLARE(CLIP, "Clip");
REGISTER_OPTYPE_DECLARE(L2NORMALIZE, "L2Normalize");
REGISTER_OPTYPE_DECLARE(CROPANDRESIZE, "CropAndResize");
REGISTER_OPTYPE_DECLARE(UNUSEDCONST, "UnusedConst");
REGISTER_OPTYPE_DECLARE(SPARSETODENSE, "SparseToDense");
REGISTER_OPTYPE_DECLARE(NONMAXSUPPRESSION, "NonMaxSuppression");
REGISTER_OPTYPE_DECLARE(TOPKV2, "TopKV2");
REGISTER_OPTYPE_DECLARE(INVERTPERMUTATION, "InvertPermutation");
REGISTER_OPTYPE_DECLARE(MULTINOMIAL, "Multinomial");
REGISTER_OPTYPE_DECLARE(REVERSESEQUENCE, "ReverseSequence");
REGISTER_OPTYPE_DECLARE(REDUCEPROD, "ReduceProd");
REGISTER_OPTYPE_DECLARE(REDUCEMAX, "ReduceMax");
REGISTER_OPTYPE_DECLARE(REDUCEMIN, "ReduceMin");
REGISTER_OPTYPE_DECLARE(EXTRACTIMAGEPATCHES, "ExtractImagePatches");
REGISTER_OPTYPE_DECLARE(SQRT, "Sqrt");
REGISTER_OPTYPE_DECLARE(REDUCEALL, "ReduceAll");
REGISTER_OPTYPE_DECLARE(RESIZENEARESTNEIGHBOR, "ResizeNearestNeighbor");
REGISTER_OPTYPE_DECLARE(SPACETOBATCHND, "SpaceToBatchND");
REGISTER_OPTYPE_DECLARE(BATCHTOSPACEND, "BatchToSpaceND");
REGISTER_OPTYPE_DECLARE(ASSERT, "Assert");
REGISTER_OPTYPE_DECLARE(GREATEREQUAL, "GreaterEqual");
REGISTER_OPTYPE_DECLARE(FLOOR, "Floor");
REGISTER_OPTYPE_DECLARE(RANDOMUNIFORM, "RandomUniform");
REGISTER_OPTYPE_DECLARE(BATCHMATMUL, "BatchMatMul");
REGISTER_OPTYPE_DECLARE(LESSEQUAL, "LessEqual");
REGISTER_OPTYPE_DECLARE(ONEHOT, "OneHot");
REGISTER_OPTYPE_DECLARE(LAYERNORM, "LayerNorm");
REGISTER_OPTYPE_DECLARE(SPACETODEPTH, "SpaceToDepth");
REGISTER_OPTYPE_DECLARE(DEPTHTOSPACE, "DepthToSpace");
REGISTER_OPTYPE_DECLARE(RINT, "Rint");
REGISTER_OPTYPE_DECLARE(ATAN, "Atan");
REGISTER_OPTYPE_DECLARE(ATAN2, "Atan2");
REGISTER_OPTYPE_DECLARE(ATANH, "Atanh");
REGISTER_OPTYPE_DECLARE(ACOS, "Acos");
REGISTER_OPTYPE_DECLARE(ASIN, "Asin");
REGISTER_OPTYPE_DECLARE(NEG, "Neg");
REGISTER_OPTYPE_DECLARE(LOG, "Log");
REGISTER_OPTYPE_DECLARE(TAN, "Tan");
REGISTER_OPTYPE_DECLARE(ROUND, "Round");
REGISTER_OPTYPE_DECLARE(UPSAMPLE, "Upsample");
REGISTER_OPTYPE_DECLARE(FLOORMOD, "FloorMod");
REGISTER_OPTYPE_DECLARE(LESS, "Less");
REGISTER_OPTYPE_DECLARE(ZEROSLIKE, "ZerosLike");
REGISTER_OPTYPE_DECLARE(EXP, "Exp");
REGISTER_OPTYPE_DECLARE(WHERE, "Where");
REGISTER_OPTYPE_DECLARE(FAKEQUANTWITHMINMAXVARS, "FakeQuantWithMinMaxVars");
REGISTER_OPTYPE_DECLARE(SOFTPLUS, "Softplus");
REGISTER_OPTYPE_DECLARE(SOFTSIGN, "Softsign");
REGISTER_OPTYPE_DECLARE(COSH, "Cosh");
REGISTER_OPTYPE_DECLARE(SINH, "Sinh");
REGISTER_OPTYPE_DECLARE(RETINAMULTIANCHORS, "RetinaMultiAnchor");
REGISTER_OPTYPE_DECLARE(SQUAREDDIFFERENCE, "SquaredDifference");
REGISTER_OPTYPE_DECLARE(REQUIREDSPACETOBATCHPADDINGS, "RequiredSpaceToBatchPaddings");  // for retinanet scope fusion
REGISTER_OPTYPE_DECLARE(SSDPOSTPROCESSOR, "SSDPostProcessor");
REGISTER_OPTYPE_DECLARE(SSDANCHORGENERATOR, "SSDAnchorGenerator");
REGISTER_OPTYPE_DECLARE(RETINANETBOXES, "RetinanetBoxes");
REGISTER_OPTYPE_DECLARE(RETINANETCLIPPEDBOXES, "RetinanetClippedBoxes");
REGISTER_OPTYPE_DECLARE(RETINANETFILTEREDDETECTIONS, "RetinanetFilteredDetections");
REGISTER_OPTYPE_DECLARE(RETINANETPOSTPROCESSOR, "RetinanetPostProcessor");
REGISTER_OPTYPE_DECLARE(RETINANETANCHORS, "RetinanetAnchors");
REGISTER_OPTYPE_DECLARE(FASTERRCNNMAP, "FasterRCNNMap");
REGISTER_OPTYPE_DECLARE(FASTERRCNNMAP1, "FasterRCNNMap1");
REGISTER_OPTYPE_DECLARE(FASTERRCNNSECONDSTAGEPOSTPROCESSOR, "FasterRCNNSecondStagePostprocessor");
REGISTER_OPTYPE_DECLARE(FASTERRCNNROIINTERPOOLING, "FasterRCNNROIInterPooling");
REGISTER_OPTYPE_DECLARE(FASTERRCNNFIRSTSTAGEPOSTPROCESSOR, "FasterRCNNFirstStagePostprocessor");
REGISTER_OPTYPE_DECLARE(FASTERRCNNGRIDANCHORGENERATOR, "FasterRCNNGridAnchorGenerator");
REGISTER_OPTYPE_DECLARE(ROIINTERPOOLING, "ROIInterPooling");
REGISTER_OPTYPE_DECLARE(FASTERRCNNCLIPTOWINDOW, "FasterRCNNClipToWindow");
REGISTER_OPTYPE_DECLARE(EMBEDLOOKUP, "EmbedLookup");
REGISTER_OPTYPE_DECLARE(HASHLOOKUP, "HashLookup");
REGISTER_OPTYPE_DECLARE(LSH_PROJ, "LshProject");
REGISTER_OPTYPE_DECLARE(SVDF, "SVDF");
REGISTER_OPTYPE_DECLARE(IDENTITY, "Identity");
REGISTER_OPTYPE_DECLARE(PLACEHOLDERWITHDEFAULT, "PlaceholderWithDefault");
REGISTER_OPTYPE_DECLARE(IDENTITYN, "IdentityN");
REGISTER_OPTYPE_DECLARE(GETSPAN, "GetSpan");
REGISTER_OPTYPE_DECLARE(STOPGRADIENT, "StopGradient");
REGISTER_OPTYPE_DECLARE(PREVENTGRADIENT, "PreventGradient");
REGISTER_OPTYPE_DECLARE(GUARANTEECONST, "GuaranteeConst");
REGISTER_OPTYPE_DECLARE(BROADCASTGRADIENTARGS, "BroadcastGradientArgs");
REGISTER_OPTYPE_DECLARE(BROADCASTARGS, "BroadcastArgs");
REGISTER_OPTYPE_DECLARE(CONCATV2, "ConcatV2");
REGISTER_OPTYPE_DECLARE(CONCATOFFSET, "ConcatOffset");
REGISTER_OPTYPE_DECLARE(LESSEQUAL, "LessEqual");
REGISTER_OPTYPE_DECLARE(SELECT, "Select");
REGISTER_OPTYPE_DECLARE(CONFUSIONMATRIX, "ConfusionMatrix");
REGISTER_OPTYPE_DECLARE(PLACEHOLDER, "PlaceHolder");
REGISTER_OPTYPE_DECLARE(END, "End");
REGISTER_OPTYPE_DECLARE(BASICLSTMCELL, "BasicLSTMCell");
REGISTER_OPTYPE_DECLARE(GETNEXT, "GetNext");
REGISTER_OPTYPE_DECLARE(ITERATOR, "Iterator");
REGISTER_OPTYPE_DECLARE(ITERATORV2, "IteratorV2");
REGISTER_OPTYPE_DECLARE(INITDATA, "InitData");
REGISTER_OPTYPE_DECLARE(TRANSSHAPE, "TransShape");
REGISTER_OPTYPE_DECLARE(REFIDENTITY, "RefIdentity");
REGISTER_OPTYPE_DECLARE(BITCAST, "Bitcast");
REGISTER_OPTYPE_DECLARE(GATHERSHAPES, "GatherShapes");
REGISTER_OPTYPE_DECLARE(FLATTENV2, "FlattenV2");
REGISTER_OPTYPE_DECLARE(FILECONSTANT, "FileConstant");

// ANN dedicated operator
REGISTER_OPTYPE_DECLARE(ANN_MEAN, "AnnMean");
REGISTER_OPTYPE_DECLARE(ANN_CONVOLUTION, "AnnConvolution");
REGISTER_OPTYPE_DECLARE(ANN_DEPCONVOLUTION, "AnnDepthConv");
REGISTER_OPTYPE_DECLARE(ANN_FULLCONNECTION, "AnnFullConnection");
REGISTER_OPTYPE_DECLARE(ANN_NETOUTPUT, "AnnNetOutput");
REGISTER_OPTYPE_DECLARE(ANN_DATA, "AnnData");
REGISTER_OPTYPE_DECLARE(ANN_RESHAPE, "AnnReshape");
REGISTER_OPTYPE_DECLARE(ANN_ADD, "AnnAdd");
REGISTER_OPTYPE_DECLARE(ANN_MUL, "AnnMul");
REGISTER_OPTYPE_DECLARE(ANN_SUB, "AnnSub");
REGISTER_OPTYPE_DECLARE(ANN_DIV, "AnnDiv");
REGISTER_OPTYPE_DECLARE(ANN_DEQUANTIZE, "AnnDequant");
REGISTER_OPTYPE_DECLARE(ANN_QUANTIZE, "AnnQuant");
REGISTER_OPTYPE_DECLARE(ANN_PAD, "AnnPad");
REGISTER_OPTYPE_DECLARE(ANN_RESIZE_BILINEAR, "AnnResizeBilinear");

// Training operator
REGISTER_OPTYPE_DECLARE(GATHERV2, "GatherV2");
REGISTER_OPTYPE_DECLARE(CONVGRADFILTER, "Conv2DBackpropFilter");
REGISTER_OPTYPE_DECLARE(CONV2D, "Conv2D");
REGISTER_OPTYPE_DECLARE(CONV2DBACKPROPINPUT, "Conv2DBackpropInput");
REGISTER_OPTYPE_DECLARE(FUSEDBATCHNORM, "FusedBatchNorm");
REGISTER_OPTYPE_DECLARE(BIASADDGRAD, "BiasAddGrad");
REGISTER_OPTYPE_DECLARE(ACTIVATIONGRAD, "ReluGrad");
REGISTER_OPTYPE_DECLARE(MAXPOOLWITHARGMAX, "MaxPoolWithArgmax");
REGISTER_OPTYPE_DECLARE(MAXPOOLGRADWITHARGMAX, "MaxPoolGradWithArgmax");
REGISTER_OPTYPE_DECLARE(SPARSESOFTMAXCROSSENTROPYWITHLOGITS, "SparseSoftmaxCrossEntropyWithLogits");
REGISTER_OPTYPE_DECLARE(SNAPSHOT, "Snapshot");
REGISTER_OPTYPE_DECLARE(LAYERNORM, "LayerNorm");
REGISTER_OPTYPE_DECLARE(HUBERLOSSGRAD, "HuberLossGrad");
REGISTER_OPTYPE_DECLARE(HUBERLOSS, "HuberLoss");
REGISTER_OPTYPE_DECLARE(NEGATIVE, "Negative");
REGISTER_OPTYPE_DECLARE(SSDCAST, "SSDCast");
REGISTER_OPTYPE_DECLARE(SSDSQUEEZEFUSION, "SsdSqueezeFusion");
REGISTER_OPTYPE_DECLARE(SPARSESOFTMAXCROSSENTROPY, "SsdSparseSoftmaxCrossEntropy");
REGISTER_OPTYPE_DECLARE(SPARSESOFTMAXCROSSENTROPYGRAD, "SsdSparseSoftmaxCrossEntropyGrad");
REGISTER_OPTYPE_DECLARE(CONCATFIVE2FOUR, "ConcatFive2Four");
REGISTER_OPTYPE_DECLARE(CONCATFOUR2FIVE, "ConcatFour2Five");
REGISTER_OPTYPE_DECLARE(SSDREALDIVTILEMUL, "SSDRealdivTileMul");
REGISTER_OPTYPE_DECLARE(SSDSUMMULREALDIVMEAN, "SSDSumMulRealdivMean");

REGISTER_OPTYPE_DECLARE(MEANGRAD, "MeanGrad");
REGISTER_OPTYPE_DECLARE(TRANSLATE, "Translate");
REGISTER_OPTYPE_DECLARE(ADDN, "AddN");
REGISTER_OPTYPE_DECLARE(L2LOSS, "L2Loss");
REGISTER_OPTYPE_DECLARE(MULTIPLY, "Multiply");
REGISTER_OPTYPE_DECLARE(RELU6GRAD, "Relu6Grad");
REGISTER_OPTYPE_DECLARE(AVGPOOLGRAD, "AvgPoolGrad");
REGISTER_OPTYPE_DECLARE(DEPTHWISECONV2DBACKPROPFILTER, "DepthwiseConv2dNativeBackpropFilter");
REGISTER_OPTYPE_DECLARE(DEPTHWISECONV2DBACKPORPINPUT, "DepthwiseConv2dNativeBackpropInput");
REGISTER_OPTYPE_DECLARE(DEPTHWISECONV2DFORWARDNATIVE, "DepthwiseConv2dNative");
REGISTER_OPTYPE_DECLARE(DROPOUTGRAD, "DropOutGrad");
REGISTER_OPTYPE_DECLARE(APPLYRMSPROPMIXEDPRECISION, "apply_rms_prop_mixed_precision");
REGISTER_OPTYPE_DECLARE(APPLYRMSPROP, "ApplyRMSProp");
REGISTER_OPTYPE_DECLARE(LARS, "Lars");
REGISTER_OPTYPE_DECLARE(DYNAMICSTITCH, "DynamicStitch");

// Variable sink related
REGISTER_OPTYPE_DECLARE(VARIABLEV2, "VariableV2");
REGISTER_OPTYPE_DECLARE(VARHANDLEOP, "VarHandleOp");
REGISTER_OPTYPE_DECLARE(TEMPORARYVARIABLE, "TemporaryVariable");
REGISTER_OPTYPE_DECLARE(DESTROYTEMPORARYVARIABLE, "DestroyTemporaryVariable");
REGISTER_OPTYPE_DECLARE(VARIABLE, "Variable");

REGISTER_OPTYPE_DECLARE(READVARIABLEOP, "ReadVariableOp");

REGISTER_OPTYPE_DECLARE(VARISINITIALIZEDOP, "VarIsInitializedOp");
REGISTER_OPTYPE_DECLARE(ISVARIABLEINITIALIZED, "IsVariableInitialized");

REGISTER_OPTYPE_DECLARE(ASSIGN, "Assign");
REGISTER_OPTYPE_DECLARE(ASSIGNVARIABLEOP, "AssignVariableOp");

REGISTER_OPTYPE_DECLARE(ASSIGNADD, "AssignAdd");
REGISTER_OPTYPE_DECLARE(ASSIGNADDVARIABLEOP, "AssignAddVariableOp");

REGISTER_OPTYPE_DECLARE(ASSIGNSUB, "AssignSub");
REGISTER_OPTYPE_DECLARE(ASSIGNSUBVARIABLEOP, "AssignSubVariableOp");

REGISTER_OPTYPE_DECLARE(APPLYMOMENTUM, "ApplyMomentum");
REGISTER_OPTYPE_DECLARE(RESOURCEAPPLYMOMENTUM, "ResourceApplyMomentum");
REGISTER_OPTYPE_DECLARE(SGD, "SGD");
REGISTER_OPTYPE_DECLARE(NOOP, "NoOp");
REGISTER_OPTYPE_DECLARE(LAYERNORMGRAD, "LayerNormGrad");

REGISTER_OPTYPE_DECLARE(SQUARE, "Square");
REGISTER_OPTYPE_DECLARE(HCOMBROADCAST, "HcomBroadcast");
REGISTER_OPTYPE_DECLARE(HCOMALLGATHER, "HcomAllGather");
REGISTER_OPTYPE_DECLARE(HCOMALLREDUCE, "HcomAllReduce");
REGISTER_OPTYPE_DECLARE(HCOMREDUCESCATTER, "HcomReduceScatter");
REGISTER_OPTYPE_DECLARE(HCOMREDUCE, "HcomReduce");
REGISTER_OPTYPE_DECLARE(HCOMSEND, "HcomSend");
REGISTER_OPTYPE_DECLARE(HCOMRECEIVE, "HcomReceive");
REGISTER_OPTYPE_DECLARE(HCOMREMOTEREAD, "HcomRemoteRead");
REGISTER_OPTYPE_DECLARE(HCOMREMOTEREFREAD, "HcomRemoteRefRead");
REGISTER_OPTYPE_DECLARE(HCOMREMOTEWRITE, "HcomRemoteWrite");
REGISTER_OPTYPE_DECLARE(HCOMREMOTESCATTERWRITE, "HcomRemoteScatterWrite");
REGISTER_OPTYPE_DECLARE(HCOMALLTOALLV, "HcomAllToAllV");
REGISTER_OPTYPE_DECLARE(HCOMGATHERALLTOALLV, "HcomGatherAllToAllV");
REGISTER_OPTYPE_DECLARE(HCOMALLTOALLVC, "HcomAllToAllVC");
REGISTER_OPTYPE_DECLARE(HCOMALLTOALL, "HcomAllToAll");
REGISTER_OPTYPE_DECLARE(HCOMREMOTELOOKUP, "HcomRemoteLookup");
REGISTER_OPTYPE_DECLARE(HCOMCOLLREMOTELOOKUP, "HcomCollRemoteLookup");
REGISTER_OPTYPE_DECLARE(HCOMCOLLREMOTEUPDATE, "HcomCollRemoteUpdate");
REGISTER_OPTYPE_DECLARE(HCOMGATHER, "HcomGather");

REGISTER_OPTYPE_DECLARE(VARASSIGN, "VarAssign");
REGISTER_OPTYPE_DECLARE(VARISINITIALIZEDOP, "VarIsInitializedOp");
REGISTER_OPTYPE_DECLARE(LogTimeStamp, "LogTimeStamp");
REGISTER_OPTYPE_DECLARE(PARALLELCONCATSTART, "_ParallelConcatStart");
REGISTER_OPTYPE_DECLARE(CONSTANTOP, "Constant");
REGISTER_OPTYPE_DECLARE(STREAMSWITCH, "StreamSwitch");
REGISTER_OPTYPE_DECLARE(STREAMSWITCHN, "StreamSwitchN");
REGISTER_OPTYPE_DECLARE(STREAMACTIVE, "StreamActive");
REGISTER_OPTYPE_DECLARE(MEMCPYASYNC, "MemcpyAsync");
REGISTER_OPTYPE_DECLARE(MEMCPYADDRASYNC, "MemcpyAddrAsync");
REGISTER_OPTYPE_DECLARE(STREAMMERGE, "StreamMerge");
REGISTER_OPTYPE_DECLARE(ENDGRAPH, "EndGraph");
REGISTER_OPTYPE_DECLARE(MODELEXIT, "ModelExit");
REGISTER_OPTYPE_DECLARE(SEND, "Send");
REGISTER_OPTYPE_DECLARE(SENDNOTIFY, "SendNotify");
REGISTER_OPTYPE_DECLARE(RECV, "Recv");
REGISTER_OPTYPE_DECLARE(RECVNOTIFY, "RecvNotify");
REGISTER_OPTYPE_DECLARE(ENDOFSEQUENCE, "EndOfSequence");
REGISTER_OPTYPE_DECLARE(STARTOFSEQUENCE, "StartOfSequence");
REGISTER_OPTYPE_DECLARE(NPUGETFLOATSTATUS, "NPUGetFloatStatus");
REGISTER_OPTYPE_DECLARE(NPUCLEARFLOATSTATUS, "NPUClearFloatStatus");
REGISTER_OPTYPE_DECLARE(NPUGETFLOATSTATUSV2, "NPUGetFloatStatusV2");
REGISTER_OPTYPE_DECLARE(NPUCLEARFLOATSTATUSV2, "NPUClearFloatStatusV2");

REGISTER_OPTYPE_DECLARE(LABELSET, "LabelSet");
REGISTER_OPTYPE_DECLARE(LABELGOTO, "LabelGoto");
REGISTER_OPTYPE_DECLARE(LABELGOTOEX, "LabelGotoEx");
REGISTER_OPTYPE_DECLARE(LABELSWITCH, "LabelSwitch");
REGISTER_OPTYPE_DECLARE(LABELSWITCHBYINDEX, "LabelSwitchByIndex");

REGISTER_OPTYPE_DECLARE(ATOMICADDRCLEAN, "AtomicAddrClean");
REGISTER_OPTYPE_DECLARE(MEMSET, "MemSet");

REGISTER_OPTYPE_DECLARE(ABS_GRAD, "AbsGrad");
REGISTER_OPTYPE_DECLARE(ACCUMULATE_N_V2, "AccumulateNV2");
REGISTER_OPTYPE_DECLARE(ACOS_GRAD, "AcosGrad");
REGISTER_OPTYPE_DECLARE(ACOSH_GRAD, "AcoshGrad");
REGISTER_OPTYPE_DECLARE(ANY, "Any");
REGISTER_OPTYPE_DECLARE(APPROXIMATE_EQUAL, "ApproximateEqual");
REGISTER_OPTYPE_DECLARE(ASIN_GRAD, "AsinGrad");
REGISTER_OPTYPE_DECLARE(ASINH_GRAD, "AsinhGrad");
REGISTER_OPTYPE_DECLARE(ATAN_GRAD, "AtanGrad");
REGISTER_OPTYPE_DECLARE(BROADCAST_TO, "BroadcastTo");
REGISTER_OPTYPE_DECLARE(ELU_GRAD, "EluGrad");
REGISTER_OPTYPE_DECLARE(ADD_V2, "AddV2");
REGISTER_OPTYPE_DECLARE(DATAFORMATDIMMAP, "DataFormatDimMap");
REGISTER_OPTYPE_DECLARE(DATAFORMATVECPERMUTE, "DataFormatVecPermute");
REGISTER_OPTYPE_DECLARE(DEQUANTIZE, "Dequantize");
REGISTER_OPTYPE_DECLARE(APPLYADADELTA, "ApplyAdadelta");
REGISTER_OPTYPE_DECLARE(APPLYADAGRAD, "ApplyAdagrad");
REGISTER_OPTYPE_DECLARE(APPLYADAGRADDA, "ApplyAdagradDA");
REGISTER_OPTYPE_DECLARE(APPLYADAM, "ApplyAdam");
REGISTER_OPTYPE_DECLARE(APPLYADAMAX, "ApplyAdaMax");
REGISTER_OPTYPE_DECLARE(APPLYADDSIGN, "ApplyAddSign");
REGISTER_OPTYPE_DECLARE(APPLYCENTEREDRMSPROP, "ApplyCenteredRMSProp");
REGISTER_OPTYPE_DECLARE(APPLYFTRL, "ApplyFtrl");
REGISTER_OPTYPE_DECLARE(APPLYFTRLV2, "ApplyFtrlv2");
REGISTER_OPTYPE_DECLARE(APPLYGRADIENTDESCENT, "ApplyGradientDescent");
REGISTER_OPTYPE_DECLARE(APPLYPOWERSIGN, "ApplyPowerSign");
REGISTER_OPTYPE_DECLARE(APPLYPROXIMALADAGRAD, "ApplyProximalAdagrad");
REGISTER_OPTYPE_DECLARE(APPLYPROXIMALGRADIENTDESCENT, "ApplyProximalGradientDescent");

REGISTER_OPTYPE_DECLARE(FOCAL_LOSS, "FocalLoss");
REGISTER_OPTYPE_DECLARE(FOCAL_LOSS_GRAD, "FocalLossGrad");
REGISTER_OPTYPE_DECLARE(SMOOTHL1_LOSS, "SmoothL1Loss");
REGISTER_OPTYPE_DECLARE(SMOOTHL1_LOSS_grad, "SmoothL1LossGrad");
REGISTER_OPTYPE_DECLARE(REDUCEMEAN, "ReduceMean");
REGISTER_OPTYPE_DECLARE(CONCAT_V2, "ConcatV2");
REGISTER_OPTYPE_DECLARE(ONEHOT_V2, "OneHotV2");
REGISTER_OPTYPE_DECLARE(SLICE_V2, "SliceV2");
REGISTER_OPTYPE_DECLARE(TILE_V2, "TileV2");
REGISTER_OPTYPE_DECLARE(SUM_V2, "SumV2");
// Common operator type when operators have the same name
REGISTER_OPTYPE_DECLARE(DETECTIONOUTPUT, "DetectionOutput");

// custom operator
REGISTER_OPTYPE_DECLARE(CUSTOMOP, "CustomOp");
REGISTER_OPTYPE_DECLARE(CUSTOMOP_NCHW, "CustomOpNchw");
REGISTER_OPTYPE_DECLARE(CUSTOMOP_NHWC, "CustomOpNhwc");
REGISTER_OPTYPE_DECLARE(CUSTOMOP_NC1HWC0, "CustomOpNc1hwc0");

// Depthwise 4d_2_6d,6d_2_4d
REGISTER_OPTYPE_DECLARE(DEPTHWISEWEIGHT4D26D, "depthwise_weight_4d_2_6d");
REGISTER_OPTYPE_DECLARE(DEPTHWISEWEIGHT6D24D, "depthwise_weight_6d_2_4d");

REGISTER_OPTYPE_DECLARE(SQRTGRAD, "SqrtGrad");
REGISTER_OPTYPE_DECLARE(SIGMOIDGRAD, "SigmoidGrad");

// Horovod operator
REGISTER_OPTYPE_DECLARE(HVDCALLBACKALLREDUCE, "HorovodAllreduce");
REGISTER_OPTYPE_DECLARE(HVDCALLBACKALLGATHER, "HorovodAllgather");
REGISTER_OPTYPE_DECLARE(HVDCALLBACKBROADCAST, "HorovodBroadcast");
REGISTER_OPTYPE_DECLARE(HVDWAIT, "HorovodWait");

// aicpu op for online_infer dynamic_dims
REGISTER_OPTYPE_DECLARE(GETDYNAMICDIMS, "GetDynamicDims");

// profiling training trace node
REGISTER_OPTYPE_DECLARE(PROFILINGTRAININGTRACE, "ProfilingTrainingTrace");

// Stack series
REGISTER_OPTYPE_DECLARE(STACK, "Stack");
REGISTER_OPTYPE_DECLARE(STACKPUSH, "StackPush");
REGISTER_OPTYPE_DECLARE(STACKPOP, "StackPop");
REGISTER_OPTYPE_DECLARE(STACKCLOSE, "StackClose");

// embedding service
REGISTER_OPTYPE_DECLARE(EMBEDDING_TABLE_FIND, "EmbeddingTableFind");
REGISTER_OPTYPE_DECLARE(EMBEDDING_TABLE_FIND_AND_INIT, "EmbeddingTableFindAndInit");
REGISTER_OPTYPE_DECLARE(EMBEDDING_APPLY_ADAM, "EmbeddingApplyAdam");
REGISTER_OPTYPE_DECLARE(EMBEDDING_APPLY_ADAM_W, "EmbeddingApplyAdamW");
REGISTER_OPTYPE_DECLARE(EMBEDDING_APPLY_ADA_GRAD, "EmbeddingApplyAdaGrad");
REGISTER_OPTYPE_DECLARE(EMBEDDING_TABLE_EXPORT, "EmbeddingTableExport");
REGISTER_OPTYPE_DECLARE(EMBEDDING_TABLE_IMPORT, "EmbeddingTableImport");
REGISTER_OPTYPE_DECLARE(EMBEDDING_COMPUTE_VAR_EXPORT, "EmbeddingComputeVarExport");
REGISTER_OPTYPE_DECLARE(TABLE_TO_RESOURCE, "TableToResource");

// Data flow
REGISTER_OPTYPE_DECLARE(FLOWNODE, "FlowNode");
REGISTER_OPTYPE_DECLARE(FLOWFUNC, "FlowFunc");

// Dsa
const char_t *const DSAGENBITMASK = "DSAGenBitMask";
const char_t *const DSARANDOMTRUNCATEDNORMAL = "DSARandomTruncatedNormal";
const char_t *const DSARANDOMNORMAL = "DSARandomNormal";
const char_t *const DSARANDOMUNIFORM = "DSARandomUniform";

const std::set<std::string> kFixedAddrNodeTypes = {DSAGENBITMASK, DSARANDOMNORMAL, DSARANDOMTRUNCATEDNORMAL,
                                                   DSARANDOMUNIFORM};
// @brief encryption type of the model file
enum ModelEncryptType {
  UNENCRYPTED,  // not encrypted
  ENCRYPTED     // encrypted
};

///
/// @brief signature verification
///
enum ModelCheckType {
  CHECK,   // signature verification
  UNCHECK  // no verification
};

///
/// @brief dynamic input type
///
enum DynamicInputType {
  FIXED = 0,  // default mode
  DYNAMIC_BATCH = 1,
  DYNAMIC_IMAGE = 2,
  DYNAMIC_DIMS = 3
};

///
/// @brief magic number of the model file
///
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t MODEL_FILE_MAGIC_NUM;

///
/// @brief model header length
///
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t MODEL_FILE_HEAD_LEN;

///
/// @brief model name length
///
constexpr uint32_t MODEL_NAME_LENGTH = 32U;

///
/// @brief length of user-defined information
///
constexpr uint32_t USER_DEFINE_INFO_LENGTH = 32U;

///
/// @brief length of the model file signature
///
constexpr uint32_t MODEL_FILE_CHECKSUM_LENGTH = 64U;

///
/// @brief length of the reserved field in the model file header
///
constexpr uint32_t MODEL_FILE_RESERVED_LENGTH = 63U;

// DATA node type
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string DATA_TYPE;

// DATA Operator Type
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string AIPP_DATA_TYPE;

// framework Operator Type
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string FRAMEWORK_OP_TYPE;

// DATA node type
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string ANN_DATA_TYPE;
// convolution node type
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_NET_OUTPUT;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_END_GRAPH;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_OP_DEBUG;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string OP_TYPE_OP_DEBUG;

// delimiter of operator configuration items
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string OP_CONF_DELIMITER;

// dim default size value
constexpr int32_t DIM_DEFAULT_SIZE = 4;

// default NCHW index
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_N;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_C;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_H;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_W;

// default NHWC index
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_N;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_H;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_W;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_C;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t MODEL_VERSION;  // model version 1.0

// flowctrl
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_STREAM_SWITCH;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_FLOWCTRL_LOOP_PER_ITER;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_FLOWCTRL_LOOP_COND;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_FLOWCTRL_LOOP_INCREMENT;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_FLOWCTRL_LOOP_RESETVALUE;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_FLOWCTRL_LOOP_ASSIGNADD;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_FLOWCTRL_LOOP_ASSIGN;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_ATOMIC_ADDR_CLEAN;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_ATOMIC_MEMSET;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t TRUE_STREAM_ID;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t STREAM_SWITCH_INPUT_NUM;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const std::string NODE_NAME_GLOBAL_STEP;

constexpr uint32_t PLATFORM_VERSION_LEN = 20U;

enum class OsCpuInfoCheckTyep :uint8_t {
  NO_CHECK,
  NEED_CHECK
};

// Definition of the file header of the model file
struct ModelFileHeader {
  uint32_t magic = MODEL_FILE_MAGIC_NUM;               // magic number of DOMI
  uint32_t headsize = MODEL_FILE_HEAD_LEN;             // length of the model header. The value is fixed at 256
  uint32_t version = MODEL_VERSION;                    // version 1.0
  uint8_t checksum[MODEL_FILE_CHECKSUM_LENGTH] = {0U};  // signature
  uint32_t length = 0U;  // Ciphertext length. In the non-encryption model, the length is the plaintext length.
  // whether encrypted 0:not encrypt, 1:encrypt
  uint8_t is_encrypt = static_cast<uint8_t>(ModelEncryptType::UNENCRYPTED);
  uint8_t is_checksum = static_cast<uint8_t>(ModelCheckType::CHECK);            // whether to check the checksum
  uint8_t modeltype = 0U;                                  // 0:IR model 1:standard model 2:OM Tiny model 3:flow model
  uint8_t genmode = 0U;                                    // 0：offline generate 1：online generate
  uint8_t name[MODEL_NAME_LENGTH] = {0U};                  // Model name, which contains 32 characters
  uint32_t ops = 0U;                                       // Computing power (Kops)
  uint8_t userdefineinfo[USER_DEFINE_INFO_LENGTH] = {0U};  // User-defined information. The value contains 32 characters
  uint32_t om_ir_version = 0U;
  uint32_t model_num = 0U;
  uint8_t platform_version[PLATFORM_VERSION_LEN] = {0U};
  uint8_t platform_type = {0U};
  uint8_t padd[3] = {0};  // For initializing aligned memory
  uint64_t model_length = 0UL;
  uint8_t need_check_os_cpu_info = static_cast<uint8_t>(OsCpuInfoCheckTyep::NO_CHECK);
  uint8_t reserved[MODEL_FILE_RESERVED_LENGTH] = {0U};  // Reserved field 64
};

constexpr uint8_t TARGET_TYPE_LTTE_8BIT = 0U;
constexpr uint8_t TARGET_TYPE_MINI_8BIT = 1U;

// number of partitions in the current model
constexpr uint32_t PARTITION_SIZE = 5U;

constexpr uint8_t MODEL_TYPE_FLOW_MODEL = 3;

enum ModelPartitionType {
  MODEL_DEF = 0,
  WEIGHTS_DATA = 1,
  TASK_INFO = 2,
  TBE_KERNELS = 3,
  CUST_AICPU_KERNELS = 4,
  SO_BINS = 5,
  FLOW_MODEL = 6,
  FLOW_SUBMODEL = 7
};

struct TinyModelPartitionMemInfo {
  ModelPartitionType type;
  uint32_t mem_offset;
  uint32_t mem_size;
};

struct TinyModelPartitionTable {
  uint32_t num;
  TinyModelPartitionMemInfo partition[0];
};

inline uint64_t SizeOfTinyModelPartitionTable(const TinyModelPartitionTable &table) {
  return sizeof(TinyModelPartitionTable) + (sizeof(TinyModelPartitionMemInfo) * static_cast<uint64_t>(table.num));
}

struct ModelPartitionMemInfo {
  ModelPartitionType type;
  uint64_t mem_offset;
  uint64_t mem_size;
};

struct ModelPartitionTable {
  uint32_t num;
  ModelPartitionMemInfo partition[0];
};

inline uint64_t SizeOfModelPartitionTable(const ModelPartitionTable &table) {
  return sizeof(ModelPartitionTable) + (sizeof(ModelPartitionMemInfo) * static_cast<uint64_t>(table.num));
}
// mode of activation
typedef enum tagDomiActivationMode {
  DOMI_ACTIVATION_SIGMOID = 0,   // sigmoid
  DOMI_ACTIVATION_RELU,          // ReLU
  DOMI_ACTIVATION_TANH,          // tanh
  DOMI_ACTIVATION_CLIPPED_RELU,  // clipped ReLU
  DOMI_ACTIVATION_ELU,           // ELU
  DOMI_ACTIVATION_LEAKY_RELU,
  DOMI_ACTIVATION_ABS,             // Abs
  DOMI_ACTIVATION_RELU1,           // relu1
  DOMI_ACTIVATION_SOFTSIGN,        // softsign
  DOMI_ACTIVATION_SOFTPLUS,        // softplus
  DOMI_ACTIVATION_HARDSIGMOID,     // hardsigmoid
  DOMI_ACTIVATION_THRESHOLD_RELU,  // threshold
  DOMI_ACTIVATION_SELU,            // selu
  DOMI_ACTIVATION_LINEAR,          // linear
  DOMI_ACTIVATION_RESERVED
} domiActivationMode_t;

enum class MemorySizeCalcType {
  NORMAL = 0,
  ALWAYS_EMPTY
};

enum AicpuWorkSpaceType {
  CUST_LOG = 0,
  INVALID_TYPE
};
}  // namespace ge

namespace domi {
/// @brief Data structure definition related to task sinking
enum BuildMode {
  GEN_TASK_WITHOUT_L2FUSION = 3,  // Carrying task data (L2 convergence function disabled)
  GEN_TASK_WITHOUT_FUSION = 4,    // Carrying task data (all convergence functions disabled)
  GEN_TASK_WITH_FUSION = 5        // Carrying task data (with UB/L1/L2 enabled for all convergence functions)
};
}  // namespace domi

#endif  // INC_FRAMEWORK_COMMON_TYPES_H_
