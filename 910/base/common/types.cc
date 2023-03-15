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

#include "framework/common/types.h"
#include "external/graph/types.h"

namespace ge {
// dump
const std::string DUMP_MODEL = "model_name";
const std::string DUMP_ALL_MODEL = "ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME";
const std::string DUMP_LAYER_OP_MODEL = "LAYER_OP_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME";
const std::string DUMP_STATUS = "status";
const std::string DUMP_LAYER = "layer";
const std::string DUMP_FILE_PATH = "path";
const std::string DUMP_MODE = "dump_mode";

// op debug mode
const std::string OP_DEBUG_AICORE = "aicore_overflow";
const std::string OP_DEBUG_ATOMIC = "atomic_overflow";
const std::string OP_DEBUG_ALL = "all";

// Profile related constant
const std::string PROFILE_MODEL_ID = "modelId";

REGISTER_OPTYPE_DEFINE(DATA, "Data");
REGISTER_OPTYPE_DEFINE(AIPPDATA, "AippData");
REGISTER_OPTYPE_DEFINE(QUEUE_DATA, "QueueData");
REGISTER_OPTYPE_DEFINE(CONVOLUTION, "Convolution");
REGISTER_OPTYPE_DEFINE(CORRELATION, "Correlation");
REGISTER_OPTYPE_DEFINE(CORRELATIONV2, "Correlation_V2");
REGISTER_OPTYPE_DEFINE(DECONVOLUTION, "Deconvolution");
REGISTER_OPTYPE_DEFINE(POOLING, "Pooling");
REGISTER_OPTYPE_DEFINE(ELTWISE, "Eltwise");
REGISTER_OPTYPE_DEFINE(RELU, "ReLU");
REGISTER_OPTYPE_DEFINE(RELU6, "ReLU6");
REGISTER_OPTYPE_DEFINE(SIGMOID, "Sigmoid");
REGISTER_OPTYPE_DEFINE(ABSVAL, "AbsVal");
REGISTER_OPTYPE_DEFINE(TANH, "TanH");
REGISTER_OPTYPE_DEFINE(PRELU, "PReLU");
REGISTER_OPTYPE_DEFINE(BATCHNORM, "BatchNorm");
REGISTER_OPTYPE_DEFINE(FUSIONBATCHNORM, "FusionBatchNorm");
REGISTER_OPTYPE_DEFINE(SCALE, "Scale");
REGISTER_OPTYPE_DEFINE(FULL_CONNECTION, "FullConnection");
REGISTER_OPTYPE_DEFINE(SOFTMAX, "Softmax");
REGISTER_OPTYPE_DEFINE(PLUS, "Plus");
REGISTER_OPTYPE_DEFINE(ACTIVATION, "Activation");
REGISTER_OPTYPE_DEFINE(FLATTEN, "Flatten");
REGISTER_OPTYPE_DEFINE(FLATTENV2, "FlattenV2");
REGISTER_OPTYPE_DEFINE(ADD, "Add");
REGISTER_OPTYPE_DEFINE(SUB, "Sub");
REGISTER_OPTYPE_DEFINE(MUL, "Mul");
REGISTER_OPTYPE_DEFINE(MATMUL, "MatMul");
REGISTER_OPTYPE_DEFINE(RSQRT, "Rsqrt");
REGISTER_OPTYPE_DEFINE(BIASADD, "BiasAdd");
REGISTER_OPTYPE_DEFINE(RESHAPE, "Reshape");
REGISTER_OPTYPE_DEFINE(REFORMAT, "ReFormat");
REGISTER_OPTYPE_DEFINE(DEPCONVOLUTION, "ConvolutionDepthwise");
REGISTER_OPTYPE_DEFINE(DROPOUT, "Dropout");
REGISTER_OPTYPE_DEFINE(DROPOUTGENMASK, "DropOutGenMask");
REGISTER_OPTYPE_DEFINE(DROPOUTDOMASK, "DropOutDoMask");
REGISTER_OPTYPE_DEFINE(DROPOUTDOMASKV3, "DropOutDoMaskV3");
REGISTER_OPTYPE_DEFINE(DROPOUTDOMASKV3D, "DropOutDoMaskV3D");
REGISTER_OPTYPE_DEFINE(SOFTMAXV2WITHDROPOUTDOMASKV3D, "SoftmaxV2WithDropOutDoMaskV3D");
REGISTER_OPTYPE_DEFINE(AXPYWITHSOFTMAXANDDROPOUTDOMASK, "AxpyWithSoftmaxAndDropOutDoMask");
REGISTER_OPTYPE_DEFINE(ATTENTIONSCORE, "AttentionScore");
REGISTER_OPTYPE_DEFINE(ATTENTIONSCOREGRAD, "AttentionScoreGrad");
REGISTER_OPTYPE_DEFINE(CONCAT, "Concat");
REGISTER_OPTYPE_DEFINE(ROIPOOLING, "ROIPooling");
REGISTER_OPTYPE_DEFINE(PROPOSAL, "Proposal");
REGISTER_OPTYPE_DEFINE(FSRDETECTIONOUTPUT, "FSRDetectionOutput");
REGISTER_OPTYPE_DEFINE(DETECTIONPOSTPROCESS, "Detectpostprocess");
REGISTER_OPTYPE_DEFINE(LRN, "LRN");
REGISTER_OPTYPE_DEFINE(TRANSDATA, "TransData");
REGISTER_OPTYPE_DEFINE(PERMUTE, "Permute");
REGISTER_OPTYPE_DEFINE(SSDNORMALIZE, "SSDNormalize");
REGISTER_OPTYPE_DEFINE(SSDPRIORBOX, "SSDPriorBox");
REGISTER_OPTYPE_DEFINE(NETOUTPUT, "NetOutput");
REGISTER_OPTYPE_DEFINE(SSDDETECTIONOUTPUT, "SSDDetectionOutput");
REGISTER_OPTYPE_DEFINE(REFINEDETDETECTIONOUTPUT, "RefinedetDetectionOutput");
REGISTER_OPTYPE_DEFINE(CHANNELAXPY, "ChannelAxpy");
REGISTER_OPTYPE_DEFINE(PSROIPOOLING, "PSROIPooling");
REGISTER_OPTYPE_DEFINE(POWER, "Power");
REGISTER_OPTYPE_DEFINE(POW, "Pow");
REGISTER_OPTYPE_DEFINE(ROIALIGN, "ROIAlign");
REGISTER_OPTYPE_DEFINE(PYTHON, "Python");
REGISTER_OPTYPE_DEFINE(FREESPACEEXTRACT, "FreespaceExtract");
REGISTER_OPTYPE_DEFINE(SPATIALTF, "SpatialTransform");
REGISTER_OPTYPE_DEFINE(SHAPE, "Shape");
REGISTER_OPTYPE_DEFINE(SHAPEN, "ShapeN");
REGISTER_OPTYPE_DEFINE(ARGMAX, "ArgMax");
REGISTER_OPTYPE_DEFINE(GATHERND, "GatherNd");
REGISTER_OPTYPE_DEFINE(GATHER, "Gather");
REGISTER_OPTYPE_DEFINE(REALDIV, "RealDiv");
REGISTER_OPTYPE_DEFINE(PACK, "Pack");
REGISTER_OPTYPE_DEFINE(SLICE, "Slice");
REGISTER_OPTYPE_DEFINE(SLICED, "SliceD");
REGISTER_OPTYPE_DEFINE(FLOORDIV, "FloorDiv");
REGISTER_OPTYPE_DEFINE(SQUEEZE, "Squeeze");
REGISTER_OPTYPE_DEFINE(UNSQUEEZE, "Unsqueeze");
REGISTER_OPTYPE_DEFINE(SQUEEZEV2, "SqueezeV2");
REGISTER_OPTYPE_DEFINE(UNSQUEEZEV2, "UnsqueezeV2");
REGISTER_OPTYPE_DEFINE(SQUEEZEV3, "SqueezeV3");
REGISTER_OPTYPE_DEFINE(UNSQUEEZEV3, "UnsqueezeV3");
REGISTER_OPTYPE_DEFINE(STRIDEDSLICE, "StridedSlice");
REGISTER_OPTYPE_DEFINE(RANGE, "Range");
REGISTER_OPTYPE_DEFINE(RPNPROPOSALS, "RpnProposals");
REGISTER_OPTYPE_DEFINE(DECODEBBOX, "DecodeBbox");
REGISTER_OPTYPE_DEFINE(PAD, "Pad");
REGISTER_OPTYPE_DEFINE(PADV2, "PadV2");
REGISTER_OPTYPE_DEFINE(MIRRORPAD, "MirrorPad");
REGISTER_OPTYPE_DEFINE(TILE, "Tile");
REGISTER_OPTYPE_DEFINE(SIZE, "Size");
REGISTER_OPTYPE_DEFINE(CLIPBOXES, "ClipBoxes");
REGISTER_OPTYPE_DEFINE(FASTRCNNPREDICTIONS, "FastrcnnPredictions");
REGISTER_OPTYPE_DEFINE(SPLIT, "Split");
REGISTER_OPTYPE_DEFINE(SPLITV, "SplitV");
REGISTER_OPTYPE_DEFINE(EXPANDDIMS, "ExpandDims");
REGISTER_OPTYPE_DEFINE(EMPTY, "Empty");
REGISTER_OPTYPE_DEFINE(MEAN, "Mean");
REGISTER_OPTYPE_DEFINE(GREATER, "Greater");
REGISTER_OPTYPE_DEFINE(SWITCH, "Switch");
REGISTER_OPTYPE_DEFINE(SWITCHN, "SwitchN");
REGISTER_OPTYPE_DEFINE(MERGE, "Merge");
REGISTER_OPTYPE_DEFINE(SYMBOLICGRADIENT, "SymbolicGradient");
REGISTER_OPTYPE_DEFINE(REMOTECALL, "RemoteCall");
REGISTER_OPTYPE_DEFINE(_IF, "_If");
REGISTER_OPTYPE_DEFINE(STATELESSIF, "StatelessIf");
REGISTER_OPTYPE_DEFINE(IF, "If");
REGISTER_OPTYPE_DEFINE(CASE, "Case");
REGISTER_OPTYPE_DEFINE(STATELESSCASE, "StatelessCase");
REGISTER_OPTYPE_DEFINE(_WHILE, "_While");
REGISTER_OPTYPE_DEFINE(WHILE, "While");
REGISTER_OPTYPE_DEFINE(STATELESSWHILE, "StatelessWhile");
REGISTER_OPTYPE_DEFINE(FOR, "For");
REGISTER_OPTYPE_DEFINE(PARTITIONEDCALL, "PartitionedCall");
REGISTER_OPTYPE_DEFINE(STATEFULPARTITIONEDCALL, "StatefulPartitionedCall");
REGISTER_OPTYPE_DEFINE(FAKEPARAM, "FakeParam");
REGISTER_OPTYPE_DEFINE(TRANSPOSE, "Transpose");
REGISTER_OPTYPE_DEFINE(TRANSPOSED, "TransposeD");
REGISTER_OPTYPE_DEFINE(CAST, "Cast");
REGISTER_OPTYPE_DEFINE(REGION, "Region");
REGISTER_OPTYPE_DEFINE(YOLO, "Yolo");
REGISTER_OPTYPE_DEFINE(YOLODETECTIONOUTPUT, "YoloDetectionOutput");
REGISTER_OPTYPE_DEFINE(FILL, "Fill");
REGISTER_OPTYPE_DEFINE(REVERSE, "Reverse");
REGISTER_OPTYPE_DEFINE(UNPACK, "Unpack");
REGISTER_OPTYPE_DEFINE(YOLO2REORG, "Yolo2Reorg");
REGISTER_OPTYPE_DEFINE(REDUCESUM, "ReduceSum");
REGISTER_OPTYPE_DEFINE(SUM, "Sum");
REGISTER_OPTYPE_DEFINE(CONSTANT, "Const");
REGISTER_OPTYPE_DEFINE(RESIZEBILINEAR, "ResizeBilinear");
REGISTER_OPTYPE_DEFINE(RESIZEBILINEARGRAD, "ResizeBilinearGrad");
REGISTER_OPTYPE_DEFINE(MAXIMUM, "Maximum");
REGISTER_OPTYPE_DEFINE(FRAMEWORKOP, "FrameworkOp");
REGISTER_OPTYPE_DEFINE(ARG, "_Arg");
REGISTER_OPTYPE_DEFINE(FUSEDBATCHNORMGRAD, "FusedBatchNormGrad");
REGISTER_OPTYPE_DEFINE(LSTM, "LSTM");
REGISTER_OPTYPE_DEFINE(HIGHWAY, "HighWay");
REGISTER_OPTYPE_DEFINE(RNN, "RNN");
REGISTER_OPTYPE_DEFINE(ATTENTIONDECODER, "AttentionDecoder");
REGISTER_OPTYPE_DEFINE(LOGICAL_NOT, "LogicalNot");
REGISTER_OPTYPE_DEFINE(LOGICAL_AND, "LogicalAnd");
REGISTER_OPTYPE_DEFINE(LOGICAL_OR, "LogicalOr");
REGISTER_OPTYPE_DEFINE(EQUAL, "Equal");
REGISTER_OPTYPE_DEFINE(NOTEQUAL, "NotEqual");
REGISTER_OPTYPE_DEFINE(INTERP, "Interp");
REGISTER_OPTYPE_DEFINE(SHUFFLECHANNEL, "ShuffleChannel");
REGISTER_OPTYPE_DEFINE(AIPP, "Aipp");
REGISTER_OPTYPE_DEFINE(MULTISHAPE, "MultiShape");
REGISTER_OPTYPE_DEFINE(RECIPROCAL, "Reciprocal");
REGISTER_OPTYPE_DEFINE(SELU, "Selu");
REGISTER_OPTYPE_DEFINE(ELU, "Elu");
REGISTER_OPTYPE_DEFINE(ACOSH, "Acosh");
REGISTER_OPTYPE_DEFINE(ASINH, "Asinh");
REGISTER_OPTYPE_DEFINE(MINIMUM, "Minimum");
REGISTER_OPTYPE_DEFINE(CLIP, "Clip");
REGISTER_OPTYPE_DEFINE(L2NORMALIZE, "L2Normalize");
REGISTER_OPTYPE_DEFINE(CROPANDRESIZE, "CropAndResize");
REGISTER_OPTYPE_DEFINE(UNUSEDCONST, "UnusedConst");
REGISTER_OPTYPE_DEFINE(SPARSETODENSE, "SparseToDense");
REGISTER_OPTYPE_DEFINE(NONMAXSUPPRESSION, "NonMaxSuppression");
REGISTER_OPTYPE_DEFINE(TOPKV2, "TopKV2");
REGISTER_OPTYPE_DEFINE(INVERTPERMUTATION, "InvertPermutation");
REGISTER_OPTYPE_DEFINE(MULTINOMIAL, "Multinomial");
REGISTER_OPTYPE_DEFINE(REVERSESEQUENCE, "ReverseSequence");
REGISTER_OPTYPE_DEFINE(REDUCEPROD, "ReduceProd");
REGISTER_OPTYPE_DEFINE(REDUCEMAX, "ReduceMax");
REGISTER_OPTYPE_DEFINE(REDUCEMIN, "ReduceMin");
REGISTER_OPTYPE_DEFINE(EXTRACTIMAGEPATCHES, "ExtractImagePatches");
REGISTER_OPTYPE_DEFINE(SQRT, "Sqrt");
REGISTER_OPTYPE_DEFINE(REDUCEALL, "ReduceAll");
REGISTER_OPTYPE_DEFINE(RESIZENEARESTNEIGHBOR, "ResizeNearestNeighbor");
REGISTER_OPTYPE_DEFINE(SPACETOBATCHND, "SpaceToBatchND");
REGISTER_OPTYPE_DEFINE(BATCHTOSPACEND, "BatchToSpaceND");
REGISTER_OPTYPE_DEFINE(ASSERT, "Assert");
REGISTER_OPTYPE_DEFINE(GREATEREQUAL, "GreaterEqual");
REGISTER_OPTYPE_DEFINE(FLOOR, "Floor");
REGISTER_OPTYPE_DEFINE(RANDOMUNIFORM, "RandomUniform");
REGISTER_OPTYPE_DEFINE(BATCHMATMUL, "BatchMatMul");
REGISTER_OPTYPE_DEFINE(SPACETODEPTH, "SpaceToDepth");
REGISTER_OPTYPE_DEFINE(DEPTHTOSPACE, "DepthToSpace");
REGISTER_OPTYPE_DEFINE(RINT, "Rint");
REGISTER_OPTYPE_DEFINE(ATAN, "Atan");
REGISTER_OPTYPE_DEFINE(ATAN2, "Atan2");
REGISTER_OPTYPE_DEFINE(ATANH, "Atanh");
REGISTER_OPTYPE_DEFINE(ACOS, "Acos");
REGISTER_OPTYPE_DEFINE(ASIN, "Asin");
REGISTER_OPTYPE_DEFINE(NEG, "Neg");
REGISTER_OPTYPE_DEFINE(LOG, "Log");
REGISTER_OPTYPE_DEFINE(TAN, "Tan");
REGISTER_OPTYPE_DEFINE(ROUND, "Round");
REGISTER_OPTYPE_DEFINE(UPSAMPLE, "Upsample");
REGISTER_OPTYPE_DEFINE(FLOORMOD, "FloorMod");
REGISTER_OPTYPE_DEFINE(LESS, "Less");
REGISTER_OPTYPE_DEFINE(LESSEQUAL, "LessEqual");
REGISTER_OPTYPE_DEFINE(ONEHOT, "OneHot");
REGISTER_OPTYPE_DEFINE(REFSWITCH, "RefSwitch");
REGISTER_OPTYPE_DEFINE(REFMERGE, "RefMerge");
REGISTER_OPTYPE_DEFINE(ENTER, "Enter");
REGISTER_OPTYPE_DEFINE(REFENTER, "RefEnter");
REGISTER_OPTYPE_DEFINE(LOOPCOND, "LoopCond");
REGISTER_OPTYPE_DEFINE(NEXTITERATION, "NextIteration");
REGISTER_OPTYPE_DEFINE(REFNEXTITERATION, "RefNextIteration");
REGISTER_OPTYPE_DEFINE(EXIT, "Exit");
REGISTER_OPTYPE_DEFINE(REFEXIT, "RefExit");
REGISTER_OPTYPE_DEFINE(CONTROLTRIGGER, "ControlTrigger");
REGISTER_OPTYPE_DEFINE(ZEROSLIKE, "ZerosLike");
REGISTER_OPTYPE_DEFINE(EXP, "Exp");
REGISTER_OPTYPE_DEFINE(WHERE, "Where");
REGISTER_OPTYPE_DEFINE(FAKEQUANTWITHMINMAXVARS, "FakeQuantWithMinMaxVars");
REGISTER_OPTYPE_DEFINE(SOFTPLUS, "Softplus");
REGISTER_OPTYPE_DEFINE(SOFTSIGN, "Softsign");
REGISTER_OPTYPE_DEFINE(COSH, "Cosh");
REGISTER_OPTYPE_DEFINE(SINH, "Sinh");
REGISTER_OPTYPE_DEFINE(SQUAREDDIFFERENCE, "SquaredDifference");
REGISTER_OPTYPE_DEFINE(REQUIREDSPACETOBATCHPADDINGS, "RequiredSpaceToBatchPaddings");  // for retinanet scope fusion
REGISTER_OPTYPE_DEFINE(SSDPOSTPROCESSOR, "SSDPostProcessor");
REGISTER_OPTYPE_DEFINE(RETINANETBOXES, "RetinanetBoxes");
REGISTER_OPTYPE_DEFINE(RETINAMULTIANCHORS, "RetinaMultiAnchor");
REGISTER_OPTYPE_DEFINE(RETINANETCLIPPEDBOXES, "RetinanetClippedBoxes");
REGISTER_OPTYPE_DEFINE(RETINANETFILTEREDDETECTIONS, "RetinanetFilteredDetections");
REGISTER_OPTYPE_DEFINE(RETINANETPOSTPROCESSOR, "RetinanetPostProcessor");
REGISTER_OPTYPE_DEFINE(RETINANETANCHORS, "RetinanetAnchors");
REGISTER_OPTYPE_DEFINE(FASTERRCNNMAP, "FasterRCNNMap");
REGISTER_OPTYPE_DEFINE(FASTERRCNNMAP1, "FasterRCNNMap1");
REGISTER_OPTYPE_DEFINE(FASTERRCNNSECONDSTAGEPOSTPROCESSOR, "FasterRCNNSecondStagePostprocessor");
REGISTER_OPTYPE_DEFINE(FASTERRCNNROIINTERPOOLING, "FasterRCNNROIInterPooling");
REGISTER_OPTYPE_DEFINE(FASTERRCNNFIRSTSTAGEPOSTPROCESSOR, "FasterRCNNFirstStagePostprocessor");
REGISTER_OPTYPE_DEFINE(FASTERRCNNGRIDANCHORGENERATOR, "FasterRCNNGridAnchorGenerator");
REGISTER_OPTYPE_DEFINE(ROIINTERPOOLING, "ROIInterPooling");
REGISTER_OPTYPE_DEFINE(FASTERRCNNCLIPTOWINDOW, "FasterRCNNClipToWindow");
REGISTER_OPTYPE_DEFINE(EMBEDLOOKUP, "EmbedLookup");
REGISTER_OPTYPE_DEFINE(HASHLOOKUP, "HashLookup");
REGISTER_OPTYPE_DEFINE(LSH_PROJ, "LshProject");
REGISTER_OPTYPE_DEFINE(SVDF, "SVDF");
REGISTER_OPTYPE_DEFINE(SSDANCHORGENERATOR, "SSDAnchorGenerator");
REGISTER_OPTYPE_DEFINE(IDENTITY, "Identity");
REGISTER_OPTYPE_DEFINE(IDENTITYN, "IdentityN");
REGISTER_OPTYPE_DEFINE(PLACEHOLDERWITHDEFAULT, "PlaceholderWithDefault");
REGISTER_OPTYPE_DEFINE(SELECT, "Select");
REGISTER_OPTYPE_DEFINE(GETSPAN, "GetSpan");
REGISTER_OPTYPE_DEFINE(STOPGRADIENT, "StopGradient");
REGISTER_OPTYPE_DEFINE(PREVENTGRADIENT, "PreventGradient");
REGISTER_OPTYPE_DEFINE(GUARANTEECONST, "GuaranteeConst");
REGISTER_OPTYPE_DEFINE(BROADCASTGRADIENTARGS, "BroadcastGradientArgs");
REGISTER_OPTYPE_DEFINE(BROADCASTARGS, "BroadcastArgs");
REGISTER_OPTYPE_DEFINE(CONFUSIONMATRIX, "ConfusionMatrix");
REGISTER_OPTYPE_DEFINE(RANK, "Rank");
REGISTER_OPTYPE_DEFINE(PLACEHOLDER, "PlaceHolder");
REGISTER_OPTYPE_DEFINE(END, "End");
REGISTER_OPTYPE_DEFINE(BASICLSTMCELL, "BasicLSTMCell");
REGISTER_OPTYPE_DEFINE(GETNEXT, "GetNext");
REGISTER_OPTYPE_DEFINE(ITERATOR, "Iterator");
REGISTER_OPTYPE_DEFINE(ITERATORV2, "IteratorV2");
REGISTER_OPTYPE_DEFINE(INITDATA, "InitData");
REGISTER_OPTYPE_DEFINE(REFIDENTITY, "RefIdentity");
REGISTER_OPTYPE_DEFINE(BITCAST, "Bitcast");
REGISTER_OPTYPE_DEFINE(GATHERSHAPES, "GatherShapes");
REGISTER_OPTYPE_DEFINE(FILECONSTANT, "FileConstant");

/***************Ann special operator*************************/
REGISTER_OPTYPE_DEFINE(ANN_MEAN, "AnnMean");
REGISTER_OPTYPE_DEFINE(ANN_CONVOLUTION, "AnnConvolution");
REGISTER_OPTYPE_DEFINE(ANN_DEPCONVOLUTION, "AnnDepthConv");
REGISTER_OPTYPE_DEFINE(ANN_FULLCONNECTION, "AnnFullConnection");
REGISTER_OPTYPE_DEFINE(ANN_NETOUTPUT, "AnnNetOutput");
REGISTER_OPTYPE_DEFINE(ANN_DATA, "AnnData");
REGISTER_OPTYPE_DEFINE(ANN_RESHAPE, "AnnReshape");
REGISTER_OPTYPE_DEFINE(ANN_ADD, "AnnAdd");
REGISTER_OPTYPE_DEFINE(ANN_MUL, "AnnMul");
REGISTER_OPTYPE_DEFINE(ANN_SUB, "AnnSub");
REGISTER_OPTYPE_DEFINE(ANN_DIV, "AnnDiv");
REGISTER_OPTYPE_DEFINE(ANN_DEQUANTIZE, "AnnDequant");
REGISTER_OPTYPE_DEFINE(ANN_QUANTIZE, "AnnQuant");
REGISTER_OPTYPE_DEFINE(ANN_PAD, "AnnPad");
REGISTER_OPTYPE_DEFINE(ANN_RESIZE_BILINEAR, "AnnResizeBilinear");

/***************************************************/
/******************Training operator*************************/
REGISTER_OPTYPE_DEFINE(GATHERV2, "GatherV2");
REGISTER_OPTYPE_DEFINE(CONVGRADFILTER, "Conv2DBackpropFilter");
REGISTER_OPTYPE_DEFINE(CONV2D, "Conv2D");
REGISTER_OPTYPE_DEFINE(CONV2DBACKPROPINPUT, "Conv2DBackpropInput");
REGISTER_OPTYPE_DEFINE(FUSEDBATCHNORM, "FusedBatchNorm");
REGISTER_OPTYPE_DEFINE(BIASADDGRAD, "BiasAddGrad");
REGISTER_OPTYPE_DEFINE(ACTIVATIONGRAD, "ReluGrad");
REGISTER_OPTYPE_DEFINE(MAXPOOLWITHARGMAX, "MaxPoolWithArgmax");
REGISTER_OPTYPE_DEFINE(MAXPOOLGRADWITHARGMAX, "MaxPoolGradWithArgmax");
REGISTER_OPTYPE_DEFINE(SPARSESOFTMAXCROSSENTROPYWITHLOGITS, "SparseSoftmaxCrossEntropyWithLogits");
REGISTER_OPTYPE_DEFINE(SNAPSHOT, "Snapshot");
REGISTER_OPTYPE_DEFINE(MEANGRAD, "MeanGrad");
REGISTER_OPTYPE_DEFINE(TRANSLATE, "Translate");
REGISTER_OPTYPE_DEFINE(ADDN, "AddN");
REGISTER_OPTYPE_DEFINE(L2LOSS, "L2Loss");
REGISTER_OPTYPE_DEFINE(MULTIPLY, "Multiply");
REGISTER_OPTYPE_DEFINE(HUBERLOSSGRAD, "HuberLossGrad");
REGISTER_OPTYPE_DEFINE(HUBERLOSS, "HuberLoss");
REGISTER_OPTYPE_DEFINE(NEGATIVE, "Negative");
REGISTER_OPTYPE_DEFINE(SSDCAST, "SSDCast");
REGISTER_OPTYPE_DEFINE(SPARSESOFTMAXCROSSENTROPY, "SsdSparseSoftmaxCrossEntropy");
REGISTER_OPTYPE_DEFINE(SPARSESOFTMAXCROSSENTROPYGRAD, "SsdSparseSoftmaxCrossEntropyGrad");
REGISTER_OPTYPE_DEFINE(SSDSQUEEZEFUSION, "SsdSqueezeFusion");
REGISTER_OPTYPE_DEFINE(CONCATFOUR2FIVE, "ConcatFour2Five");
REGISTER_OPTYPE_DEFINE(CONCATFIVE2FOUR, "ConcatFive2Four");
REGISTER_OPTYPE_DEFINE(SSDREALDIVTILEMUL, "SSDRealdivTileMul");
REGISTER_OPTYPE_DEFINE(SSDSUMMULREALDIVMEAN, "SSDSumMulRealdivMean");

REGISTER_OPTYPE_DEFINE(VARIABLEV2, "VariableV2");
REGISTER_OPTYPE_DEFINE(VARHANDLEOP, "VarHandleOp");
REGISTER_OPTYPE_DEFINE(TEMPORARYVARIABLE, "TemporaryVariable");
REGISTER_OPTYPE_DEFINE(DESTROYTEMPORARYVARIABLE, "DestroyTemporaryVariable");
REGISTER_OPTYPE_DEFINE(VARIABLE, "Variable");
REGISTER_OPTYPE_DEFINE(ASSIGN, "Assign");
REGISTER_OPTYPE_DEFINE(ASSIGNVARIABLEOP, "AssignVariableOp");
REGISTER_OPTYPE_DEFINE(ASSIGNADD, "AssignAdd");
REGISTER_OPTYPE_DEFINE(ASSIGNADDVARIABLEOP, "AssignAddVariableOp");
REGISTER_OPTYPE_DEFINE(ASSIGNSUB, "AssignSub");
REGISTER_OPTYPE_DEFINE(ASSIGNSUBVARIABLEOP, "AssignSubVariableOp");
REGISTER_OPTYPE_DEFINE(APPLYMOMENTUM, "ApplyMomentum");
REGISTER_OPTYPE_DEFINE(RESOURCEAPPLYMOMENTUM, "ResourceApplyMomentum");
REGISTER_OPTYPE_DEFINE(SGD, "SGD");
REGISTER_OPTYPE_DEFINE(NOOP, "NoOp");
REGISTER_OPTYPE_DEFINE(READVARIABLEOP, "ReadVariableOp");
REGISTER_OPTYPE_DEFINE(PARALLELCONCATSTART, "_ParallelConcatStart");
REGISTER_OPTYPE_DEFINE(CONSTANTOP, "Constant");
REGISTER_OPTYPE_DEFINE(DEPTHWISECONV2DBACKPROPFILTER, "DepthwiseConv2dNativeBackpropFilter");
REGISTER_OPTYPE_DEFINE(DEPTHWISECONV2DBACKPORPINPUT, "DepthwiseConv2dNativeBackpropInput");
REGISTER_OPTYPE_DEFINE(DEPTHWISECONV2DFORWARDNATIVE, "DepthwiseConv2dNative");
REGISTER_OPTYPE_DEFINE(DROPOUTGRAD, "DropOutGrad");
REGISTER_OPTYPE_DEFINE(APPLYRMSPROPMIXEDPRECISION, "apply_rms_prop_mixed_precision");
REGISTER_OPTYPE_DEFINE(APPLYRMSPROP, "ApplyRMSProp");
REGISTER_OPTYPE_DEFINE(RELU6GRAD, "Relu6Grad");
REGISTER_OPTYPE_DEFINE(AVGPOOLGRAD, "AvgPoolGrad");
REGISTER_OPTYPE_DEFINE(CONCATV2, "ConcatV2");
REGISTER_OPTYPE_DEFINE(CONCATOFFSET, "ConcatOffset");
REGISTER_OPTYPE_DEFINE(LAYERNORMGRAD, "LayerNormGrad");
REGISTER_OPTYPE_DEFINE(LAYERNORM, "LayerNorm");
REGISTER_OPTYPE_DEFINE(LARS, "Lars");
REGISTER_OPTYPE_DEFINE(DYNAMICSTITCH, "DynamicStitch");

/***************************************************/
REGISTER_OPTYPE_DEFINE(SQUARE, "Square");
REGISTER_OPTYPE_DEFINE(HCOMBROADCAST, "HcomBroadcast");
REGISTER_OPTYPE_DEFINE(HCOMALLGATHER, "HcomAllGather");
REGISTER_OPTYPE_DEFINE(HCOMALLREDUCE, "HcomAllReduce");
REGISTER_OPTYPE_DEFINE(HCOMREDUCESCATTER, "HcomReduceScatter");
REGISTER_OPTYPE_DEFINE(HCOMREDUCE, "HcomReduce");
REGISTER_OPTYPE_DEFINE(HCOMSEND, "HcomSend");
REGISTER_OPTYPE_DEFINE(HCOMRECEIVE, "HcomReceive");
REGISTER_OPTYPE_DEFINE(HCOMREMOTEREAD, "HcomRemoteRead");
REGISTER_OPTYPE_DEFINE(HCOMREMOTEREFREAD, "HcomRemoteRefRead");
REGISTER_OPTYPE_DEFINE(HCOMREMOTEWRITE, "HcomRemoteWrite");
REGISTER_OPTYPE_DEFINE(HCOMREMOTESCATTERWRITE, "HcomRemoteScatterWrite");
REGISTER_OPTYPE_DEFINE(HCOMALLTOALLV, "HcomAllToAllV");
REGISTER_OPTYPE_DEFINE(HCOMGATHERALLTOALLV, "HcomGatherAllToAllV");
REGISTER_OPTYPE_DEFINE(HCOMALLTOALLVC, "HcomAllToAllVC");
REGISTER_OPTYPE_DEFINE(HCOMALLTOALL, "HcomAllToAll");
REGISTER_OPTYPE_DEFINE(HCOMREMOTELOOKUP, "HcomRemoteLookup");
REGISTER_OPTYPE_DEFINE(HCOMCOLLREMOTELOOKUP, "HcomCollRemoteLookup");
REGISTER_OPTYPE_DEFINE(HCOMCOLLREMOTEUPDATE, "HcomCollRemoteUpdate");

REGISTER_OPTYPE_DEFINE(VARASSIGN, "VarAssign");
REGISTER_OPTYPE_DEFINE(VARISINITIALIZEDOP, "VarIsInitializedOp");
REGISTER_OPTYPE_DEFINE(LogTimeStamp, "LogTimeStamp");
REGISTER_OPTYPE_DEFINE(ISVARIABLEINITIALIZED, "IsVariableInitialized");
REGISTER_OPTYPE_DEFINE(STREAMSWITCH, "StreamSwitch");
REGISTER_OPTYPE_DEFINE(STREAMSWITCHN, "StreamSwitchN");
REGISTER_OPTYPE_DEFINE(STREAMACTIVE, "StreamActive");
REGISTER_OPTYPE_DEFINE(MEMCPYASYNC, "MemcpyAsync");
REGISTER_OPTYPE_DEFINE(MEMCPYADDRASYNC, "MemcpyAddrAsync");
REGISTER_OPTYPE_DEFINE(STREAMMERGE, "StreamMerge");
REGISTER_OPTYPE_DEFINE(ENDGRAPH, "EndGraph");
REGISTER_OPTYPE_DEFINE(MODELEXIT, "ModelExit");
REGISTER_OPTYPE_DEFINE(SEND, "Send");
REGISTER_OPTYPE_DEFINE(RECV, "Recv");
REGISTER_OPTYPE_DEFINE(ENDOFSEQUENCE, "EndOfSequence");
REGISTER_OPTYPE_DEFINE(STARTOFSEQUENCE, "StartOfSequence");
REGISTER_OPTYPE_DEFINE(NPUGETFLOATSTATUS, "NPUGetFloatStatus");
REGISTER_OPTYPE_DEFINE(NPUCLEARFLOATSTATUS, "NPUClearFloatStatus");
REGISTER_OPTYPE_DEFINE(NPUGETFLOATSTATUSV2, "NPUGetFloatStatusV2");
REGISTER_OPTYPE_DEFINE(NPUCLEARFLOATSTATUSV2, "NPUClearFloatStatusV2");

REGISTER_OPTYPE_DEFINE(LABELSET, "LabelSet");
REGISTER_OPTYPE_DEFINE(LABELGOTO, "LabelGoto");
REGISTER_OPTYPE_DEFINE(LABELGOTOEX, "LabelGotoEx");
REGISTER_OPTYPE_DEFINE(LABELSWITCH, "LabelSwitch");
REGISTER_OPTYPE_DEFINE(LABELSWITCHBYINDEX, "LabelSwitchByIndex");

REGISTER_OPTYPE_DEFINE(ATOMICADDRCLEAN, "AtomicAddrClean");
REGISTER_OPTYPE_DEFINE(MEMSET, "MemSet");

REGISTER_OPTYPE_DEFINE(ABS_GRAD, "AbsGrad");
REGISTER_OPTYPE_DEFINE(ACCUMULATE_N_V2, "AccumulateNV2");
REGISTER_OPTYPE_DEFINE(ACOS_GRAD, "AcosGrad");
REGISTER_OPTYPE_DEFINE(ACOSH_GRAD, "AcoshGrad");
REGISTER_OPTYPE_DEFINE(ANY, "Any");
REGISTER_OPTYPE_DEFINE(APPROXIMATE_EQUAL, "ApproximateEqual");
REGISTER_OPTYPE_DEFINE(ASIN_GRAD, "AsinGrad");
REGISTER_OPTYPE_DEFINE(ASINH_GRAD, "AsinhGrad");
REGISTER_OPTYPE_DEFINE(ATAN_GRAD, "AtanGrad");
REGISTER_OPTYPE_DEFINE(BROADCAST_TO, "BroadcastTo");
REGISTER_OPTYPE_DEFINE(ELU_GRAD, "EluGrad");
REGISTER_OPTYPE_DEFINE(ADD_V2, "AddV2");
REGISTER_OPTYPE_DEFINE(DATAFORMATDIMMAP, "DataFormatDimMap");
REGISTER_OPTYPE_DEFINE(DATAFORMATVECPERMUTE, "DataFormatVecPermute");
REGISTER_OPTYPE_DEFINE(APPLYADADELTA, "ApplyAdadelta");
REGISTER_OPTYPE_DEFINE(APPLYADAGRAD, "ApplyAdagrad");
REGISTER_OPTYPE_DEFINE(APPLYADAGRADDA, "ApplyAdagradDA");
REGISTER_OPTYPE_DEFINE(APPLYADAM, "ApplyAdam");
REGISTER_OPTYPE_DEFINE(APPLYADAMAX, "ApplyAdaMax");
REGISTER_OPTYPE_DEFINE(APPLYADDSIGN, "ApplyAddSign");
REGISTER_OPTYPE_DEFINE(APPLYCENTEREDRMSPROP, "ApplyCenteredRMSProp");
REGISTER_OPTYPE_DEFINE(APPLYFTRL, "ApplyFtrl");
REGISTER_OPTYPE_DEFINE(APPLYFTRLV2, "ApplyFtrlV2");
REGISTER_OPTYPE_DEFINE(APPLYGRADIENTDESCENT, "ApplyGradientDescent");
REGISTER_OPTYPE_DEFINE(APPLYPOWERSIGN, "ApplyPowerSign");
REGISTER_OPTYPE_DEFINE(APPLYPROXIMALADAGRAD, "ApplyProximalAdagrad");
REGISTER_OPTYPE_DEFINE(APPLYPROXIMALGRADIENTDESCENT, "ApplyProximalGradientDescent");
REGISTER_OPTYPE_DEFINE(DEQUANTIZE, "Dequantize");

REGISTER_OPTYPE_DEFINE(FOCAL_LOSS, "FocalLoss");
REGISTER_OPTYPE_DEFINE(FOCAL_LOSS_GRAD, "FocalLossGrad");
REGISTER_OPTYPE_DEFINE(SMOOTHL1_LOSS, "SmoothL1Loss");
REGISTER_OPTYPE_DEFINE(SMOOTHL1_LOSS_grad, "SmoothL1LossGrad");
REGISTER_OPTYPE_DEFINE(REDUCEMEAN, "ReduceMean");
REGISTER_OPTYPE_DEFINE(CONCAT_V2, "ConcatV2");
REGISTER_OPTYPE_DEFINE(ONEHOT_V2, "OneHotV2");
REGISTER_OPTYPE_DEFINE(SLICE_V2, "SliceV2");
REGISTER_OPTYPE_DEFINE(TILE_V2, "TileV2");
REGISTER_OPTYPE_DEFINE(SUM_V2, "SumV2");
// Common type when the operator has the same name
REGISTER_OPTYPE_DEFINE(DETECTIONOUTPUT, "DetectionOutput");
// Custom operator
REGISTER_OPTYPE_DEFINE(CUSTOMOP, "CustomOp");
REGISTER_OPTYPE_DEFINE(CUSTOMOP_NCHW, "CustomOpNchw");
REGISTER_OPTYPE_DEFINE(CUSTOMOP_NHWC, "CustomOpNhwc");
REGISTER_OPTYPE_DEFINE(CUSTOMOP_NC1HWC0, "CustomOpNc1hwc0");

// Depthwise 4d_2_6d,6d_2_4d
REGISTER_OPTYPE_DEFINE(DEPTHWISEWEIGHT4D26D, "depthwise_weight_4d_2_6d");
REGISTER_OPTYPE_DEFINE(DEPTHWISEWEIGHT6D24D, "depthwise_weight_6d_2_4d");

REGISTER_OPTYPE_DEFINE(SQRTGRAD, "SqrtGrad");
REGISTER_OPTYPE_DEFINE(SIGMOIDGRAD, "SigmoidGrad");

REGISTER_OPTYPE_DEFINE(TRANSSHAPE, "TransShape");

// Horovod operator
REGISTER_OPTYPE_DEFINE(HVDCALLBACKALLREDUCE, "HorovodAllreduce");
REGISTER_OPTYPE_DEFINE(HVDCALLBACKALLGATHER, "HorovodAllgather");
REGISTER_OPTYPE_DEFINE(HVDCALLBACKBROADCAST, "HorovodBroadcast");
REGISTER_OPTYPE_DEFINE(HVDWAIT, "HorovodWait");

// aicpu op for online_infer dynamic_dims
REGISTER_OPTYPE_DEFINE(GETDYNAMICDIMS, "GetDynamicDims");

// profiling training trace node
REGISTER_OPTYPE_DEFINE(PROFILINGTRAININGTRACE, "ProfilingTrainingTrace");

// Stack series
REGISTER_OPTYPE_DEFINE(STACK, "Stack");
REGISTER_OPTYPE_DEFINE(STACKPUSH, "StackPush");
REGISTER_OPTYPE_DEFINE(STACKPOP, "StackPop");
REGISTER_OPTYPE_DEFINE(STACKCLOSE, "StackClose");

REGISTER_OPTYPE_DEFINE(EMBEDDING_TABLE_FIND, "EmbeddingTableFind");
REGISTER_OPTYPE_DEFINE(EMBEDDING_TABLE_FIND_AND_INIT, "EmbeddingTableFindAndInit");
REGISTER_OPTYPE_DEFINE(EMBEDDING_APPLY_ADAM, "EmbeddingApplyAdam");
REGISTER_OPTYPE_DEFINE(EMBEDDING_APPLY_ADA_GRAD, "EmbeddingApplyAdaGrad");
REGISTER_OPTYPE_DEFINE(TABLE_TO_RESOURCE, "TableToResource");

// Data flow
REGISTER_OPTYPE_DEFINE(FLOWNODE, "FlowNode");
REGISTER_OPTYPE_DEFINE(FLOWFUNC, "FlowFunc");

const std::string MODEL_ATTR_TASKS = "tasks";
const std::string MODEL_ATTR_TASK_GEN_BASE_ADDR = "task_gen_base_addr";
const std::string MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR = "task_gen_host_base_addr";
const std::string MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR = "task_gen_host_svm_base_addr";
const std::string MODEL_ATTR_HOST_MEMORY_SIZE = "host_memory_size";
const std::string MODEL_ATTR_HOST_SVM_SIZE = "host_svm_size";
const std::string MODEL_ATTR_TASK_GEN_WEIGHT_ADDR = "task_gen_weight_addr";
const std::string MODEL_ATTR_FUSION_MODEL_DEF = "fm";

#if !defined(__ANDROID__) && !defined(ANDROID)
const uint64_t ALLOC_MEMORY_MAX_SIZE = 8589934592U;  // Max size of 8 GB.
#else
const uint64_t ALLOC_MEMORY_MAX_SIZE = 536870912;  // Max size of 512M.
#endif

///
/// @brief Magic number of model file
///
const uint32_t MODEL_FILE_MAGIC_NUM = 0x444F4D49U;  // magic number

///
/// @brief Model head length
///
const uint32_t MODEL_FILE_HEAD_LEN = 256U;

///
/// @ingroup domi_omg
/// @brief DATA node type
///
const std::string DATA_TYPE = "Data";

///
/// @ingroup domi_omg
/// @brief DATA node type
///
const std::string AIPP_DATA_TYPE = "AippData";

///
/// @ingroup domi_omg
/// @brief Frame operator type
///
const std::string FRAMEWORK_OP_TYPE = "FrameworkOp";

///
/// @ingroup domi_omg
/// @brief Data node type
///
const std::string ANN_DATA_TYPE = "AnnData";
///
/// @ingroup domi_omg
/// @brief Convolution node type
///
const std::string NODE_NAME_NET_OUTPUT = "Node_Output";

const std::string NODE_NAME_END_GRAPH = "Node_EndGraph";

const std::string NODE_NAME_OP_DEBUG = "Node_OpDebug";
const std::string OP_TYPE_OP_DEBUG = "Opdebug";

///
/// @ingroup domi_omg
/// @brief Operator configuration item separator
///
const std::string OP_CONF_DELIMITER = ":";

///
/// @ingroup domi_omg
/// @brief NCHW index default value
///
const uint32_t NCHW_DIM_N = 0U;
const uint32_t NCHW_DIM_C = 1U;
const uint32_t NCHW_DIM_H = 2U;
const uint32_t NCHW_DIM_W = 3U;

///
/// @ingroup domi_omg
/// @brief NHWC index default value
///
const uint32_t NHWC_DIM_N = 0U;
const uint32_t NHWC_DIM_H = 1U;
const uint32_t NHWC_DIM_W = 2U;
const uint32_t NHWC_DIM_C = 3U;

const uint32_t MODEL_VERSION = 0x20000000U; ///< Model version 2.0///

// flowctrl
const std::string NODE_NAME_STREAM_SWITCH = "IteratorCtrl_StreamSwitch";
const std::string NODE_NAME_FLOWCTRL_LOOP_PER_ITER = "npu_runconfig/iterations_per_loop";
const std::string NODE_NAME_FLOWCTRL_LOOP_COND = "npu_runconfig/loop_cond";
const std::string NODE_NAME_FLOWCTRL_LOOP_INCREMENT = "npu_runconfig/one";
const std::string NODE_NAME_FLOWCTRL_LOOP_RESETVALUE = "npu_runconfig/zero";
const std::string NODE_NAME_FLOWCTRL_LOOP_ASSIGNADD = "FlowCtrl_LoopCond_ASSIGNADD";
const std::string NODE_NAME_FLOWCTRL_LOOP_ASSIGN = "FlowCtrl_LoopCond_ASSIGN";
const std::string NODE_NAME_ATOMIC_ADDR_CLEAN = "atomic_addr_clean";
const std::string NODE_NAME_ATOMIC_MEMSET = "atomic_memset";
const uint32_t TRUE_STREAM_ID = 0U;
const uint32_t STREAM_SWITCH_INPUT_NUM = 2U;

const std::string NODE_NAME_GLOBAL_STEP = "ge_global_step";
}  // namespace ge