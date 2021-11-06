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

#ifndef INC_FRAMEWORK_COMMON_OP_OP_PARSER_UTIL_H_
#define INC_FRAMEWORK_COMMON_OP_OP_PARSER_UTIL_H_

#include <climits>
#include <cmath>
#include <cstdint>

namespace ge {
// general
const float DEFAULT_ALPHA_VALUE = 1.0;
const float DEFAULT_BETA_VALUE = 0.0;
const uint32_t NORMAL_INPUT_NUM = 1;
const uint32_t NORMAL_OUTPUT_NUM = 1;
const uint32_t NORMAL_WORKSPACE_NUM = 0;
const int32_t NORMAL_1D_DIM_NUM = 1;
const int32_t NORMAL_SCALE_DIM_NUM = 0;
const int NORMAL_TENSOR_SIZE = 4;
const uint32_t DEFAULT_REAL_DIM_CNT = 4;

// const
const uint32_t CONST_OP_INPUT_NUM = 0;
const uint32_t CONST_OP_NORMAL_WEIGHT_SIZE = 1;

// MatMul
const uint32_t MATMUL_INPUT_NUM = 2;

// ActivationGrad
const int32_t ACTIVATIONGRAD_INPUT_NUM = 2;

// FusedBatchNorm
const int32_t FUSED_BATCH_NORM_WORKSPACE_NUM = 1;
const int32_t FUSED_BATCH_NORM_INPUT_NUM = 5;
const int32_t FUSED_BATCH_NORM_OUTPUT_NUM = 5;
// FusedBatchNormGrad
const int32_t FUSEDBATCHNORMGRAD_WORKSPACE_NUM = 1;
const int32_t FUSEDBATCHNORMGRAD_INPUT_NUM = 5;
const int32_t FUSEDBATCHNORMGRAD_OUTPUT_NUM = 3;

// conv
const uint32_t CONVOLUTION_WORKSPACE_NUM = 1;
const uint32_t CONVOLUTION_PAD_SIZE = 4;
const uint32_t CONVOLUTION_STRIDE_SIZE = 2;
const uint32_t CONVOLUTION_DILATION_SIZE = 2;
const int32_t CONVOLUTION_ADJ_SIZE = 2;
const int32_t CONVOLUTION_TARGET_SHAPE_SIZE = 2;

// ConvGradFilter
const uint32_t CONVGRADFILTER_WORKSPACE_NUM = 1;
const uint32_t CONVGRADFILTER_INPUT_NUM = 3;

// Pooling
const uint32_t POOLING_WINDOW_SIZE = 2;
const uint32_t POOLING_STRIDE_SIZE = 2;
const uint32_t POOLING_PAD_SIZE = 4;

// Add Sub Mul
const uint32_t ADD_INPUT_NUM = 2;
const uint32_t SUB_INPUT_NUM = 2;
const uint32_t MUL_INPUT_NUM = 2;
const uint32_t DIV_INPUT_NUM = 2;
const uint32_t ADD_WORKSPACE_NUM = 1;
const uint32_t SUB_WORKSPACE_NUM = 1;
const uint32_t MUL_WORKSPACE_NUM = 1;
const uint32_t DIV_WORKSPACE_NUM = 1;

const int32_t DEFAULT_AXIS_VALUE = -1;

const int32_t RESHAPE_AXIS_DEFAULT_VALUE = 0;
const int32_t RESHAPE_NUM_AXES_DEFAULT_VALUE = -1;
const uint32_t RESHAPE_WORKSPACE_NUM = 1;

const uint32_t FLATTEN_WORKSPACE_NUM = 1;

const int32_t CONCAT_MIN_INPUT_SIZE = 1;
const int32_t CONCAT_DEFAULT_AXIS = 1;
const uint32_t CONCAT_WORKSPACE_NUM = 1;

// The value for LRN parameters
const uint32_t LRN_DEFAULT_NORM_REGION = 0;
const float LRN_DEFAULT_K = 1.0;
const uint32_t LRN_DEFAULT_LOCAL_SIZE = 5;
const float LRN_DEFAULT_ALPHA = 1.0;
const float LRN_DEFAULT_BETA = 0.75;

///
///  @ingroup domi_common
///  @brief roipooling default value
///
const uint32_t ROIPOOLING_DEFAULT_POOLED_H = 0;
const uint32_t ROIPOOLING_DEFAULT_POOLED_W = 0;
const float ROIPOOLING_DEFAULT_SPATIAL_SCALE = 1;
const int32_t ROIPOOLING_DEFAULT_SAMPLING_RATIO = -1;

// DetectionOutput
const int32_t DETECTIONOUTPUT_INPUT_SIZE = 3;
const int32_t DETECTIONOUTPUT_OUTPUT_SIZE = 2;
const int32_t DETECTIONOUTPUT_WORKSPACE_NUM = 1;
const int DETECTIONOUTPUT_CLASS_NUM = 20;  // Number of background categories
const int DETECTIONOUTPUT_NUM_CLASSES_DEFAULT_VALUE = 21;
const float DETECTIONOUTPUT_NMS_THRESHOLD_DEFAULT_VALUE = 0.3;
const float DETECTIONOUTPUT_CONFIDENCE_THRESHOLD_DEFAULT_VALUE = 0.8;

// Proposal
const int32_t PROPOSAL_INPUT_SIZE = 3;
const int32_t PROPOSAL_OUTPUT_MAX_SIZE = 2;
const int32_t PROPOSAL_WORKSPACE_NUM = 1;
const float PROPOSAL_BASE_SIZE_DEFAULT_VALUE = 16;
const float PROPOSAL_RATIO_DIM_0_DEFAULT_VALUE = 0.5;
const float PROPOSAL_RATIO_DIM_1_DEFAULT_VALUE = 1;
const float PROPOSAL_RATIO_DIM_2_DEFAULT_VALUE = 2;
const float PROPOSAL_SCALE_DIM_0_DEFAULT_VALUE = 8;
const float PROPOSAL_SCALE_DIM_1_DEFAULT_VALUE = 16;
const float PROPOSAL_SCALE_DIM_2_DEFAULT_VALUE = 32;
const float PROPOSAL_MIN_SIZE_DEFAULT_VALUE = 16;
const int PROPOSAL_PRE_NMS_TOPN_DEFAULT_VALUE = 6000;
const int PROPOSAL_POST_NMS_TOPN_DEFAULT_VALUE = 304;
const float PROPOSAL_NMS_THRESH_DEFAULT_VALUE = 0.7;
const float PROPOSAL_FILTER_THRESH_DEFAULT_VALUE = 0;

// TVM OP
const uint32_t DEFAULT_KERNEL_BLOCK_DIM = 1;

// Softmax
const int32_t SOFTMAX_WORKSPACE_NUM = 1;

// SoftmaxCrossEntropy
const int32_t SOFTMAXCROSSENTROPY_INPUT_NUM = 2;
const int32_t SOFTMAXCROSSENTROPY_OUTPUT_NUM = 2;

// Permute
const int32_t PERMUTE_INPUT_NUM = 1;
const int32_t PERMUTE_OUTPUT_NUM = 1;
const int32_t PERMUTE_WORKSPACE_NUM = 1;
const int32_t PERMUTE_ORDER_NUM = 4;

// Ssd normalize
const int SSD_NORMALIZE_INPUT_SIZE = 1;
const float SSD_NORMALIZE_EPS_DEFAULT_VALUE = 2e-7;

// SsdPriroBox
const int32_t SSD_PRIOR_BOX_WORKSPACE_NUM = 1;
const int32_t SSD_PRIOR_BOX_INPUT_NUM = 2;
const bool SSD_PRIOR_BOX_FLIP_VALUE = true;
const bool SSD_PRIOR_BOX_CLIP_VALUE = false;
const double SSD_PRIOR_BOX_ASPECT_OFFSET_VALUE = 0.5;
const double SSD_PRIORBOX_VARIANCE_VALUE = 0.1;
const double SSD_PRIORBOX_VARIANCE_SIZE_ONE = 1;
const double SSD_PRIORBOX_VARIANCE_SIZE_FOUR = 4;
const double SSD_PRIORBOX_ASPECT_RATIO_VALUE = 1.0;
const int SSD_PRIOR_BOX_CODETYPE_CORNER_VALUE = 1;
const int SSD_PRIOR_BOX_CODETYPE_CENTER_SIZE_VALUE = 2;
const int SSD_PRIOR_BOX_CODETYPE_CORNER_SIZE_VALUE = 3;

// Ssd DetectionOutput
const int32_t SSD_DETECTIONOUTPUT_INPUT_SIZE = 3;
const int32_t SSD_DETECTIONOUTPUT_INPUT_SIZE_AFTER_FUSION = 2;
const int32_t SSD_DETECTIONOUTPUT_OUTPUT_SIZE = 2;
const int32_t SSD_DETECTIONOUTPUT_OUTPUT_SIZE_AFTER_FUSION = 3;
const int32_t SSD_DETECTIONOUTPUT_WORKSPACE_NUM = 1;
const int32_t SSD_DETECTIONOUTPUT_WORKSPACE_NUM_AFTER_FUSION = 0;
const bool SSD_DETECTIONOUTPUT_SHARED_LOCATION_DEFAULT_VALUE = true;
const int32_t SSD_DETECTIONOUTPUT_BACKGROUND_LABEL_ID_DEFAULT_VALUE = 0;
const float SSD_DETECTIONOUTPUT_NMS_THRESHOLD_DEFAULT_VALUE = 0.3;
const int32_t SSD_DETECTIONOUTPUT_TOP_K_DEFAULT_VALUE = 200;
const float SSD_DETECTIONOUTPUT_ETA_DEFAULT_VALUE = 1.0;
const int32_t SSD_DETECTIONOUTPUT_KEEP_TOP_K_DEFAULT_VALUE = 200;
const bool SSD_DETECTIONOUTPUT_VARIANCE_ENCODED_IN_TARGET_DEFAULT_VALUE = false;
const float SSD_DETECTIONOUTPUT_CONFIDENCE_THRESHOLD_DEFAULT_VALUE = 0.1;

// Refinedet DetectionOutput
const int32_t REFINEDET_DETECTIONOUTPUT_INPUT_SIZE = 5;
const int32_t REFINEDET_DETECTIONOUTPUT_INPUT_SIZE_AFTER_FUSION = 2;
const int32_t REFINEDET_DETECTIONOUTPUT_OUTPUT_SIZE = 2;
const int32_t REFINEDET_DETECTIONOUTPUT_OUTPUT_SIZE_AFTER_FUSION = 3;
const int32_t REFINEDET_DETECTIONOUTPUT_WORKSPACE_NUM = 1;
const bool REFINEDET_DETECTIONOUTPUT_SHARED_LOCATION_DEFAULT_VALUE = true;
const int32_t REFINEDET_DETECTIONOUTPUT_BACKGROUND_LABEL_ID_DEFAULT_VALUE = 0;
const float REFINEDET_DETECTIONOUTPUT_NMS_THRESHOLD_DEFAULT_VALUE = 0.3;
const int32_t REFINEDET_DETECTIONOUTPUT_TOP_K_DEFAULT_VALUE = 200;
const float REFINEDET_DETECTIONOUTPUT_ETA_DEFAULT_VALUE = 1.0;
const bool REFINEDET_DETECTIONOUTPUT_VARIANCE_ENCODED_IN_TARGET_DEFAULT_VALUE = false;
const int32_t REFINEDET_DETECTIONOUTPUT_KEEP_TOP_K_DEFAULT_VALUE = 200;
const float REFINEDET_DETECTIONOUTPUT_CONFIDENCE_THRESHOLD_DEFAULT_VALUE = 0.1;
const float REFINEDET_DETECTIONOUTPUT_OBJECTNESS_SCORE_DEFAULT_VALUE = 0;

// Channel axpy
const int32_t CHANNEL_AXPY_INPUT_NUM = 3;
const int32_t CHANNEL_AXPY_INPUT_DIM_SIZE = 4;
const int32_t CHANNEL_AXPY_WORKSPACE_NUM = 1;

// Psroi pooling
const int PSROI_POOLING_INPUT_COUNT = 2;
const int PSROI_POOLING_WORKSPACE_NUM = 1;

// MaxPoolWithArgmax
const uint32_t MAX_POOL_WITH_ARGMAX_OUTPUT_NUM = 2;
const uint32_t MAX_POOL_GRAD_WITH_ARGMAX_INPUT_NUM = 3;

// AvgPoolGrad
const uint32_t AVG_POOL_GRAD_INPUT_NUM = 2;

// ROIAlign
const int32_t ROIALIGN_INPUT_SIZE = 2;
const int32_t ROIALIGN_WORKSPACE_NUM = 1;
const int32_t ROIALIGN_DEFAULT_POOLED_H = 1;
const int32_t ROIALIGN_DEFAULT_POOLED_W = 1;

// Correlation
const uint32_t CORRELATION_INPUT_NUM = 2;
const int CORRELATION_WORKSPACE_NUM = 1;

// Detectionpostprocess
const int32_t POSTPROCESS_INPUT_SIZE = 4;
const int32_t POSTPROCESS_OUTPUT_SIZE = 2;
const int32_t POSTPROCESS_WORKSPACE_NUM = 1;
const uint32_t POSTPROCESS_CLS_NUM_DEFAULT_VALUE = 12;
const uint32_t POSTPROCESS_POST_NMS_TOPN_DEFAULT_VALUE = 100;
const float POSTPROCESS_NMS_THRESH_DEFAULT_VALUE = 0.3;
const float POSTPROCESS_CONF_THRESH_DEFAULT_VALUE = 0.5;
const float POSTPROCESS_BBOX_REG_WEIGHT_DIM_DEFAULT_VALUE = 1.0;
const int32_t POSTPROCESS_BBOX_REG_WEIGHT_SIZE_DEFAULT_VALUE = 4;

// Split
const int32_t SPLIT_INPUT_NUM = 2;
const int32_t SPLIT_DEFAULT_AXIS_VALUE = 1;
const int32_t SPLIT_MIN_OUTPUT_SIZE = 1;

const uint32_t STRIDEDSLICE_INPUT_NUM = 4;
// Slice
const int32_t SLICE_INPUT_NUM = 3;
const int32_t SLICE_WEIGHT_NUM = 2;

// GatherNd
const int32_t GATHERND_INPUT_NUM = 2;
// ArgMax
const int32_t ARGMAX_INPUT_NUM = 2;
const int32_t ARGMAX_REAL_INPUT_NUM = 1;

// HighWay
const int32_t HIGHWAY_INPUT_NUM = 4;
const int32_t HIGHWAY_WORKSPACE_NUM = 1;
// RealDiv
const int32_t REALDIV_INPUT_NUM = 2;

// Range
const int32_t RANGE_INPUT_NUM = 3;
const int32_t RANGE_OUTPUT_NUM = 1;
const int32_t RANGE_INPUT_DIM_SIZE = 0;

// Pad
const int32_t PAD_WEIGHT_NUM = 1;
const int32_t PAD_DIM_SIZE = 2;
const int32_t PAD_DIM0 = 4;
const int32_t PAD_DIM1 = 2;
const int32_t PAD_WEIGHT_WITH_CONSTANT_NUM = 2;
const int32_t PAD_CONSTATNT_DEFAULT_VALUE = 0;
const int32_t PAD_PADDINGS_SIZE = 8;

// Tile
const int32_t TILE_WEIGHT_NUM = 1;
const int32_t TILE_MULTIPLES_DIM_SIZE = 1;

// DecodeBbox
const int32_t DECODE_BBOX_INPUT_NUM = 2;

// GenerateRpnProposals
const int32_t GENERATE_RPN_PROPOSAL_INPUT_SIZE = 2;
const int32_t GENERATE_RPN_PROPOSAL_OUTPUT_SIZE = 3;

// Decode_BBox
const int32_t DECODE_BBOX_INPUT_SIZE = 2;
const int32_t DEFAULT_DECODE_CLIP_VALUE = 0;

// FastRcnnPredictions
const int32_t FASTRCNN_PREDICTIONS_INPUT_SIZE = 2;
const int32_t FASTRCNN_PREDICTIONS_OUTPUT_SIZE = 4;

const int32_t CLIP_BOXES_INPUT_NUM = 1;
const int32_t CLIP_BOXES_WEIGHT_SIZE = 1;
const int32_t CLIP_BOXES_WEIGHT_ITEM_SIZE = 2;
const int32_t CLIP_BOXES_OUTPUT_NUM = 1;

const int32_t FLOORDIV_INPUT_NUM = 2;
// Mean
const int32_t MEAN_WEIGHT_SIZE = 1;
const int32_t MEAN_WEIGHT_DIM_SIZE = 1;
const int32_t MEAN_WEIGHT_DIM = 2;
const int32_t MEAN_FIRST_AXIS = 2;
const int32_t MEAN_SECOND_AXIS = 3;
const int32_t MEAN_STRIDE_PLACE_HOLD = 1;
// Switch
const uint32_t SWITCH_INPUT_NUM = 2;
const uint32_t SWITCH_OUTPUT_NUM = 2;
// Merge
const uint32_t MERGE_INPUT_NUM = 2;
// Greater
const uint32_t GREATER_OUTPUT_NUM = 1;
const uint32_t GREATER_INPUT_NUM = 0;
const uint32_t GREATER_WEIGHT_NUM = 2;

// Yolo region
const uint32_t YOLO_REGION_OUTPUT_NUM = 3;
const uint32_t YOLO_REGION_WORKSPACE_NUM = 1;
const uint32_t YOLO_REGION_COORDS = 4;
const uint32_t YOLO_REGION_CLASSES = 20;
const uint32_t YOLO_REGION_BOXES = 1;
const bool YOLO_REGION_BACKGROUND = false;
const bool YOLO_REGION_SOFTMAX = false;
const bool YOLO_REGION_SOFTMAX_TREE = false;

// Yolo detectionoutput
const uint32_t YOLO_DETECTIONOUTPUT_INPUT_SIZE = 4;
const uint32_t YOLO_DETECTIONOUTPUT_OUTPUT_SIZE = 2;
const uint32_t YOLO_DETECTION_OUTPUT_WORKSPACE_NUM = 1;
const uint32_t YOLO_DETECTION_OUTPUT_CLASSES = 20;
const uint32_t YOLO_DETECTION_OUTPUT_BOXES_V2 = 5;
const uint32_t YOLO_DETECTION_OUTPUT_BOXES_V3 = 3;
const bool YOLO_DETECTION_OUTPUT_RELATIVE = true;
const float YOLO_DETECTION_OUTPUT_OBJECTNESS_THRESHOLD = 0.5;
const float YOLO_DETECTION_OUTPUT_CLASS_THRESHOLD = 0.5;
const uint32_t YOLO_DETECTION_OUTPUT_POST_TOP_K = UINT_MAX;
const float YOLO_DETECTION_OUTPUT_NMS_THRESHOLD = 0;
const float YOLO_DETECTION_OUTPUT_IOU_THRESHOLD_DECAY = 1.0;
const float YOLO_DETECTION_OUTPUT_COOR_SCALE_FACTOR = 1.0;

// Reorg
const int32_t REORG_DEFAULT_STRIDE = 2;
const uint32_t REORG_INPUT_COUNT = 1;
// Reshape
const int32_t RESHAPE_INPUT_NUM = 2;
// Maximum
const int32_t MAXIMUM_INPUT_NUM = 2;

// Spatialtf
const int32_t SPATIALTF_WORKSPACE_NUM = 1;

const int32_t REVERSE_DEFAULT_AXIS = 1;
// Crop
const int32_t CROP_AXIS = 2;
const int32_t CROP_INPUT_NUM = 2;

// ConvGradInput
const uint32_t CONVGRADINPUT_WORKSPACE_NUM = 1;
const uint32_t CONVGRADINPUT_INPUT_NUM = 3;

// RNN
const uint32_t RNN_WORKSPACE_NUM = 1;

// Cropandresize
const int32_t CROPANDRESIZE_WEIGHT_NUM = 1;
const int32_t CROPANDRESIZE_CROP_DIM_SIZE = 1;
const int32_t CROP_DIM0 = 2;

// Attention decoder weight index
const uint32_t ATTENTION_DECODER_WEIGHT_ATTENW0 = 0;
const uint32_t ATTENTION_DECODER_WEIGHT_ATTENTION0_KERNEL = 1;
const uint32_t ATTENTION_DECODER_WEIGHT_ATTNOUTPUTPROJECTION_KERNEL = 2;
const uint32_t ATTENTION_DECODER_WEIGHT_ATTENTION_DECODER_KERNEL = 3;
const uint32_t ATTENTION_DECODER_WEIGHT_CELL0_GATES_KERNEL = 4;
const uint32_t ATTENTION_DECODER_WEIGHT_CELL0_CANDIDATE_KERNEL = 5;
const uint32_t ATTENTION_DECODER_WEIGHT_CELL1_GATES_KERNEL = 6;
const uint32_t ATTENTION_DECODER_WEIGHT_CELL1_CANDIDATE_KERNEL = 7;
const uint32_t ATTENTION_DECODER_WEIGHT_ATTENTION0_BIAS = 8;
const uint32_t ATTENTION_DECODER_WEIGHT_ATTNOUTPUTPROJECTION_BIAS = 9;
const uint32_t ATTENTION_DECODER_WEIGHT_ATTENTION_DECODER_BIAS = 10;
const uint32_t ATTENTION_DECODER_WEIGHT_CELL0_GATES_BIAS = 11;
const uint32_t ATTENTION_DECODER_WEIGHT_CELL0_CANDIDATE_BIAS = 12;
const uint32_t ATTENTION_DECODER_WEIGHT_CELL1_GATES_BIAS = 13;
const uint32_t ATTENTION_DECODER_WEIGHT_CELL1_CANDIDATE_BIAS = 14;
const uint32_t ATTENTION_DECODER_WEIGHT_EMBEDDING = 15;
const uint32_t ATTENTION_DECODER_WEIGHT_ATTENVA = 16;
const uint32_t ATTENTION_DECODER_WEIGHT_DECODER_INITIAL = 17;
// Attention decoder weight size
const uint32_t ATTENTION_DECODER_WEIGHT_SIZE = 18;

const uint32_t ATTENTION_DECODER_INPUT_SIZE = 2;
const uint32_t ATTENTION_DECODER_WORKSPACE_NUM = 1;
const uint32_t ATTENTION_DECODER_INPUT_DECODER_INPUTS = 0;
const uint32_t ATTENTION_DECODER_INPUT_DECODER_INITIAL_HIDDEN = 1;

const int ATTENTION_DECODER_ALGO_NORMAL = 0;
const int ATTENTION_DECODER_SYMBOLS = 10000;
const int ATTENTION_DECODER_EMBEDDING_SIZE = 128;
const int ATTENTION_DECODER_ATTENTION_NUM_HIDDEN = 256;
const int ATTENTION_DECODER_DECODER_NUM_HIDDEN = 128;
const int ATTENTION_DECODER_DECODER_NUM_LAYERS = 2;
const int ATTENTION_DECODER_RNN_UNBIDIRECTIONAL = 0;
const int ATTENTION_DECODER_SEQLEN_VALUE = 57;
const int ATTENTION_DECODER_GRU = 3;

// Logicaland
const int32_t LOGICAL_AND_INPUT_NUM = 2;
const int32_t EQUAL_INPUT_NUM = 2;

static const int32_t OP_WEIGHT_MEM_BASE_OFFSET = 512;

// MultiShape
const uint32_t MULTI_SHAPE_INPUT_NUM = 2;

// Shufflechannel
const uint32_t SHUFFLECHANNEL_DEFAULT_GROUP = 1;
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_OP_OP_PARSER_UTIL_H_
