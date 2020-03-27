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

#include <vector>
#include <cce/cce.h>
#include <cce/dnn.h>
#include <cce/compiler_stub.h>
#include <cce/taskdown_api.h>

#include "cce/optimizer/fusion_engine.h"
#include "common/op/attr_value_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"

using namespace cce;
using namespace std;
using namespace ge;
using namespace fusion;

uint64_t global_mem_base = 0;

namespace cce {
#define DIM_MAX_SIZE 8
static const uint32_t C0 = 16;
struct tagCcPad {};
struct tagCcConvolution {};

struct tagCcLRN {};

struct tagCcFasterRcnnProposal {};
struct tagCcRoiAlign {};
struct tagCcBatchNorm {};
struct tagCcDetectpostprocess {};

struct tagCcSsdDetectionOutput {};

struct tagCcRefinedetDetectionOutput {};

struct tagCcMsrGenerateRpnProposals {};

struct tagCcFilter {
  vector<uint32_t> dims;
};

struct tagCcTensor {
  ccTensorFormat_t format;
  ccDataType_t data_type;
  uint32_t dim_cnt;
  int32_t realDimCnt;
  uint32_t data_size;
  int32_t dim_buf[DIM_MAX_SIZE];
  int32_t stride_buf[DIM_MAX_SIZE];
};

typedef struct tagCcPooling {
  ccPoolingMode_t mode;
  ccPaddingMode_t padMode;
  ccNanPropagation_t maxpoolingNanOpt;
  uint32_t dimCnt;
  int32_t windowDim[6];
  int32_t padding[6];
  int32_t stride[6];
} ccPooling_t;

struct tagCcActivation {};

struct tagCcFasterRcnnDetectionOutput {};
struct tagCcSpatialTransformer {};

struct tagCcPower {};
struct tagCcResizeBilinear {};
struct tagCcSsdNormalize {};
struct tagCcSsdPostProcessor {};
struct tagCcSsdPriorBox {};
struct tagCcPsRoiPooling {};

struct tagMsrFastRcnnPredictions {};
struct tagCcPRelu {};
struct tagCcStridedSlice {};

struct tagCcStridedSliceAttrs {};

struct tagCcRnn {};

struct tagCcArgmaxmin {};

typedef struct tagCcLog {
  ccDataType_t dataType;
  uint32_t paramCnt;
} ccLog_t;
typedef struct tagCcLog *ccLogDescriptor_t;

struct tagCcPadV2 {};

ccStatus_t ccGetPadV2OutputDim(const ccTensorDescriptor_t xDesc, const ccPadV2Descriptor_t padDesc, int32_t *dimCnt,
                               int32_t dim[], int32_t dimLen) {
  *dimCnt = 4;
  dim[0] = 1;
  dim[1] = 2;
  dim[2] = 2;
  dim[3] = 3;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccPadV2Forward(ccHandle_t handle, const ccPadV2Descriptor_t padDesc, const void *alpha,
                          const ccTensorDescriptor_t xDesc, const void *x, const void *beta,
                          const ccTensorDescriptor_t outputDesc, void *output) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccCreatePadV2Descriptor(ccPadV2Descriptor_t *padDesc) { return CC_STATUS_SUCCESS; }

ccStatus_t ccDestroyPadV2Descriptor(ccPadV2Descriptor_t *padDesc) { return CC_STATUS_SUCCESS; }

ccStatus_t ccSetKernelOpMap(ccHandle_t handle) { return CC_STATUS_SUCCESS; }

ccStatus_t ccDataDumpForward(ccHandle_t handle, const void *buffer, const uint64_t bufLen, const uint32_t taskIndex) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetPadV2Descriptor(ccPadV2Descriptor_t padDesc, const int32_t padShapeCnt, const int32_t padShapeLow[],
                                const int32_t padShapeHigh[], const ccPadMode_t padMode, const void *padValue,
                                const ccDataType_t padValueType) {
  return CC_STATUS_SUCCESS;
}

struct tagCcYoloDetectionOutput {
  ccYoloVersion_t yoloVersion;
  uint32_t netH;
  uint32_t netW;
  uint32_t postTopK;
  uint32_t classes;
  float nmsThreshold;
  float iouThreDecay;
  float coorScaleFactor;
  bool relative;
  float objThreshold;
  float clsThreshold;
  uint32_t biasNum;
  float *bias;
};

struct tagCcYoloRegion {};

struct tagCcEltwise {};

struct tagCcHashTableLookup {};

struct tagCcEmbeddingAttnDecoder {};
struct tagNonMaxSuppression {};

struct tagCcArcSinCos {};
struct tagCcPow {};
struct tagCcConcatFive2Four_t {};
struct tagCcConcatFour2Five_t {};

ccStatus_t ccCreatePowDescriptor(ccPowDescriptor_t *powDesc) {
  *powDesc = new tagCcPow();
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetPowDescriptor(ccPowDescriptor_t powDesc, ccDataType_t dataType, uint32_t paramCnt) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccDestroyPowDescriptor(ccPowDescriptor_t *powDesc) {
  if (nullptr == powDesc) {
    return CC_STATUS_BAD_PARAM;
  }

  delete *powDesc;
  *powDesc = 0;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccPowForward(ccHandle_t handle, const ccPowDescriptor_t powDesc, const void *powParam, const void *alpha,
                        const ccTensorDescriptor_t xDesc, const void *x, const ccTensorDescriptor_t yDesc,
                        const void *y, const void *beta, const ccTensorDescriptor_t zDesc, void *z) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccLogicalOrForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                              const ccTensorDescriptor_t yDesc, const void *y, const void *beta,
                              const ccTensorDescriptor_t outputDesc, void *output) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccCompareForward(ccHandle_t handle, ccCompareType_t compareType, const void *alpha,
                            const ccTensorDescriptor_t xDesc, const void *x, const ccTensorDescriptor_t yDesc,
                            const void *y, const void *beta, const ccTensorDescriptor_t outputDesc, void *output) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccGetCompareOutputDim(const ccTensorDescriptor_t xDesc, const ccTensorDescriptor_t yDesc, int32_t *dimCnt,
                                 int32_t *dim, int32_t dimLen) {
  *dimCnt = 4;
  dim[0] = 1;
  dim[1] = 1;
  dim[2] = 1;
  dim[3] = 1;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccArcTanForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                           const void *beta, const ccTensorDescriptor_t yDesc, void *y) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccAtanhForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t xDesc, const void *x,
                          const void *beta, const ccTensorDescriptor_t yDesc, void *y) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccIsDepthwiseHighPerformance(int32_t inputN, int32_t inputC, int32_t inputH, int32_t inputW, int32_t filterN,
                                        int32_t filterC, int32_t filterH, int32_t filterW, int32_t dilationH,
                                        int32_t dilationW, int32_t padHHead, int32_t padHTail, int32_t padWHead,
                                        int32_t padWTail, int32_t strideH, int32_t strideW, int32_t groupNum,
                                        bool &isHighPerformance, bool isquant, ccDataType_t inputDataType,
                                        ccDataType_t outputDataType) {
  isHighPerformance = true;
  return CC_STATUS_SUCCESS;
}

struct tagCcSpaceToBatch {};

struct tagCcBatchToSpace {};

struct tagCcResizeNearestNeighbor {};

ccStatus_t ccGetStream(ccHandle_t handle, rtStream_t *streamId) { return CC_STATUS_SUCCESS; }

ccStatus_t ccGetRtVersion(uint32_t *count) { return CC_STATUS_SUCCESS; }

ccStatus_t ccDestroyTensorDescriptor(ccTensorDescriptor_t *tensorDesc) {
  if (nullptr == tensorDesc) {
    return CC_STATUS_BAD_PARAM;
  }
  delete *tensorDesc;
  *tensorDesc = 0;
  return CC_STATUS_SUCCESS;
}
ccStatus_t ccDestroyFilterDescriptor(ccFilterDescriptor_t *filterDesc) {
  delete *filterDesc;
  *filterDesc = 0;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccGetFilterSizeInBytes(const ccFilterDescriptor_t filterDesc, uint32_t *size) {
  *size = filterDesc->dims[0] * filterDesc->dims[1] * filterDesc->dims[2] * filterDesc->dims[3] *
          sizeof(float);
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccTransFilter(const ccFilterDescriptor_t wDesc, const void *w, ccFilterDescriptor_t yDesc, void *y,
                         uint32_t ySizeInBytes) {
  y = const_cast<void *>(w);

  return CC_STATUS_SUCCESS;
}

ccStatus_t ccCreateTensorDescriptor(ccTensorDescriptor_t *tensorDesc) {
  *tensorDesc = new tagCcTensor();
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetTensor4dDescriptor(ccTensorDescriptor_t tensorDesc, ccTensorFormat_t format, ccDataType_t dataType,
                                   int32_t n, int32_t c, int32_t h, int32_t w) {
  if (CC_TENSOR_NHWC == format) {
    tensorDesc->dim_buf[0] = n;
    tensorDesc->dim_buf[1] = h;
    tensorDesc->dim_buf[2] = w;
    tensorDesc->dim_buf[3] = c;
  } else {
    tensorDesc->dim_buf[0] = n;
    tensorDesc->dim_buf[1] = c;
    tensorDesc->dim_buf[2] = h;
    tensorDesc->dim_buf[3] = w;
  }
  tensorDesc->dim_cnt = 4;
  tensorDesc->data_type = dataType;
  tensorDesc->format = format;
  tensorDesc->data_size = n * c * h * w * sizeof(dataType);
  return CC_STATUS_SUCCESS;
}
ccStatus_t ccGetTensorSizeInBytes(const ccTensorDescriptor_t tensorDesc, uint32_t *size) {
  if ((NULL == tensorDesc) || (NULL == size)) {
    return CC_STATUS_BAD_PARAM;
  }
  *size = tensorDesc->data_size;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccGetTensorMemorySizeInBytes(const ccTensorDescriptor_t tensorDesc, uint32_t *size) {
  *size = tensorDesc->data_size;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccCreateFilterDescriptor(ccFilterDescriptor_t *filterDesc) {
  *filterDesc = new tagCcFilter();
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetFilter4dDescriptor(ccFilterDescriptor_t filterDesc, ccTensorFormat_t format, ccDataType_t dataType,
                                   int32_t k, int32_t c, int32_t h, int32_t w) {
  filterDesc->dims.push_back(k);
  filterDesc->dims.push_back(c);
  filterDesc->dims.push_back(h);
  filterDesc->dims.push_back(w);

  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetFilterFractalDescriptor(ccFilterDescriptor_t filterDesc, ccTensorFormat_t format, ccDataType_t dataType,
                                        int32_t k, int32_t c, int32_t h, int32_t w) {
  filterDesc->dims.push_back(k);
  filterDesc->dims.push_back(c);
  filterDesc->dims.push_back(h);
  filterDesc->dims.push_back(w);

  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetStream(ccHandle_t handle, rtStream_t streamId) { return CC_STATUS_SUCCESS; }
ccStatus_t ccCreatePoolingMaskDescriptor(ccTensorDescriptor_t *poolingMaskDesc) {
  *poolingMaskDesc = new tagCcTensor();
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetPoolingMaskTensorDescriptor(ccTensorDescriptor_t tensorDesc, ccTensorFormat_t format,
                                            ccDataType_t dataType, int32_t n, int32_t c, int32_t h, int32_t w,
                                            int32_t windowH, int32_t windowW) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetFilter6dDescriptor(ccTensorDescriptor_t filterDesc, ccTensorFormat_t format, ccDataType_t dataType,
                                   int32_t c1, int32_t h, int32_t w, int32_t n, int32_t co, int32_t c0) {
  return CC_STATUS_SUCCESS;
}

/// @ingroup dnn
/// @brief get the format and dimcnt of GeTensor
/// @param [in] tensorDesc   descriptor of tensor
/// @param [in|out] format   point to format
/// @return ccStatus_t
ccStatus_t ccGetTensorFormat(const ccTensorDescriptor_t tensorDesc, ccTensorFormat_t *format) {
  *format = tensorDesc->format;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccTransTensor(const ccTensorDescriptor_t xDesc, const void *x, const ccTensorDescriptor_t yDesc, void *y,
                         uint32_t ySizeInBytes) {
  return CC_STATUS_SUCCESS;
}
void cceSysInit() {}

bool compilerStubFree() { return true; }

bool compilerStubInit() { return true; }

ccStatus_t ccSetInt8Filter4dDescriptor(ccFilterDescriptor_t filterDesc, ccTensorFormat_t format, ccDataType_t dataType,
                                       int32_t k, int32_t c, int32_t h, int32_t w, ccDataType_t outputDataType) {
  filterDesc->dims.push_back(k);
  filterDesc->dims.push_back(c);
  filterDesc->dims.push_back(h);
  filterDesc->dims.push_back(w);

  return CC_STATUS_SUCCESS;
}
ccStatus_t ccSetTensorNdDescriptor(ccTensorDescriptor_t tensorDesc, ccDataType_t dataType, int32_t dimCnt,
                                   int32_t dimA[]) {
  tensorDesc->data_type = dataType;
  tensorDesc->data_size = sizeof(dataType);
  for (int32_t i = 0; i < dimCnt; i++) {
    tensorDesc->data_size = tensorDesc->data_size * dimA[i];
  }
  tensorDesc->format = CC_TENSOR_ND;
  return CC_STATUS_SUCCESS;
}

ccStatus_t CceProfilingConfig(const char *target, const char *job_ctx, uint32_t flag) { return CC_STATUS_SUCCESS; }
ccStatus_t ccSetTensorRealDimCnt(ccTensorDescriptor_t tensorDesc, int32_t realDimCnt) {
  if (tensorDesc != NULL && tensorDesc != nullptr) {
    tensorDesc->realDimCnt = realDimCnt;
  }
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccGetTensorRealDimCnt(ccTensorDescriptor_t tensorDesc, int32_t *realDimCnt) {
  *realDimCnt = tensorDesc->realDimCnt;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetQuantizeFactors(ccQuantizeDescriptor_t quantizeInfo, ccScaleValueMode_t scaleValMode,
                                const uint16_t *scale, const uint16_t *offset, const uint8_t *offsetPad) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetReQuantizeFactors(ccQuantizeDescriptor_t quantizeInfo, ccScaleValueMode_t scaleValMode,
                                  const uint16_t *scaleRq, const uint16_t *nextLayerOffset, const int32_t *offsetw) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetDeQuantizeFactors(ccQuantizeDescriptor_t quantizeInfo, ccScaleValueMode_t scaleValMode,
                                  const uint16_t *scaleDq, const int32_t *offsetw) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetQuantizeAlgoAndScaleType(ccQuantizeDescriptor_t quantizeInfo, ccQuantizeAlgo_t quantAlgo,
                                         ccScaleType_t scaleType, bool reluFlag) {
  return CC_STATUS_SUCCESS;
}
ccStatus_t ccPrintTimeStat() { return CC_STATUS_SUCCESS; }
ccStatus_t ccSetModelId(ccHandle_t handle, uint32_t modelId) { return CC_STATUS_SUCCESS; }

ccStatus_t ccGetKernelContext(rtStream_t streamId, ccOpContext &opContext) {
  if (streamId == nullptr) {
    opContext.kernelType = ccKernelType::TE;
  } else {
    opContext.kernelType = ccKernelType::CCE_AI_CORE;
    opContext.opId = 1;
    opContext.kernelFuncId = 1;
    opContext.isFlowtable = true;
    opContext.opCount = 1;
    opContext.opIndex2[0] = 0;
  }

  return CC_STATUS_SUCCESS;
}

ccStatus_t ccUpdateKernelArgs(ccOpContext &opContext, uint64_t dataBaseAddr, uint64_t weightBaseAddr,
                              uint64_t variableBaseAddr, void *argsAddr, uint64_t argsSize, void *l2ctrlAddr) {
  return CC_STATUS_SUCCESS;
}
ccStatus_t ccGetKernelArgsAddrs(ccOpContext &opContext, void *argsAddr, uint64_t argsSize, void *l2ctrlAddr,
                                std::vector<ccOpAddrsInfo> &opAddrsInfo) {
  // cce
  ccOpAddrsInfo tmpOpAddrsInfo;
  uint64_t tmpInput = (uint64_t)global_mem_base;
  tmpOpAddrsInfo.addrPos = &tmpInput;
  tmpOpAddrsInfo.addrData = tmpInput;
  opAddrsInfo.push_back(tmpOpAddrsInfo);

  uint64_t tmpOutput = (uint64_t)(global_mem_base + 5476352);
  tmpOpAddrsInfo.addrPos = &tmpOutput;
  tmpOpAddrsInfo.addrData = tmpOutput;
  opAddrsInfo.push_back(tmpOpAddrsInfo);
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetKernelArgs(std::vector<ccOpAddrsInfo> &dateInfo) { return CC_STATUS_SUCCESS; }
}  // namespace cce
// ccFusion no namespace
ccStatus_t ccFusionStart(ccHandle_t handle, uint32_t graphId, uint32_t initFlag, CceFusionMemCfg_t memCfg) {
  return CC_STATUS_SUCCESS;
}

//???ccFusion ????namespace cce??
ccStatus_t ccFusionStart(ccHandle_t handle, uint32_t graphId, uint32_t initFlag, uint32_t addrChangeFlag) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccFusionEnd(ccHandle_t handle, uint32_t graphId) { return CC_STATUS_SUCCESS; }

ccStatus_t ccFusionTaskEnd(ccHandle_t handle, uint32_t graphId) { return CC_STATUS_SUCCESS; }

ccStatus_t ccKernelLaunchRepeat(ccHandle_t handle) { return CC_STATUS_SUCCESS; }

ccStatus_t ccKernelDelete(ccHandle_t handle) { return CC_STATUS_SUCCESS; }

ccStatus_t cce::ccSetTensorFormat(cce::tagCcTensor *, cce::tagCcTensorFormat) { return CC_STATUS_SUCCESS; }

namespace fusion {
uint32_t BufferFusion(std::shared_ptr<ge::ComputeGraph>, std::shared_ptr<ge::ComputeGraph>, bool) { return 0; }

uint32_t BufferFusionTrain(std::shared_ptr<ge::ComputeGraph>, std::shared_ptr<ge::ComputeGraph>) { return 0; }

uint32_t GraphFusionTrain(ge::ComputeGraphPtr origGraph, ge::ComputeGraphPtr fusionGraph) { return 0; }
}  // namespace fusion
namespace fusion {
using namespace ge;

uint32_t Fusion(ComputeGraphPtr modelGraph, ComputeGraphPtr fusionGraph, kScopeNodeMap_t &tefusionMap) {
  OpDescPtr op_def_a = std::make_shared<OpDesc>();
  op_def_a->SetName("reduction_nd");
  op_def_a->SetType("reduction_nd");

  GeTensorDescPtr v_input_desc = std::make_shared<GeTensorDesc>();
  op_def_a->AddInputDesc(*v_input_desc);

  vector<int64_t> v_input;
  v_input.push_back(0);
  op_def_a->SetInputOffset(v_input);

  GeTensorDesc input_desc = op_def_a->GetInputDesc(0);
  input_desc.SetFormat(FORMAT_NCHW);
  input_desc.SetDataType(DT_FLOAT);
  input_desc.SetShape(GeShape({1, 3, 5, 5}));
  ge::TensorUtils::SetSize(input_desc, 192);
  ge::TensorUtils::SetRealDimCnt(input_desc, 4);

  GeTensorDescPtr output_desc = std::make_shared<GeTensorDesc>();
  op_def_a->AddOutputDesc(*output_desc);

  output_desc->SetFormat(FORMAT_NCHW);
  output_desc->SetDataType(DT_FLOAT);
  output_desc->SetShape(GeShape({1, 3, 5}));
  ge::TensorUtils::SetSize(*output_desc, 96);
  ge::TensorUtils::SetRealDimCnt(*output_desc, 3);

  OpDescPtr op_def_b = std::make_shared<OpDesc>();
  op_def_b->SetName("transdata_1");
  op_def_b->SetType("TransData");

  int stream_num = 1;
  int flag = 0;

  // make_graph_nd(graph);
  NodePtr node_a = fusionGraph->AddNode(op_def_a);
  NodePtr node_b = fusionGraph->AddNode(op_def_b);

  GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
  int32_t a = 1;
  int32_t b = 2;

  AttrUtils::SetInt(op_def_a, "fusion_scope", a);
  AttrUtils::SetInt(op_def_b, "fusion_scope", b);

  vector<NodePtr> node_list1;
  node_list1.push_back(node_a);
  vector<NodePtr> node_list2;
  node_list2.push_back(node_b);
  tefusionMap[1] = node_list1;
  tefusionMap[2] = node_list2;

  return FUSION_STATUS_SUCCESS;
}

uint32_t FusionTaskBuild(cce::ccHandle_t ccHandle, ge::ComputeGraphPtr fusionGraph, ge::Buffer &buffer,
                         ModelRes &modelRes, std::vector<TaskDef> &task_def_list_) {
  TaskDef taskDefTemp;
  task_def_list_.push_back(taskDefTemp);

  return FUSION_STATUS_SUCCESS;
}
uint32_t GraphFusion(ge::ComputeGraphPtr origGraph, ge::ComputeGraphPtr fusionGraph) {
  *fusionGraph = *origGraph;
  return FUSION_STATUS_SUCCESS;
}

void FusionTaskBuildComplete(std::vector<ccHandle_t> cchandleList) { return; }

}  // namespace fusion

ccStatus_t cce::ccSetTensorDescriptorQuantizeParam(ccTensorDescriptor_t tensorDesc,
                                                   const ccVecQuantizePara_t *vecQuantizePara) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t cce::ccSetAllOffsetQuantizeFactors(ccQuantizeDescriptor_t quantizeInfo, const uint8_t *offsetW,
                                              const uint8_t *offsetD, const uint16_t *scaleReq,
                                              const uint16_t *offsetDNext) {
  return CC_STATUS_SUCCESS;
}
