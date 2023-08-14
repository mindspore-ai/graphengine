/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef INC_FRAMEWORK_EXECUTOR_GE_C_EXECUTOR_H_
#define INC_FRAMEWORK_EXECUTOR_GE_C_EXECUTOR_H_

#include <stdint.h>
#include "external/ge/ge_error_codes.h"
#include "framework/executor_c/ge_executor_types.h"
#include "framework/executor_c/types.h"
#if defined(__cplusplus)
extern "C" {
#endif

GE_FUNC_VISIBILITY Status GeInitialize();
GE_FUNC_VISIBILITY Status GeFinalize();
GE_FUNC_VISIBILITY Status GetModelDescInfo(uint32_t modelId, ModelInOutInfo *info);

GE_FUNC_VISIBILITY Status GetMemAndWeightSize(const char *fileName, size_t *workSize, size_t *weightSize);
GE_FUNC_VISIBILITY Status ExecModel(uint32_t modelId, ExecHandleDesc *execDesc, bool basync, InputData *inputData,
                                    OutputData *outputData);
GE_FUNC_VISIBILITY Status LoadModelFromData(uint32_t *modelId, const ModelData *modelData, void *weightPtr,
                                            size_t weightSize);
GE_FUNC_VISIBILITY Status LoadDataFromFile(const char *modelPath, ModelData *data);
GE_FUNC_VISIBILITY void FreeModelData(ModelData *data);
GE_FUNC_VISIBILITY Status UnloadModel(uint32_t modelId);
GE_FUNC_VISIBILITY Status GetModelDescInfoFromMem(const ModelData *modelData, ModelInOutInfo *info);
GE_FUNC_VISIBILITY void DestoryModelInOutInfo(ModelInOutInfo *info);
GE_FUNC_VISIBILITY Status GeDbgInit(const char *configPath);
GE_FUNC_VISIBILITY Status GeDbgDeInit(void);
GE_FUNC_VISIBILITY Status GeNofifySetDevice(uint32_t chipId, uint32_t deviceId);
#if defined(__cplusplus)
}
#endif

#endif  // INC_FRAMEWORK_EXECUTOR_GE_C_EXECUTOR_H_
