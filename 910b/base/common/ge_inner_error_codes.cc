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

#include "framework/common/ge_inner_error_codes.h"

namespace ge {
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_PARAM_INVALID, "Parameter invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_NOT_INIT, "GE executor not initialized yet.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "Model file path invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID, "Model id invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID, "Data size of model invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID, "Model addr invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID, "Queue id of model invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED, "The model loaded repeatedly.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID, "Dynamic input addr invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID, "Dynamic input size invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID, "Dynamic batch size invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_AIPP_BATCH_EMPTY, "AIPP batch parameter empty.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_AIPP_NOT_EXIST, "AIPP parameter not exist.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_AIPP_MODE_INVALID, "AIPP mode invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_OP_TASK_TYPE_INVALID, "Task type invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID, "Kernel type invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_PLGMGR_PATH_INVALID, "Plugin path is invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_FORMAT_INVALID, "Format is invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_SHAPE_INVALID, "Shape is invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DATATYPE_INVALID, "Datatype is invalid.");

GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_MEMORY_ALLOCATION, "Memory allocation error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "Failed to operate memory.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED, "Failed to allocate operate memory.");

GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_INTERNAL_ERROR, "Internal error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_LOAD_MODEL, "Load model error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED, "Failed to load model partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED, "Failed to load weight partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED, "Failed to load task partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED, "Failed to load op kernel partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_RELEASE_MODEL_DATA, "Failed to release the model data.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_COMMAND_HANDLE, "Command handle error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_GET_TENSOR_INFO, "Get tensor info error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_UNLOAD_MODEL, "Load model error.");
} // namespace ge