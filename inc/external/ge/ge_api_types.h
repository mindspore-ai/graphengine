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

#ifndef INC_EXTERNAL_GE_GE_API_TYPES_H_
#define INC_EXTERNAL_GE_GE_API_TYPES_H_

#include <cstdint>
#include <string>
#include <vector>

namespace ge {
// Option key: graph run mode
const char *const OPTION_GRAPH_RUN_MODE = "ge.graphRunMode";

// Option key: ome init
const char *const OPTION_EXEC_SESSION_ID = "ge.exec.sessionId";
const char *const OPTION_EXEC_DEVICE_ID = "ge.exec.deviceId";
const char *const OPTION_EXEC_JOB_ID = "ge.exec.jobId";
const char *const OPTION_EXEC_IS_USEHCOM = "ge.exec.isUseHcom";
const char *const OPTION_EXEC_RANK_ID = "ge.exec.rankId";
const char *const OPTION_EXEC_POD_NAME = "ge.exec.podName";
const char *const OPTION_EXEC_DEPLOY_MODE = "ge.exec.deployMode";
const char *const OPTION_EXEC_RANK_TABLE_FILE = "ge.exec.rankTableFile";
const char *const GE_AICPU_FLAG = "ge.aicpuFlag";
const char *const OPTION_EXEC_EXTERN_PLUGIN_PATH = "ge.soLoadPath";
const char *const OPTION_EXEC_ENABLE_DUMP = "ge.exec.enableDump";
const char *const OPTION_EXEC_DUMP_PATH = "ge.exec.dumpPath";
// Hccl flag, if ge.exec.hcclFlag =1, it means load plugin for opskernel, else:ge.exec.hcclFlag =0
const char *const OPTION_EXEC_HCCL_FLAG = "ge.exec.hcclFlag";
const char *const OPTION_EXEC_ATOMIC_FLAG = "ge.exec.enable_atomic";

// Option key: memory init
const char *const GRAPH_MEMORY_MAX_SIZE = "ge.graphMemoryMaxSize";
const char *const VARIABLE_MEMORY_MAX_SIZE = "ge.variableMemoryMaxSize";

// Configure stream num by Session constructor options param,
// its value should be int32_t type, default value is "1"
const std::string STREAM_NUM = "ge.streamNum";

// Configure add head stream to model.
// its value should be "0" or "1", default value is "0"
const std::string HEAD_STREAM = "ge.headStream";

// Configure perf level by Session constructor options param,
// its value please see enum PerfLevel, default value is "4"
const std::string PERF_LEVEL = "ge.perfLevel";

// Configure encrypt mode by Session constructor options param,
// its value should be int32_t type, default value is "-1"
const std::string ENCRYPT_MODE = "ge.encryptMode";

// configure ek file by Session constructor options param,
// its value should be file path, default value is ""
const std::string EK_FILE = "ge.ekFile";

// Configure cert file by Session constructor options param,
// its value should be file path, default value is ""
const std::string CERT_FILE = "ge.certFile";

// Configure hw key file by Session constructor options param,
// its value should be file path, default value is ""
const std::string HW_KEY_FILE = "ge.hwKeyFile";

// Configure private file by Session constructor options param,
// its value should be file path, default value is ""
const std::string PRIVATE_KEY_FILE = "ge.privateKeyFile";

// Configure framework type by Session constructor options param,
// its value please see enum FrameworkType, default value is "3"
const std::string FRAMEWORK_TYPE = "ge.frameworkType";

// Configure calibration info file by Session constructor options param,
// its value should be file path, default value is ""
const std::string CALIBRATION_CONF_FILE = "ge.calibrationConfFile";

// Configure insert op info file by Session constructor options param,
// its value should be file path, default value is ""
const std::string INSERT_OP_FILE = "ge.insertOpFile";

// Configure output node name by Session constructor options param,
// its value should be std::string type, default value is ""
const std::string OUTPUT_NODE_NAME = "ge.outputNodeName";

// Configure weight compress flag by Session constructor options param,
// its value should be "0" or "1", default value is "0"
const std::string COMPRESS_FLAG = "ge.compressFlag";

const std::string ATUO_PRECISION_FLAG = "ge.exec.auto_mix_precision";

// Configure single op flag for FE
// its value should be "0" or "1", default value is "0"
const std::string SINGLE_OP_FLAG = "ge.exec.single_op";

// Configure train flag by Session constructor options param,
// its value should be "0" or "1", default value is "0"
const std::string TRAIN_FLAG = "ge.trainFlag";

// Configure run flag by Session constructor options param,
// its value should be "0" or "1", default value is "0"
const std::string RUN_FLAG = "ge.runFlag";

// Configure run flag by Session constructor options param,
// its value should be "0" or "1", default value is "0"
// this option is to enable local framework op feature
const std::string LOCAL_FMKOP_FLAG = "ge.enabledLocalFmkop";

// Configure run flag by Session constructor options param,
// its value should be a path
// this option is to obtain the TBE op plugin path
const std::string TBE_PLUGIN_PATH_FLAG = "ge.TBE_plugin_path";

// Configure run flag by Session constructor options param,
// its value should be a path
// this option is to obtain the DDK Version info
const std::string DDK_VERSION_FLAG = "ge.DDK_version";

// Configure run flag by Session constructor options param,
// its value should be a path
// this option is to obtain fe flag
const std::string GE_FE_FLAG = "ge.feFlag";

// Configure stream max parallel num only by Session constructor options param,
// its value should be stream:int, such as "DNN_V100:2,DNN_HCCL:3",
// default value is "1", such as "DNN_V100:1,DNN_HCCL:1"
// this option is to obtain stream max parallel num
const std::string STREAM_MAX_PARALLEL_NUM = "ge.streamMaxParallelNum";

// congigure outputDatatype to setting net output type
const std::string OUTPUT_DATATYPE = "ge.outputDatatype";

// configure whether to enable hcom parallel by session constructor options param,
// its value should be "0" or "1", default value is "0"
const std::string HCOM_PARALLEL = "ge.hcomParallel";

// Configure auto tune mode, this option only take effect while AUTO_TUNE_FLAG is Y,
// example: GA|RL, support configure multiple, split by |
const std::string AUTO_TUNE_MODE = "ge.autoTuneMode";

// Configure core type "VectorEngine", default value is "AIcoreEngine"
const std::string CORE_TYPE = "ge.engineType";

// Configure soc version , example: "Ascend310"
const std::string SOC_VERSION = "ge.socVersion";

// Save original model
const std::string SAVE_ORIGINAL_MODEL = "ge.saveOriginalModel";

// Save original model file name
const std::string ORIGINAL_MODEL_FILE = "ge.originalModelFile";

const char *const OPTION_GE_MAX_DUMP_FILE_NUM = "ge.maxDumpFileNum";
const char *const OPTION_GE_MAX_DUMP_FILE_SIZE = "ge.maxDumpFileSize";
const char *const OPTION_GE_MAX_DUMP_OP_NUM = "ge.maxDumpOpNum";

// Graph run mode
enum GraphRunMode { PREDICTION = 0, TRAIN };

// Data description
struct DataDesc {
  void *data = nullptr;  // data address
  uint32_t length = 0;   // data size
  bool isDataSupportMemShare = false;
};

// Input/Output shape description
struct ShapeDesc {
  int64_t num = 0;
  int64_t channel = 0;
  int64_t height = 0;
  int64_t width = 0;
  std::vector<int64_t> dims;
};

// Input/Output tensor info
struct TensorInfo {
  uint32_t dataType;    // data type
  DataDesc data;        // tensor data
  ShapeDesc shapeInfo;  // tensor shape
};
}  // namespace ge

#endif  // INC_EXTERNAL_GE_GE_API_TYPES_H_
