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

#ifndef INC_EXTERNAL_GE_GE_API_TYPES_H_
#define INC_EXTERNAL_GE_GE_API_TYPES_H_

#include <cstdint>
#include <string>
#include <vector>
#include <set>
#include <functional>
#include <memory>
#include "graph/tensor.h"
#include "graph/types.h"

#ifndef GE_API_TYPES_DEF
#define GE_API_TYPES_DEF
namespace ge {
// Option key: graph run mode
const char_t *const OPTION_GRAPH_RUN_MODE = "ge.graphRunMode";
const char_t *const OPTION_DEVICE_TYPE = "ge.deviceType";
// Option key: ome init
const char_t *const OPTION_EXEC_SESSION_ID = "ge.exec.sessionId";
const char_t *const OPTION_EXEC_DEVICE_ID = "ge.exec.deviceId";
const char_t *const OPTION_EXEC_JOB_ID = "ge.exec.jobId";
const char_t *const OPTION_EXEC_IS_USEHCOM = "ge.exec.isUseHcom";
const char_t *const OPTION_EXEC_IS_USEHVD = "ge.exec.isUseHvd";
const char_t *const OPTION_EXEC_RANK_ID = "ge.exec.rankId";
const char_t *const OPTION_EXEC_POD_NAME = "ge.exec.podName";
const char_t *const OPTION_EXEC_DEPLOY_MODE = "ge.exec.deployMode";
const char_t *const OPTION_EXEC_RANK_TABLE_FILE = "ge.exec.rankTableFile";
const char_t *const GE_AICPU_FLAG = "ge.aicpuFlag";
const char_t *const OPTION_EXEC_EXTERN_PLUGIN_PATH = "ge.soLoadPath";
// Dump flag and para
const char_t *const OPTION_EXEC_ENABLE_DUMP = "ge.exec.enableDump";
const char_t *const OPTION_EXEC_DUMP_PATH = "ge.exec.dumpPath";
const char_t *const OPTION_EXEC_DUMP_STEP = "ge.exec.dumpStep";
const char_t *const OPTION_EXEC_DUMP_MODE = "ge.exec.dumpMode";
const char_t *const OPTION_EXEC_DUMP_DATA = "ge.exec.dumpData";
const char_t *const OPTION_EXEC_DUMP_LAYER = "ge.exec.dumpLayer";
const char_t *const OPTION_EXEC_ENABLE_DUMP_DEBUG = "ge.exec.enableDumpDebug";
const char_t *const OPTION_EXEC_DUMP_DEBUG_MODE = "ge.exec.dumpDebugMode";
const char_t *const OPTION_EXEC_ENABLE_INCRE_BUILD = "ge.exec.enableIncreBuild";
const char_t *const OPTION_EXEC_INCRE_BUILD_CACHE_PATH = "ge.exec.increBuildCachePath";
const char_t *const OPTION_EXEC_ENABLE_EXCEPTION_DUMP = "ge.exec.enable_exception_dump";
const char_t *const OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES = "ge.exec.enableScopeFusionPasses";
const char_t *const OPTION_EXEC_PROFILING_FPPONIT_OPTIONS = "ge.exec.profilingFpPointOptions";
const char_t *const OPTION_EXEC_PROFILING_BPPONIT_OPTIONS = "ge.exec.profilingBpPointOptions";
// profiling flag
const char_t *const OPTION_EXEC_PROFILING_MODE = "ge.exec.profilingMode";
const char_t *const OPTION_EXEC_PROFILING_OPTIONS = "ge.exec.profilingOptions";
// Hccl flag, if ge.exec.hcclFlag =1, it means load plugin for opskernel, else:ge.exec.hcclFlag =0
const char_t *const OPTION_EXEC_HCCL_FLAG = "ge.exec.hcclFlag";
const char_t *const OPTION_EXEC_ATOMIC_FLAG = "ge.exec.enable_atomic";
const char_t *const OPTION_EXEC_DISABLE_REUSED_MEMORY = "ge.exec.disableReuseMemory";
const char_t *const OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION = "ge.exec.isTailingOptimization";
// Dynamic input flag. ge.exec.dynamicInput=1, means enable dynaimc input,
// ge.exec.dynamicGraphExecuteMode, dynamic_execute[default]
const char_t *const OPTION_EXEC_DYNAMIC_INPUT = "ge.exec.dynamicInput";
const char_t *const OPTION_EXEC_DYNAMIC_EXECUTE_MODE = "ge.exec.dynamicGraphExecuteMode";
const char_t *const OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE = "ge.exec.dataInputsShapeRange";
const char_t *const OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR = "ge.exec.enableCopyOutputAddr";
const char_t *const OPTION_EXEC_GRAPH_EXEC_TIMEOUT = "ge.exec.graphExecTimeout";
const char_t *const OPTION_EXEC_OPTIMIZE_SHAPE = "ge.exec.dataInputsOptimizeShape";

// Option key: memory init
const char_t *const GRAPH_MEMORY_MAX_SIZE = "ge.graphMemoryMaxSize";
const char_t *const VARIABLE_MEMORY_MAX_SIZE = "ge.variableMemoryMaxSize";
const char_t *const OPTION_EXEC_REUSE_ZERO_COPY_MEMORY = "ge.exec.reuseZeroCopyMemory";
const char_t *const OPTION_INPUT_REUSE_MEM_INDEXES = "ge.exec.inputReuseMemIndexes";
const char_t *const OPTION_OUTPUT_REUSE_MEM_INDEXES = "ge.exec.outputReuseMemIndexes";
const char_t *const OPTION_GRAPH_IO_MEM_ALLOC_MODE = "ge.exec.graphIOMemAllocMode";

const std::string ATOMIC_CLEAN_POLICY = "ge.exec.atomicCleanPolicy";
const std::string MEMORY_OPTIMIZATION_POLICY = "ge.exec.memoryOptimizationPolicy";
const std::string STATIC_MEMORY_POLICY = "ge.exec.staticMemoryPolicy";
const char_t *const OPTION_FEATURE_BASE_REFRESHABLE = "ge.featureBaseRefreshable";
const char_t *const OPTION_CONST_LIFECYCLE = "ge.constLifecycle";

const char_t *const OPTION_EXEC_LOGICAL_DEVICE_CLUSTER_DEPLOY_MODE = "ge.exec.logicalDeviceClusterDeployMode";
const char_t *const OPTION_EXEC_LOGICAL_DEVICE_ID = "ge.exec.logicalDeviceId";
const char_t *const OPTION_EXEC_MODEL_DEPLOY_MODE = "ge.exec.modelDeployMode";
const char_t *const OPTION_EXEC_MODEL_DEPLOY_DEVICELIST = "ge.exec.modelDeployDevicelist";
const char_t *const OPTION_EXEC_ENABLE_FUSION = "ge.exec.enableFusion";

const std::string OPTION_EXEC_CM_CHIEF_IP = "ge.cmChiefIp";
const std::string OPTION_EXEC_CM_CHIEF_PORT = "ge.cmChiefPort";
const std::string OPTION_EXEC_CM_CHIEF_DEVICE = "ge.cmChiefWorkerDevice";
const std::string OPTION_EXEC_CM_WORKER_IP = "ge.cmWorkerIp";
const std::string OPTION_EXEC_CM_WORKER_SIZE = "ge.cmWorkerSize";

const std::string OPTION_NAME_MAP = "ge.optionNameMap";

const std::string OPTION_EXEC_STREAM_SYNC_TIMEOUT = "stream_sync_timeout";
const std::string OPTION_EXEC_EVENT_SYNC_TIMEOUT = "event_sync_timeout";

// Option key: embedding service
const char_t *const OPTION_EXEC_PS_ID = "ge.exec.psId";
const char_t *const OPTION_EXEC_CLUSTER_SPEC = "ge.exec.clusterSpec";
const char_t *const OPTION_EXEC_RANK_TABLE_ADDR = "ge.exec.rankTableAddr";
const char_t *const OPTION_EXEC_ROLE_TABLE_ADDR = "ge.exec.roleTableAddr";
const char_t *const OPTION_EXEC_RANK_TABLE_LEN = "ge.exec.rankTableLen";
const char_t *const OPTION_EXEC_ROLE_TABLE_LEN = "ge.exec.roleTableLen";
const char_t *const OPTION_EXEC_WORKER_NUM = "ge.exec.workerNum";
const char_t *const OPTION_EXECUTE_TIMES = "ge.execute_times";
const char_t *const OPTION_MAX_KEY_NUM = "ge.max_num";
const char_t *const OPTION_EMBEDDING_DIM = "ge.embedding_dim";
const char_t *const OPTION_USE_COUNTER_FILTER = "ge.use_counter_filter";

// Option key: Offload
constexpr char_t const OPTION_EXEC_RANK_MAP[] = "ge.exec.rankMap";

// Option key: host env os & cpu
const char_t *const OPTION_HOST_ENV_OS = "ge.host_env_os";
const char_t *const OPTION_HOST_ENV_CPU = "ge.host_env_cpu";

// config value should be a exist dir
const char_t *const OPTION_GRAPH_COMPILER_CACHE_DIR = "ge.graph_compiler_cache_dir";
// graph unique key
const char_t *const OPTION_GRAPH_KEY = "ge.graph_key";

// optimizations to disable, split by comma
const char_t *const OPTION_DISABLE_OPTIMIZATIONS = "ge.disableOptimizations";
const char_t *const ENABLE_GRAPH_PARALLEL = "ge.enableGraphParallel";
const char_t *const GRAPH_PARALLEL_OPTION_PATH = "ge.graphParallelOptionPath";
const std::string DISTRIBUTED_CLUSTER_BUILD = "ge.distributed_cluster_build";
const std::string MODEL_RELATION_CONFIG = "ge.offline_model_relation";
const std::string CLUSTER_CONFIG = "ge.cluster_config";
const std::string OPTION_HCCL_COMPILER_OFFLINE = "ge.offline_hccl_compile";

// option for screen log
const char_t *const OPTION_SCREEN_PRINT_MODE = "ge.screen_print_mode";

namespace configure_option {
const char_t *const STREAM_NUM = "ge.streamNum";
const char_t *const HEAD_STREAM = "ge.headStream";
const char_t *const PERF_LEVEL = "ge.perfLevel";
const char_t *const ENCRYPT_MODE = "ge.encryptMode";
const char_t *const EK_FILE = "ge.ekFile";
const char_t *const CERT_FILE = "ge.certFile";
const char_t *const HW_KEY_FILE = "ge.hwKeyFile";
const char_t *const PRIVATE_KEY_FILE = "ge.privateKeyFile";
const char_t *const FRAMEWORK_TYPE = "ge.frameworkType";
const char_t *const CALIBRATION_CONF_FILE = "ge.calibrationConfFile";
const char_t *const INSERT_OP_FILE = "ge.insertOpFile";
const char_t *const OUTPUT_NODE_NAME = "ge.outputNodeName";
const char_t *const COMPRESS_FLAG = "ge.compressFlag";
const char_t *const PRECISION_MODE = "ge.exec.precision_mode";
const char_t *const PRECISION_MODE_V2 = "ge.exec.precision_mode_v2";
const char_t *const SINGLE_OP_FLAG = "ge.exec.single_op";
const char_t *const TRAIN_FLAG = "ge.trainFlag";
const char_t *const RUN_FLAG = "ge.runFlag";
const char_t *const LOCAL_FMKOP_FLAG = "ge.enabledLocalFmkop";
const char_t *const TBE_PLUGIN_PATH_FLAG = "ge.TBE_plugin_path";
const char_t *const DDK_VERSION_FLAG = "ge.DDK_version";
const char_t *const GE_FE_FLAG = "ge.feFlag";
const char_t *const STREAM_MAX_PARALLEL_NUM = "ge.streamMaxParallelNum";
const char_t *const OUTPUT_DATATYPE = "ge.outputDatatype";
const char_t *const OP_SELECT_IMPL_MODE = "ge.opSelectImplmode";
const char_t *const OPTYPELIST_FOR_IMPLMODE = "ge.optypelistForImplmode";
const char_t *const HCOM_PARALLEL = "ge.hcomParallel";
const char_t *const AUTO_TUNE_MODE = "ge.autoTuneMode";
const char_t *const SOC_VERSION = "ge.socVersion";
const char_t *const VIRTUAL_TYPE = "ge.virtual_type";
const char_t *const CORE_TYPE = "ge.engineType";
const char_t *const AICORE_NUM = "ge.aicoreNum";
const char_t *const L1_FUSION = "ge.l1Fusion";
const char_t *const BUFFER_OPTIMIZE = "ge.bufferOptimize";
const char_t *const ENABLE_SMALL_CHANNEL = "ge.enableSmallChannel";
const char_t *const ENABLE_COMPRESS_WEIGHT = "ge.enableCompressWeight";
const char_t *const FUSION_SWITCH_FILE = "ge.fusionSwitchFile";
const char_t *const SAVE_ORIGINAL_MODEL = "ge.saveOriginalModel";
const char_t *const ORIGINAL_MODEL_FILE = "ge.originalModelFile";
const char_t *const INPUT_FP16_NODES = "ge.INPUT_NODES_SET_FP16";
const char_t *const OP_DEBUG_LEVEL = "ge.opDebugLevel";
const char_t *const PERFORMANCE_MODE = "ge.performance_mode";
const char_t *const SHAPE_GENERALIZED_BUILD_MODE = "ge.shape_generalized_build_mode";
const char_t *const MODIFY_MIXLIST = "ge.exec.modify_mixlist";
const char_t *const OP_PRECISION_MODE = "ge.exec.op_precision_mode";
const char_t *const ALLOW_HF32 = "ge.exec.allow_hf32";
const char_t *const CUSTOMIZE_DTYPES = "ge.customizeDtypes";
const char_t *const COMPRESSION_OPTIMIZE_CONF = "ge.compressionOptimizeConf";
const char_t *const OP_DEBUG_CONFIG = "op_debug_config";
const char_t *const ATOMIC_CLEAN_POLICY = "ge.exec.atomicCleanPolicy";
const char_t *const STATIC_MEMORY_POLICY = "ge.exec.staticMemoryPolicy";
const char_t *const EXTERNAL_WEIGHT = "ge.externalWeight";
static const char_t *const DETERMINISTIC = "ge.deterministic";
}  // namespace configure_option
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

const std::string PRECISION_MODE = "ge.exec.precision_mode";

const std::string PRECISION_MODE_V2 = "ge.exec.precision_mode_v2";

const std::string TUNE_DEVICE_IDS = "ge.exec.tuneDeviceIds";

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

// congigure opSelectImplmode to setting op select implmode
const std::string OP_SELECT_IMPL_MODE = "ge.opSelectImplmode";

// congigure optypelist_for_implmode to setting which op use implmode
const std::string OPTYPELIST_FOR_IMPLMODE = "ge.optypelistForImplmode";

// configure whether to enable hcom parallel by session constructor options param,
// its value should be "0" or "1", default value is "0"
const std::string HCOM_PARALLEL = "ge.hcomParallel";

// configure whether to use dynamic batch size
const char_t *const kDynamicBatchSize = "ge.dynamicBatchSize";

// configure threshold of fusion data size for communication op
const std::string FUSION_TENSOR_SIZE = "ge.fusionTensorSize";

const std::string INPUT_SHAPE = "ge.inputShape";

const std::string DYNAMIC_NODE_TYPE = "ge.dynamicNodeType";
// configure whether to use dynamic image size
const char_t *const kDynamicImageSize = "ge.dynamicImageSize";

// Configure whether to use dynamic dims
const char_t *const kDynamicDims = "ge.dynamicDims";

// Configure auto tune mode, this option only take effect while AUTO_TUNE_FLAG is Y,
// example: GA|RL, support configure multiple, split by |
const std::string AUTO_TUNE_MODE = "ge.autoTuneMode";

// Configure soc version , example: "Ascend310"
const std::string SOC_VERSION = "ge.socVersion";

// configure whether to enable virtualization,
// its value should be "0" or "1", default value is "0"
const std::string VIRTUAL_TYPE = "ge.virtual_type";

// Configure core type "VectorEngine", default value is "AIcoreEngine"
const std::string CORE_TYPE = "ge.engineType";

// Configure graph exclude one or more engines
const std::string EXCLUDE_ENGINES = "ge.exec.exclude_engines";

// Configure AICORE NUM
const std::string AICORE_NUM = "ge.aicoreNum";

// Configure L1FUSION
const std::string L1_FUSION = "ge.l1Fusion";

// Configure l1,l2,and others optimize option
const std::string BUFFER_OPTIMIZE = "ge.bufferOptimize";

// Configure Small Channel flag
const std::string ENABLE_SMALL_CHANNEL = "ge.enableSmallChannel";

// Configure Compress Weight flag
const std::string ENABLE_COMPRESS_WEIGHT = "ge.enableCompressWeight";

// Configure Sparse Matrix Weight flag
const std::string ENABLE_SPARSE_MATRIX_WEIGHT = "ge.enableSparseMatrixWeight";

// Configure fusion switch file path
const std::string FUSION_SWITCH_FILE = "ge.fusionSwitchFile";

// Configure compression optimize file path
const std::string COMPRESSION_OPTIMIZE_CONF = "ge.compressionOptimizeConf";

// Configure customize dtypes path
const std::string CUSTOMIZE_DTYPES = "ge.customizeDtypes";

// Configure switch for op debug config such as op memory detection
const std::string OP_DEBUG_CONFIG = "op_debug_config";

// Save original model
const std::string SAVE_ORIGINAL_MODEL = "ge.saveOriginalModel";

// Save original model file name
const std::string ORIGINAL_MODEL_FILE = "ge.originalModelFile";

const char_t *const OPTION_GE_MAX_DUMP_FILE_NUM = "ge.maxDumpFileNum";
const char_t *const OPTION_GE_MAX_DUMP_FILE_SIZE = "ge.maxDumpFileSize";
const char_t *const OPTION_GE_MAX_DUMP_OP_NUM = "ge.maxDumpOpNum";

// Configure for print op pass
// Its value should be "0" or "1", default value is "1"
const char_t *const ENABLE_PRINT_OP_PASS = "ge.enablePrintOpPass";

// Configure operator compilation path
// Its value should be file path, default value is "./"
const char_t *const DEBUG_DIR = "ge.debugDir";

// Configure switch for op status check such as overflow
// Its value should be true of flase
const char_t *const STATUS_CHECK = "ge.status_check";

// Configure operator compiler cache path
// Its value should be file path, default value is "./"
const char_t *const OP_COMPILER_CACHE_DIR = "ge.op_compiler_cache_dir";

// Configure operator compiler cache mode
// Its value should be "disable", "enable" or "force", default value is "disable"
const char_t *const OP_COMPILER_CACHE_MODE = "ge.op_compiler_cache_mode";

// Configure build model type. FE need this option to judge inner model or not
// Its value should be "true" or "false"
const char_t *const BUILD_INNER_MODEL = "ge.build_inner_model";

// Configure whether to use single stream.
// Its value should be "true" or "false", default value is "false"
const char_t *const ENABLE_SINGLE_STREAM = "ge.enableSingleStream";

// Configure input fp16 nodes
const std::string INPUT_FP16_NODES = "ge.INPUT_NODES_SET_FP16";

// Configure debug level, its value should be 0(default), 1 or 2.
// 0: close debug; 1: open TBE compiler; 2: open ccec compiler
const std::string OP_DEBUG_LEVEL = "ge.opDebugLevel";

// Configure model bank path
const std::string MDL_BANK_PATH_FLAG = "ge.mdl_bank_path";

// Configure display_model_info flag
const std::string DISPLAY_MODEL_INFO = "ge.display_model_info";

// Configure op bank path
const std::string OP_BANK_PATH_FLAG = "ge.op_bank_path";
const std::string OP_BANK_UPDATE_FLAG = "ge.op_bank_update";

// Configure for fix hcombroadcast format.
// when config model multi, broadcast format should be fixed
// 0: data multi; 1: model multi;
const std::string HCOM_MULTI_MODE = "ge.hcomMultiMode";

// atc and ir option
const char_t *const INPUT_SHAPE_RANGE = "input_shape_range";

// Configure express high compile performance or high execute performance
// normal: no need to compile, used saved .o files directly
// high: need to recompile, high execute performance mode
const std::string PERFORMANCE_MODE = "ge.performance_mode";

// For selecting the mode of shape generalization when build graph.
// shape_generalized: Shape will be generalized during graph build.
// shape_precise: Shape will not be generalized, use precise shape.
const std::string SHAPE_GENERALIZED_BUILD_MODE = "ge.shape_generalized_build_mode";

const std::string JIT_COMPILE = "ge.jit_compile";

const std::string MODIFY_MIXLIST = "ge.exec.modify_mixlist";

const std::string OP_PRECISION_MODE = "ge.exec.op_precision_mode";

const std::string ALLOW_HF32 = "ge.exec.allow_hf32";

const std::string OP_WAIT_TIMEOUT = "ge.exec.opWaitTimeout";

const std::string OP_EXECUTE_TIMEOUT = "ge.exec.opExecuteTimeout";

const char_t *const FILE_CONSTANT_PATH = "ge.exec.value_bins";

// Configure whether convert const to fileconstant and save weight to file.
// Its value should be "0" or "1", default value is "0".
const std::string EXTERNAL_WEIGHT = "ge.externalWeight";

const std::string DETERMINISTIC = "ge.deterministic";

constexpr char_t EVENT[] = "ge.event";

// Graph run mode
enum GraphRunMode : std::int32_t { PREDICTION = 0, TRAIN };
// Input/Output tensor info
struct InputTensorInfo {
  uint32_t data_type;         // data type
  std::vector<int64_t> dims;  // shape description
  void *data;                 // tensor data
  int64_t length;             // tensor length
};

struct OutputTensorInfo {
  uint32_t data_type{};             // data type
  std::vector<int64_t> dims;        // shape description
  std::unique_ptr<uint8_t[]> data;  // tensor data
  int64_t length{};                 // tensor length
  OutputTensorInfo() : dims({}), data(nullptr) {}
  OutputTensorInfo(OutputTensorInfo &&out) noexcept = default;
  OutputTensorInfo &operator=(OutputTensorInfo &&out)& noexcept {
    if (this != &out) {
      data_type = out.data_type;
      dims = out.dims;
      data = std::move(out.data);
      length = out.length;
    }
    return *this;
  }
  OutputTensorInfo(const OutputTensorInfo &) = delete;
  OutputTensorInfo &operator=(const OutputTensorInfo &)& = delete;
};

struct ModelDistibuteDesc {
  uint32_t logic_device_number;
};

using Status = uint32_t;
using RunAsyncCallback = std::function<void(Status, std::vector<ge::Tensor> &)>;

// for ir build
namespace ir_option {
static const char_t *const INPUT_FORMAT = "input_format";
static const char_t *const INPUT_SHAPE = "input_shape";
static const char_t *const INPUT_SHAPE_RANGE = ge::INPUT_SHAPE_RANGE;
static const char_t *const OP_NAME_MAP = "op_name_map";
static const char_t *const IS_DYNAMIC_INPUT = "is_dynamic_input";
static const char_t *const IS_INPUT_ADJUST_HW_LAYOUT = "is_input_adjust_hw_layout";
static const char_t *const IS_OUTPUT_ADJUST_HW_LAYOUT = "is_output_adjust_hw_layout";
static const char_t *const ENABLE_SCOPE_FUSION_PASSES = "enable_scope_fusion_passes";
static const char_t *const OUTPUT = "output";
static const char_t *const DYNAMIC_BATCH_SIZE = kDynamicBatchSize;
static const char_t *const DYNAMIC_IMAGE_SIZE = kDynamicImageSize;
static const char_t *const DYNAMIC_DIMS = kDynamicDims;
static const char_t *const INSERT_OP_FILE = ge::INSERT_OP_FILE.c_str();
static const char_t *const PRECISION_MODE = ge::PRECISION_MODE.c_str();
static const char_t *const PRECISION_MODE_V2 = ge::PRECISION_MODE_V2.c_str();
static const char_t *const TUNE_DEVICE_IDS = ge::TUNE_DEVICE_IDS.c_str();
static const char_t *const EXEC_DISABLE_REUSED_MEMORY = ge::OPTION_EXEC_DISABLE_REUSED_MEMORY;
static const char_t *const AUTO_TUNE_MODE = ge::AUTO_TUNE_MODE.c_str();
static const char_t *const CORE_TYPE = ge::CORE_TYPE.c_str();
static const char_t *const SOC_VERSION = ge::SOC_VERSION.c_str();
static const char_t *const VIRTUAL_TYPE = ge::VIRTUAL_TYPE.c_str();
static const char_t *const ENABLE_SINGLE_STREAM = ge::ENABLE_SINGLE_STREAM;
static const char_t *const AICORE_NUM = ge::AICORE_NUM.c_str();
static const char_t *const FUSION_SWITCH_FILE = ge::FUSION_SWITCH_FILE.c_str();
static const char_t *const ENABLE_SMALL_CHANNEL = ge::ENABLE_SMALL_CHANNEL.c_str();
static const char_t *const OP_SELECT_IMPL_MODE = ge::OP_SELECT_IMPL_MODE.c_str();
static const char_t *const OUTPUT_TYPE = ge::OUTPUT_DATATYPE.c_str();
static const char_t *const BUFFER_OPTIMIZE = ge::BUFFER_OPTIMIZE.c_str();
static const char_t *const ENABLE_COMPRESS_WEIGHT = ge::ENABLE_COMPRESS_WEIGHT.c_str();
static const char_t *const SPARSITY = ge::ENABLE_SPARSE_MATRIX_WEIGHT.c_str();
static const char_t *const COMPRESS_WEIGHT_CONF = "compress_weight_conf";
static const char_t *const OUT_NODES = ge::OUTPUT_NODE_NAME.c_str();
static const char_t *const INPUT_FP16_NODES = ge::INPUT_FP16_NODES.c_str();
static const char_t *const LOG_LEVEL = "log";
static const char_t *const OPTYPELIST_FOR_IMPLMODE = ge::OPTYPELIST_FOR_IMPLMODE.c_str();
static const char_t *const DEBUG_DIR = ge::DEBUG_DIR;
static const char_t *const OP_COMPILER_CACHE_DIR = ge::OP_COMPILER_CACHE_DIR;
static const char_t *const OP_COMPILER_CACHE_MODE = ge::OP_COMPILER_CACHE_MODE;
static const char_t *const BUILD_INNER_MODEL = ge::BUILD_INNER_MODEL;
static const char_t *const MDL_BANK_PATH = ge::MDL_BANK_PATH_FLAG.c_str();
static const char_t *const OP_BANK_PATH = ge::OP_BANK_PATH_FLAG.c_str();
static const char_t *const OP_BANK_UPDATE = ge::OP_BANK_UPDATE_FLAG.c_str();
static const char_t *const OP_DEBUG_LEVEL = ge::OP_DEBUG_LEVEL.c_str();
static const char_t *const PERFORMANCE_MODE = ge::PERFORMANCE_MODE.c_str();
static const char_t *const SHAPE_GENERALIZED_BUILD_MODE = ge::SHAPE_GENERALIZED_BUILD_MODE.c_str();
static const char_t *const MODIFY_MIXLIST = ge::MODIFY_MIXLIST.c_str();
static const char_t *const OP_PRECISION_MODE = ge::OP_PRECISION_MODE.c_str();
static const char_t *const ALLOW_HF32 = ge::ALLOW_HF32.c_str();
static const char_t *const CUSTOMIZE_DTYPES = "ge.customizeDtypes";
static const char_t *const COMPRESSION_OPTIMIZE_CONF = "ge.compressionOptimizeConf";
static const char_t *const INPUT_DATA_NAMES = "input_data_names";
static const char_t *const OP_DEBUG_CONFIG = "op_debug_config";
static const char_t *const ATOMIC_CLEAN_POLICY = "ge.exec.atomicCleanPolicy";
static const char_t *const EXTERNAL_WEIGHT = ge::EXTERNAL_WEIGHT.c_str();
static const char_t *const EXCLUDE_ENGINES = ge::EXCLUDE_ENGINES.c_str();
static const char_t *const DETERMINISTIC = ge::DETERMINISTIC.c_str();
static const char_t *const DISTRIBUTED_CLUSTER_BUILD = ge::DISTRIBUTED_CLUSTER_BUILD.c_str();
static const char_t *const MODEL_RELATION_CONFIG = ge::MODEL_RELATION_CONFIG.c_str();
static const char_t *const CLUSTER_CONFIG = ge::CLUSTER_CONFIG.c_str();
static const char_t *const ENABLE_GRAPH_PARALLEL = "ge.enableGraphParallel";
static const char_t *const GRAPH_PARALLEL_OPTION_PATH = "ge.graphParallelOptionPath";
// for interface: aclgrphBuildModel
#ifdef __GNUC__
const std::set<std::string> ir_builder_suppported_options = {INPUT_FORMAT,
                                                             INPUT_SHAPE,
                                                             INPUT_SHAPE_RANGE,
                                                             OP_NAME_MAP,
                                                             DYNAMIC_BATCH_SIZE,
                                                             DYNAMIC_IMAGE_SIZE,
                                                             DYNAMIC_DIMS,
                                                             INSERT_OP_FILE,
                                                             OP_PRECISION_MODE,
                                                             ALLOW_HF32,
                                                             PRECISION_MODE,
                                                             PRECISION_MODE_V2,
                                                             TUNE_DEVICE_IDS,
                                                             EXEC_DISABLE_REUSED_MEMORY,
                                                             AUTO_TUNE_MODE,
                                                             OUTPUT_TYPE,
                                                             OUT_NODES,
                                                             INPUT_FP16_NODES,
                                                             LOG_LEVEL,
                                                             OP_DEBUG_LEVEL,
                                                             DEBUG_DIR,
                                                             OP_COMPILER_CACHE_DIR,
                                                             OP_COMPILER_CACHE_MODE,
                                                             MDL_BANK_PATH,
                                                             OP_BANK_PATH,
                                                             OP_BANK_UPDATE,
                                                             PERFORMANCE_MODE,
                                                             SHAPE_GENERALIZED_BUILD_MODE,
                                                             MODIFY_MIXLIST,
                                                             CUSTOMIZE_DTYPES,
                                                             BUILD_INNER_MODEL,
                                                             OP_DEBUG_CONFIG,
                                                             EXCLUDE_ENGINES,
                                                             EXTERNAL_WEIGHT,
                                                             DISTRIBUTED_CLUSTER_BUILD,
                                                             MODEL_RELATION_CONFIG,
                                                             ENABLE_GRAPH_PARALLEL,
                                                             GRAPH_PARALLEL_OPTION_PATH};

// for interface: aclgrphParse
const std::set<std::string> ir_parser_suppported_options = {
  INPUT_FP16_NODES, IS_INPUT_ADJUST_HW_LAYOUT, IS_OUTPUT_ADJUST_HW_LAYOUT, OUTPUT,
  OUT_NODES, ENABLE_SCOPE_FUSION_PASSES, INPUT_DATA_NAMES, INPUT_SHAPE};

// for interface: aclgrphBuildInitialize
const std::set<std::string> global_options = {CORE_TYPE,
                                              SOC_VERSION,
                                              VIRTUAL_TYPE,
                                              BUFFER_OPTIMIZE,
                                              ENABLE_COMPRESS_WEIGHT,
                                              COMPRESS_WEIGHT_CONF,
                                              SPARSITY,
                                              PRECISION_MODE,
                                              PRECISION_MODE_V2,
                                              ALLOW_HF32,
                                              TUNE_DEVICE_IDS,
                                              EXEC_DISABLE_REUSED_MEMORY,
                                              AUTO_TUNE_MODE,
                                              ENABLE_SINGLE_STREAM,
                                              AICORE_NUM,
                                              FUSION_SWITCH_FILE,
                                              ENABLE_SMALL_CHANNEL,
                                              OP_SELECT_IMPL_MODE,
                                              OPTYPELIST_FOR_IMPLMODE,
                                              OP_DEBUG_LEVEL,
                                              DEBUG_DIR,
                                              OP_COMPILER_CACHE_DIR,
                                              OP_COMPILER_CACHE_MODE,
                                              MODIFY_MIXLIST,
                                              COMPRESSION_OPTIMIZE_CONF,
                                              OP_DEBUG_CONFIG,
                                              DETERMINISTIC,
                                              CLUSTER_CONFIG};
#endif
}  // namespace ir_option
}  // namespace ge
#endif // GE_API_TYPES_DEF
#endif  // INC_EXTERNAL_GE_GE_API_TYPES_H_
