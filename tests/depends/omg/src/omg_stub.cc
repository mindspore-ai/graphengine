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

#include <map>
#include <fstream>
#include <unordered_map>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "mmpa/mmpa_api.h"
#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"
#include "common/util.h"
#include "common/string_util.h"
#include "common/properties_manager.h"
#include "common/model_parser/base.h"
#include "graph/model.h"
#include "cce/dnn.h"
#include "ge/ge_api_types.h"
#include "framework/common/ge_types.h"
#include "graph/utils/op_desc_utils.h"
#include "common/profiling/profiling_manager.h"

using namespace domi;
using namespace cce;
using namespace ge;

struct PROC_PARAM {
  uint8_t *model_name;

  // ISV Ek buffer
  uint8_t *model_key;
  uint32_t model_key_len;

  // ISV  root certificate buffer
  uint8_t *root_cert;
  uint32_t root_cert_len;

  // ISV private key buffer
  uint8_t *pri_key;
  uint32_t pri_key_len;

  // Raw AI Module Image buffer
  uint8_t *ai_image;
  uint32_t ai_image_len;

  // ISV HW key buffer
  uint8_t *hw_key;
  uint32_t hw_key_len;
};

#ifdef __cplusplus
extern "C" {
#endif
using namespace ge;
namespace {
const char FMK_STATUS_FILE_DIR_ENV[] = "FMK_STATUS_FILE_DIR";
const char JOBSTATE_FILE_NAME[] = "jobstateupdate_framework";
const char HCOM_DETECT_FILE_NAME[] = "hcom_detection_result";
const char FILE_SEPARATE[] = "/";
}  // namespace

#ifdef __cplusplus
}
#endif

namespace ge {
struct GeModelPartition {
  ModelPartitionType type_ = MODEL_DEF;
  uint8_t *data_ = nullptr;
  size_t size_ = 0;

  GeModelPartition() = default;

  GeModelPartition(const GeModelPartition &partition){};

  GeModelPartition &operator=(const GeModelPartition &partition) = delete;

  ~GeModelPartition() {
    if (data_ != nullptr) {
      delete[] data_;
      data_ = nullptr;
    }
  }

  domi::Status SetData(uint8_t *data, size_t size) {
    size_ = size;
    data_ = new (std::nothrow) uint8_t[size]();
    errno_t err;
    err = memcpy_s(data_, size_, data, size);
    if (err) {
      DOMI_LOGE("[GeModel Partition] Error occur when copy GeModel Partition data.");
      return domi::FAILED;
    }
    return domi::SUCCESS;
  }

  domi::Status SetType(ModelPartitionType type) {
    type_ = type;
    return domi::SUCCESS;
  }
};
struct OmFileContext {
  vector<GeModelPartition> partition_datas_;
  vector<char> partition_table_;
  uint32_t model_data_len_;
};

class SubGraphInfo;
using SubGraphInfoPtr = std::shared_ptr<ge::SubGraphInfo>;

using GeModelPartitionPtr = std::shared_ptr<GeModelPartition>;
using ModelPtr = std::shared_ptr<ge::Model>;
class GeModel {
 public:
  explicit GeModel(const ModelPtr &modelPtr);
  ~GeModel() = default;
  GeModel(const GeModel &other) = delete;
  GeModel &operator=(const GeModel &other) = delete;

  ModelPtr GetModelPtr() const;
  domi::Status AddPartition(uint8_t *data, size_t size, ModelPartitionType type);
  domi::Status GetPartition(ModelPartitionType type, GeModelPartitionPtr &partition);
  uint8_t GetPlatformType() const;
  void SetPlatformType(const uint8_t platformType) { platform_type_ = platformType; }

 private:
  std::map<ModelPartitionType, GeModelPartitionPtr> partitions_;
  ModelPtr model_ = nullptr;
  uint8_t platform_type_ = {0};
};
using GeModelPtr = std::shared_ptr<ge::GeModel>;

GeModel::GeModel(const ModelPtr &modelPtr) { this->model_ = modelPtr; }

ModelPtr GeModel::GetModelPtr() const { return this->model_; }

uint8_t GeModel::GetPlatformType() const { return platform_type_; }

domi::Status GeModel::AddPartition(uint8_t *data, size_t size, ModelPartitionType type) {
  if (size == 0) {
    return domi::FAILED;
  }

  if (data == nullptr) {
    return domi::FAILED;
  }

  auto iter = partitions_.find(type);
  if (iter != partitions_.end()) {
    return domi::FAILED;
  }

  GeModelPartitionPtr partition = nullptr;
  GE_MAKE_SHARED(partition = std::make_shared<ge::GeModelPartition>(), return domi::FAILED);
  domi::Status ret = partition->SetType(type);
  if (ret != domi::SUCCESS) {
    return domi::FAILED;
  }
  ret = partition->SetData(data, size);
  if (ret != domi::SUCCESS) {
    return domi::FAILED;
  }

  partitions_.insert(std::pair<ModelPartitionType, GeModelPartitionPtr>(type, partition));
  return domi::SUCCESS;
}

domi::Status GeModel::GetPartition(ModelPartitionType type, GeModelPartitionPtr &partition) {
  auto iter = partitions_.find(type);
  if (iter == partitions_.end()) {
    return domi::FAILED;
  }

  partition = iter->second;
  return domi::SUCCESS;
}
class OmFileSaveHelper {
 public:
  OmFileSaveHelper();
  ~OmFileSaveHelper();
  vector<GeModelPartition> &GetModelPartitions();
  ModelPartitionTable *GetPartitionTable();
  ModelFileHeader model_header_;
  ModelFileHeader &GetModelFileHeader() { return model_header_; }
  void AddPartition(GeModelPartition &partition);

 private:
  OmFileContext context_;
};

OmFileSaveHelper::OmFileSaveHelper() {}

OmFileSaveHelper::~OmFileSaveHelper() {}

vector<GeModelPartition> &OmFileSaveHelper::GetModelPartitions() {
  static std::vector<GeModelPartition> tmp;
  return tmp;
}

ModelPartitionTable *OmFileSaveHelper::GetPartitionTable() { return nullptr; }

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void OmFileSaveHelper::AddPartition(GeModelPartition &partition) {
  context_.partition_datas_.push_back(partition);
  context_.model_data_len_ += partition.size_;
}
class ModelBuilder {
 public:
  ModelBuilder(ge::ComputeGraphPtr compute_graph, int stream_num, int mode);
  ModelBuilder(ge::ComputeGraphPtr compute_graph, const std::vector<SubGraphInfoPtr> &subgraphs,
               const std::map<std::string, int> &stream_max_parallel_num, bool hcom_parallel, int mode);
  virtual ~ModelBuilder();
  Status FuseGraphAndGetTaskInfo(ge::Model &modelDef, bool isOnlineBuild);
  Status Build();
  Status BuildModel(ge::Model &modelDef);
  Status SerializeModelDef(ge::Model &modelDef, std::shared_ptr<OmFileSaveHelper> &om_file_helper);
  Status SaveWeightsToModel(ge::Model &model);
  Status SaveDataToModel(ge::GeModel &model);
  Status SaveDataToModel(ge::Model &model, ge::GeModel &geModel);
  Status PreBuildModel();
  Status BuildModelForGetTask(ge::Model &model_def);
  Status PreBuildModelForInference();
  Status BuildModelForInferenceTask(ge::Model &model_def, bool supportGetTask);
  ge::Buffer GetWeightBuffer() const;
  Status SyncMemoryInfoToOriginalGraph();
  ge::OpDescPtr GetMultiShapeOp(void);
  void SetModelVersion(ge::Model &modelDef);
  Status AdjustCustomOp();

 public:
  ge::Buffer weight_buffer_;
};

ModelBuilder::ModelBuilder(ge::ComputeGraphPtr compute_graph, int stream_num, int mode) {
  weight_buffer_ = ge::Buffer(4100000);
}
ModelBuilder::ModelBuilder(ge::ComputeGraphPtr compute_graph, const std::vector<SubGraphInfoPtr> &subgraphs,
                           const std::map<std::string, int> &stream_max_parallel_num, bool hcom_parallel, int mode) {
  weight_buffer_ = ge::Buffer(4100000);
}
ModelBuilder::~ModelBuilder() {}
Status ModelBuilder::FuseGraphAndGetTaskInfo(ge::Model &modelDef, bool isOnlineBuild) { return domi::SUCCESS; }
Status ModelBuilder::Build() { return domi::SUCCESS; }

Status ModelBuilder::SerializeModelDef(ge::Model &modelDef, std::shared_ptr<OmFileSaveHelper> &om_file_helper) {
  return domi::SUCCESS;
}
Status ModelBuilder::SaveWeightsToModel(ge::Model &model) { return domi::SUCCESS; }

Status ModelBuilder::BuildModel(ge::Model &modelDef) { return domi::SUCCESS; }

Status ModelBuilder::SaveDataToModel(ge::GeModel &model) {
  ge::GeModelPartitionPtr partition_weight = nullptr;
  GE_MAKE_SHARED(partition_weight = std::make_shared<ge::GeModelPartition>(), return FAILED);
  uint8_t *data1 = new (std::nothrow) uint8_t[16]();
  model.AddPartition(data1, 16, ge::ModelPartitionType::WEIGHTS_DATA);

  uint8_t *data2 = new (std::nothrow) uint8_t[16]();
  model.AddPartition(data2, 16, ge::ModelPartitionType::TASK_INFO);

  uint8_t *data3 = new (std::nothrow) uint8_t[16]();
  model.AddPartition(data3, 16, ge::ModelPartitionType::TBE_KERNELS);

  model.SetPlatformType(0);

  delete[] data1;
  data1 = nullptr;
  delete[] data2;
  data2 = nullptr;
  delete[] data3;
  data3 = nullptr;

  return SUCCESS;
}

Status ModelBuilder::SaveDataToModel(ge::Model &model, ge::GeModel &geModel) { return SUCCESS; }
Status ModelBuilder::PreBuildModel() { return domi::SUCCESS; }

Status ModelBuilder::BuildModelForGetTask(ge::Model &model_def) { return domi::SUCCESS; }

Status ModelBuilder::PreBuildModelForInference() { return domi::SUCCESS; }

Status ModelBuilder::BuildModelForInferenceTask(ge::Model &model_def, bool supportGetTask) { return domi::SUCCESS; }

ge::OpDescPtr ModelBuilder::GetMultiShapeOp(void) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("A", "Data");
  vector<int64_t> dims = {1, 3, 224, 224};
  GeShape desc_shape(dims);
  GeTensorDesc desc_anchor;
  desc_anchor.SetShape(desc_shape);
  op_desc->AddOutputDesc(desc_anchor);
  vector<uint32_t> batch = {2};
  ge::AttrUtils::SetListInt(op_desc, "multi_shape_batchlist", batch);
  return op_desc;
}
void ModelBuilder::SetModelVersion(ge::Model &modelDef) { return; }

ge::Buffer ModelBuilder::GetWeightBuffer() const { return ge::Buffer(4100000); }

Status ModelBuilder::SyncMemoryInfoToOriginalGraph() { return domi::SUCCESS; }

Status ModelBuilder::AdjustCustomOp() { return SUCCESS; }
}  // namespace ge

using ProcParam = struct PROC_PARAM;
namespace domi {
#include <iostream>
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_N = 0;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_C = 1;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_H = 2;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_W = 3;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_N = 0;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_H = 1;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_W = 2;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_C = 3;

const uint32_t MODEL_FILE_MAGIC_NUM = 0x444F4D49;
const uint32_t MODEL_FILE_HEAD_LEN = 256;
const uint32_t MODEL_VERSION = 0x10000000;
const int MAX_FILE_SIZE_LIMIT = INT_MAX;  // the max length of input, is 2G
bool FC_WEIGHT_COMPRESS_FLAG = false;

bool ReadBytesFromBinaryFile(const char *file_name, char **buffer, int &length) {
  length = 10;
  *buffer = new (std::nothrow) char[10]();
  GE_CHK_BOOL_TRUE_EXEC_RET_STATUS(*buffer == nullptr, false, "new an object failed.");
  return true;
}
bool ReadProtoFromText(const char *file, google::protobuf::Message *message) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((nullptr == file || nullptr == message), return false,
                                 "incorrect parameter. nullptr == file || nullptr == message");
  string realPath = RealPath(file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(realPath.empty(), return false, "proto file path '%s' not valid", file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(GetFileLength(realPath.c_str()) == -1, return false, "file size not valid.");
  std::ifstream fs(realPath.c_str(), std::ifstream::in);

  if (!fs.is_open()) {
    DOMI_LOGE("proto file '%s' open fail.", file);
    return false;
  }
  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(ret != true,
                  GELOGI("call [google::protobuf::TextFormat::Parse] func ret fail, please check your text file."));
  fs.close();
  return ret;
}

uint64_t GetCurrentTimestap() { return 0; }

// get length of file
long GetFileLength(const std::string &input_file) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(input_file.empty(), return -1, "input_file path is null.");
  string realPath = RealPath(input_file.c_str());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(realPath.empty(), return -1, "input_file path '%s' not valid", input_file.c_str());
  unsigned long long fileLength = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(mmGetFileSize(input_file.c_str(), &fileLength) != EN_OK, return -1,
                                 "open file failed.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((fileLength <= 0), return -1, "file length <= 0, not valid.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(fileLength > MAX_FILE_SIZE_LIMIT, return -1, "file size %ld is out of limit: %d.",
                                 fileLength, MAX_FILE_SIZE_LIMIT);
  return fileLength;
}
string RealPath(const char *path) {
  string s = path;
  if (s.size() >= PATH_MAX) {
    return "";
  }
  if (s == "." || s == "1") {
    return path;
    // for insert_aipp_op unittest
  } else if (s.substr(0, 3) == "llt") {
    return path;
  } else {
    return "22";
  }
}

bool CheckInputPathValid(const string &file_path) { return true; }
bool ReadProtoFromArray(const void *data, int size, Message *proto) { return true; }

struct ModelPartition {
  ModelPartitionType type;
  uint8_t *data = 0;
  uint32_t size = 0;
};

struct OmFileContext {
  vector<ModelPartition> partition_datas_;
  vector<char> partition_table_;
  uint32_t model_data_len_;
};

class TaskDef {};

class ModelTaskDef {};

class OmFileSaveHelper {
 public:
  OmFileSaveHelper();
  ~OmFileSaveHelper();
  vector<ModelPartition> &GetModelPartitions();
  ModelPartitionTable *GetPartitionTable();
  ModelFileHeader model_header_;
  ModelFileHeader &GetModelFileHeader() { return model_header_; }
  void AddPartition(ModelPartition &partition);

 private:
  OmFileContext context_;
};

OmFileSaveHelper::OmFileSaveHelper() {}

OmFileSaveHelper::~OmFileSaveHelper() {}

vector<ModelPartition> &OmFileSaveHelper::GetModelPartitions() {
  static std::vector<ModelPartition> tmp;
  return tmp;
}

ModelPartitionTable *OmFileSaveHelper::GetPartitionTable() { return nullptr; }

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void OmFileSaveHelper::AddPartition(ModelPartition &partition) {
  context_.partition_datas_.push_back(partition);
  context_.model_data_len_ += partition.size;
}

class InsertNewOpUtil {
 public:
  InsertNewOpUtil();
  ~InsertNewOpUtil();
  Status InsertNewOps(const ComputeGraphPtr &graph);
  Status InsertAippOps(ge::ComputeGraphPtr graph, std::string &aippConfigPath);
  Status Parse(const char *conf_path);
};

InsertNewOpUtil::InsertNewOpUtil() {}

Status InsertNewOpUtil::InsertNewOps(const ComputeGraphPtr &graph) { return SUCCESS; }

Status InsertNewOpUtil::InsertAippOps(ge::ComputeGraphPtr graph, std::string &aippConfigPath) { return SUCCESS; }

Status InsertNewOpUtil::Parse(const char *conf_path) { return SUCCESS; }

Status UnloadModel(uint32_t model_id) { return SUCCESS; }
Status InitOME() { return SUCCESS; }
class GraphOptimizer {
 public:
  Status Optimize();
  Status OptimizeAfterCal();
  Status AdjustDataOpDesc();
  Status InsertTransOp();
  Status FusionFmkop();
  Status Optimize4Cloud();
  Status Optimize4FlowCtrl();
  Status OptimizeBeforeBuild();
};
Status GraphOptimizer::Optimize() { return SUCCESS; }

Status GraphOptimizer::OptimizeAfterCal() { return SUCCESS; }

Status GraphOptimizer::AdjustDataOpDesc() { return domi::SUCCESS; }

Status GraphOptimizer::InsertTransOp() { return domi::SUCCESS; }

Status GraphOptimizer::FusionFmkop() { return domi::SUCCESS; }

Status GraphOptimizer::Optimize4Cloud() { return domi::SUCCESS; }

Status GraphOptimizer::Optimize4FlowCtrl() { return domi::SUCCESS; }

Status GraphOptimizer::OptimizeBeforeBuild() { return domi::SUCCESS; }
Status BuildOnline(const ge::Graph &graph, const std::vector<ge::GeTensor> &inputs,
                   const std::map<string, string> &options, std::shared_ptr<ge::Model> &modelOut) {
  return SUCCESS;
}

Status Init(Options options) { return SUCCESS; }

Status Shutdown(Options options) { return SUCCESS; }

Status GetInputOutputDescInfo(uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                              std::vector<InputOutputDescInfo> &output_desc) {
  InputOutputDescInfo outDesc;
  outDesc.size = 512;
  outDesc.data_type = 0;
  outDesc.shape_info.dims = {1, 1, 10, 10};

  output_desc.push_back(outDesc);
  return SUCCESS;
}

class Session {
 public:
  // singleton
  static Session *Instance();
  const uint32_t &DeviceId() const;
};

const uint32_t &Session::DeviceId() const { return 0; }

Session *Session::Instance() {
  static Session instance;
  return &instance;
}
struct OmgContext {
  domiTensorFormat_t format;

  // input of format from cmd
  std::unordered_map<std::string, domiTensorFormat_t> input_nodes_format_map;
  std::vector<domiTensorFormat_t> output_formats;

  // user-designate input dims
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
  // global input dims
  std::unordered_map<std::string, std::vector<int64_t>> input_dims;

  // resolve rename op e.g. Detectionoutput:SsdDetectiontOutput
  std::map<std::string, std::string> op_conf_map;
  // save output node of network: key is op name, value = index, index is the output index of op
  std::map<std::string, std::vector<int32_t>> out_nodes_map;
  // user-designate out nodes (this is used for determing the orders)
  std::vector<std::pair<std::string, int32_t>> user_out_nodes;
  // save the path of custom_aicpu
  std::vector<std::string> aicpu_op_run_paths;
  // save ddk
  std::string ddk_version;
  // save format
  domiTensorFormat_t net_format;

  FrameworkType type;
  // RunMode run_mode;
  bool train_flag = false;

  std::string output_type;

  /// save the name of network
  /// eg:faster-rcnn, based on FirstStageProcessor after scope_fusion is faster-rcnn
  /// then reorder conv+reshape of FirstStageBoxPredictor/BoxEncodingPredictor
  /// need to delete op of reshape
  std::string net_name;
};
OmgContext &GetContext() {
  static OmgContext tmp;
  return tmp;
}

}  // namespace domi

namespace ge {
#include <iostream>
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_N = 0;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_C = 1;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_H = 2;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NCHW_DIM_W = 3;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_N = 0;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_H = 1;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_W = 2;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t NHWC_DIM_C = 3;

const uint32_t MODEL_FILE_MAGIC_NUM = 0x444F4D49;
const uint32_t MODEL_FILE_HEAD_LEN = 256;
const uint32_t MODEL_VERSION = 0x10000000;
const int MAX_FILE_SIZE_LIMIT = INT_MAX;
bool FC_WEIGHT_COMPRESS_FLAG = false;

bool ReadBytesFromBinaryFile(const char *file_name, char **buffer, int &length) {
  length = 10;
  *buffer = new (std::nothrow) char[10]();
  GE_CHK_BOOL_TRUE_EXEC_RET_STATUS(*buffer == nullptr, false, "new an object failed.");
  return true;
}
bool ReadProtoFromText(const char *file, google::protobuf::Message *message) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((nullptr == file || nullptr == message), return false,
                                 "incorrect parameter. nullptr == file || nullptr == message");
  string realPath = RealPath(file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(realPath.empty(), return false, "proto file path '%s' not valid", file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(GetFileLength(realPath.c_str()) == -1, return false, "file size not valid.");
  std::ifstream fs(realPath.c_str(), std::ifstream::in);

  if (!fs.is_open()) {
    DOMI_LOGE("proto file '%s' open fail.", file);
    return false;
  }
  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(ret != true,
                  GELOGI("call [google::protobuf::TextFormat::Parse] func ret fail, please check your text file."));
  fs.close();
  return ret;
}

uint64_t GetCurrentTimestap() { return 0; }

// get length of file
long GetFileLength(const std::string &input_file) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(input_file.empty(), return -1, "input_file path is null.");
  string realPath = RealPath(input_file.c_str());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(realPath.empty(), return -1, "input_file path '%s' not valid", input_file.c_str());
  unsigned long long fileLength = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(mmGetFileSize(input_file.c_str(), &fileLength) != EN_OK, return -1,
                                 "open file failed.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((fileLength <= 0), return -1, "file length <= 0, not valid.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(fileLength > MAX_FILE_SIZE_LIMIT, return -1, "file size %ld is out of limit: %d.",
                                 fileLength, MAX_FILE_SIZE_LIMIT);
  return fileLength;
}
string RealPath(const char *path) {
  string s = path;
  if (s.size() >= PATH_MAX) {
    return "";
  }
  if (s == "." || s == "1") {
    return path;
    // for insert_aipp_op unittest
  } else if (s.substr(0, 3) == "llt") {
    return path;
  } else {
    return "22";
  }
}

bool CheckInputPathValid(const string &file_path) { return true; }
bool ReadProtoFromArray(const void *data, int size, Message *proto) { return true; }

struct ModelPartition {
  ModelPartitionType type;
  uint8_t *data = 0;
  uint32_t size = 0;
};

class InsertNewOpUtil {
 public:
  InsertNewOpUtil();
  ~InsertNewOpUtil();
  Status InsertNewOps(const ComputeGraphPtr &graph);
  Status InsertAippOps(ge::ComputeGraphPtr graph, std::string &aippConfigPath);
  Status Parse(const char *conf_path);
};

InsertNewOpUtil::InsertNewOpUtil() {}

Status InsertNewOpUtil::InsertNewOps(const ComputeGraphPtr &graph) { return SUCCESS; }

Status InsertNewOpUtil::InsertAippOps(ge::ComputeGraphPtr graph, std::string &aippConfigPath) { return SUCCESS; }

Status InsertNewOpUtil::Parse(const char *conf_path) { return SUCCESS; }

Status InitOME() { return SUCCESS; }
class GraphOptimizer {
 public:
  Status Optimize();
  Status OptimizeAfterCal();
  Status AdjustDataOpDesc();
  Status InsertTransOp();
  Status FusionFmkop();
  Status Optimize4Cloud();
  Status Optimize4FlowCtrl();
  Status OptimizeBeforeBuild();
};
Status GraphOptimizer::Optimize() { return SUCCESS; }

Status Init(Options options) { return SUCCESS; }

Status Shutdown(Options options) { return SUCCESS; }

class Session {
 public:
  // singleton
  static Session *Instance();
  const uint32_t &DeviceId() const;
};

const uint32_t &Session::DeviceId() const { return 0; }

Session *Session::Instance() {
  static Session instance;
  return &instance;
}
struct OmgContext {
  domiTensorFormat_t format;

  // get input format from cmd
  std::unordered_map<std::string, domiTensorFormat_t> input_nodes_format_map;
  std::vector<domiTensorFormat_t> output_formats;

  // user-designate input dims
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
  // global input dims
  std::unordered_map<std::string, std::vector<int64_t>> input_dims;

  // solve rename op e.g: Detectionoutput:SsdDetectiontOutput
  std::map<std::string, std::string> op_conf_map;
  // save output node of network: key is op name, value = index, index is the output index of op
  std::map<std::string, std::vector<int32_t>> out_nodes_map;
  // user-designate out nodes (this is used for determing the orders)
  std::vector<std::pair<std::string, int32_t>> user_out_nodes;
  // save the path of cutsom_aicpu
  std::vector<std::string> aicpu_op_run_paths;
  // save ddk
  std::string ddk_version;
  // save format
  domiTensorFormat_t net_format;

  FrameworkType type;
  // RunMode run_mode;
  bool train_flag = false;

  std::string output_type;

  /// save the name of network
  /// eg:faster-rcnn, based on FirstStageProcessor after scope_fusion is faster-rcnn
  /// then reorder conv+reshape of FirstStageBoxPredictor/BoxEncodingPredictor
  /// need to delete op of reshape
  std::string net_name;
};
OmgContext &GetContext() {
  static OmgContext tmp;
  return tmp;
}

}  // namespace ge
namespace ge {
class OpUtils {
 public:
  static Status InitTensorDescriptor(const GeTensorDesc &tensor, ccTensorDescriptor_t &cc_tensor);
  static Status InitTensorDescriptor(int32_t format, int32_t data_type, const std::vector<int64_t> &dim,
                                     ccTensorDescriptor_t &cc_tensor, uint32_t real_dim_cnt);
  static void DestroyTensorDescriptor(ccTensorDescriptor_t &cc_tensor);
};
Status OpUtils::InitTensorDescriptor(const GeTensorDesc &tensor, ccTensorDescriptor_t &cc_tensor) {
  ccCreatePoolingMaskDescriptor(&cc_tensor);
  return domi::SUCCESS;
}
Status OpUtils::InitTensorDescriptor(int32_t format, int32_t data_type, const std::vector<int64_t> &dim,
                                     ccTensorDescriptor_t &cc_tensor, uint32_t real_dim_cnt) {
  Status ret = SUCCESS;
  return ret;
}

class FileSaver {
 public:
  Status SaveToFile(const string &file_path, ModelFileHeader &model_file_header,
                    ModelPartitionTable &model_partition_table, const std::vector<ModelPartition> &partitionDatas);
  Status SaveToFileWithEncrypt(const std::string file_path, const ProcParam proc_param,
                               const ModelFileHeader *model_file_header, bool check_sum);
};

Status FileSaver::SaveToFile(const string &file_path, ModelFileHeader &model_file_header,
                             ModelPartitionTable &model_partition_table,
                             const std::vector<ModelPartition> &partitionDatas) {
  return domi::SUCCESS;
}

Status FileSaver::SaveToFileWithEncrypt(const std::string file_path, const ProcParam proc_param,
                                        const ModelFileHeader *model_file_header, bool check_sum) {
  return domi::SUCCESS;
}

class ModelSaver : public FileSaver {};

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void OpUtils::DestroyTensorDescriptor(
    ccTensorDescriptor_t &cc_tensor) {
  if (nullptr != cc_tensor) {
    ccStatus_t ret = ccDestroyTensorDescriptor(&cc_tensor);
    GE_LOGE_IF(CC_STATUS_SUCCESS != ret, "ccDestroyTensorDescriptor failed. ret = %d", ret);
    cc_tensor = nullptr;
  }
}

}  // namespace ge

namespace domi {
class OpRegistrationData {};

class OpRegistry {
 public:
  static OpRegistry *Instance();
  std::vector<OpRegistrationData> registrationDatas;

  domi::ImplyType GetImplyType(const std::string &op_type);
  void GetOpTypeByImplyType(std::vector<std::string> &vec_op_type, const domi::ImplyType &imply_type);
};

OpRegistry *OpRegistry::Instance() {
  static OpRegistry instance;
  return &instance;
}

void OpRegistry::GetOpTypeByImplyType(std::vector<std::string> &vec_op_type, const domi::ImplyType &imply_type) {
  if (imply_type == domi::ImplyType::AI_CPU) {
    vec_op_type.push_back("square");
  }
}

class OpRegistrationTbe {
 public:
  static OpRegistrationTbe *Instance();

  bool Finalize(OpRegistrationData &reg_data, bool is_train);
};

OpRegistrationTbe *OpRegistrationTbe::Instance() {
  static OpRegistrationTbe instance;
  return &instance;
}

bool OpRegistrationTbe::Finalize(OpRegistrationData &reg_data, bool is_train) { return true; }
}  // namespace domi

namespace ge {
class GraphPrepare {
 private:
  Status OptimizeForPreprocess(ge::ComputeGraphPtr &compute_graph);
};

Status GraphPrepare::OptimizeForPreprocess(ge::ComputeGraphPtr &compute_graph) { return SUCCESS; }
}  // namespace ge

namespace ge {

class FlowCtrlUtil {
 public:
  static bool IsNeedTrainIteFlowCtrl(ge::ComputeGraphPtr computeGraph);
};

bool FlowCtrlUtil::IsNeedTrainIteFlowCtrl(ge::ComputeGraphPtr computeGraph) { return true; }

Status GetOriginalType(const ge::NodePtr &node, string &type) {
  type = node->GetType();
  GE_IF_BOOL_EXEC(type != FRAMEWORKOP, return SUCCESS);
  ge::AttrUtils::GetStr(node->GetOpDesc(), "original_type", type);
  return SUCCESS;
}

Status SetCycleEvent(const ge::NodePtr &node) { return SUCCESS; }

Status SetStreamLabel(const ge::NodePtr &node, const std::string &label) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = AttrUtils::CloneOpDesc(node->GetOpDesc());
  GE_CHECK_NOTNULL(tmp_desc);

  if (!AttrUtils::SetStr(tmp_desc, "_stream_label", label)) {
    DOMI_LOGE("Op :%s set ATTR_NAME_STREAM_LABEL failed", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status SetActiveLabelList(const ge::NodePtr &node, const std::vector<std::string> &label) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  // add list of active_label
  if (!AttrUtils::SetListStr(tmp_desc, "_active_label", label)) {
    DOMI_LOGE("Op: %s set ATTR_NAME_ACTIVE_LABEL_LIST failed", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status SetSwitchBranchNodeLabel(const ge::NodePtr &node, const std::string &branch_label) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  // add branch_label of switch
  if (!AttrUtils::SetStr(tmp_desc, "_switch_branch_node_label", branch_label)) {
    DOMI_LOGE("Op :%s set ATTR_NAME_SWITCH_BRANCH_NODE_LABEL failed", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status SetSwitchTrueBranchFlag(const ge::NodePtr &node, bool value) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  // add switch_true_branch_flag
  if (!AttrUtils::SetBool(tmp_desc, "_switch_true_branch_flag", value)) {
    DOMI_LOGE("Op :%s set ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG failed", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status SetOriginalNodeName(const ge::NodePtr &node, const std::string &orig_name) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  // record original_node_name
  if (!AttrUtils::SetStr(tmp_desc, "_original_node_name", orig_name)) {
    DOMI_LOGE("Op :%s set ATTR_NAME_ORIG_NODE_NAME failed", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status SetCyclicDependenceFlag(const ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  // add cyclic_dependence_flag
  if (!AttrUtils::SetBool(tmp_desc, "_cyclic_dependence_flag", true)) {
    DOMI_LOGE("Op :%s set ATTR_NAME_CYCLIC_DEPENDENCE_FLAG failed", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status SetNextIteration(const ge::NodePtr &node, const std::string &next) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);

  if (!AttrUtils::SetStr(tmp_desc, "_next_iteration_node", next)) {
    DOMI_LOGE("Op: %s set ATTR_NAME_NEXT_ITERATION failed", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace ge

namespace cce {
bool ccGetFuncState(ccFuncParamType_t type) { return true; }
}  // namespace cce

namespace ge {
Status UnloadModel(uint32_t model_id) { return SUCCESS; }

Status GetInputOutputDescInfo(uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                              vector<InputOutputDescInfo> &output_desc) {
  return SUCCESS;
}

Status DataInput(const InputData *input_data, OutputData *output_data) { return SUCCESS; }

class ModelManager {
 public:
  static std::shared_ptr<ModelManager> GetInstance();
  static void FinalizeForPtr(ModelManager *) {}
  Status DataInputTensor(uint32_t model_id, const std::vector<ge::TensorInfo> &inputs,
                         std::vector<ge::TensorInfo> &outputs);
  Status DataInput(const InputData &input_data, OutputData &output_data);
  Status GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                std::vector<InputOutputDescInfo> &output_desc);
  Status GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                std::vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &inputFormats,
                                std::vector<uint32_t> &outputFormats);
  Status GetInputOutputDescInfoForZeroCopy(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                           std::vector<InputOutputDescInfo> &output_desc,
                                           std::vector<uint32_t> &inputFormats, std::vector<uint32_t> &outputFormats);
  Status Stop(uint32_t model_id);
  Status Unload(uint32_t model_id);
  Status LoadModelOnline(uint32_t &model_id, std::shared_ptr<ge::Model> &model,
                         std::shared_ptr<ModelListener> listener);
  Status Start(uint32_t model_id);
  Status GetMaxUsedMemory(const uint32_t model_id, uint64_t &max_size);
  Status LoadModelOffline(uint32_t &model_id, const ModelData &model, std::shared_ptr<ModelListener> listener = nullptr,
                          void *dev_ptr = nullptr, size_t memsize = 0, void *weight_ptr = nullptr,
                          size_t weightsize = 0);
  Status LoadModelWithQ(uint32_t &model_id, const ModelData &model_data, const std::vector<uint32_t> &input_queue_ids,
                        const std::vector<uint32_t> &output_queue_ids);

  Status HandleCommand(const Command &command);
  Status ExecuteModel(uint32_t model_id, rtStream_t stream, bool async_mode, const InputData &input_data,
                      OutputData &output_data);
  void DestroyAicpuSession(uint64_t session_id);
};
void ModelManager::DestroyAicpuSession(uint64_t session_id) {}
std::shared_ptr<ModelManager> ModelManager::GetInstance() {
  static std::shared_ptr<ModelManager> instance_ptr =
      shared_ptr<ModelManager>(new ModelManager(), ModelManager::FinalizeForPtr);
  return instance_ptr;
}

Status ModelManager::DataInputTensor(uint32_t model_id, const std::vector<ge::TensorInfo> &inputs,
                                     std::vector<ge::TensorInfo> &outputs) {
  return SUCCESS;
}

Status ModelManager::DataInput(const InputData &input_data, OutputData &output_data) { return SUCCESS; }

Status ModelManager::GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                            std::vector<InputOutputDescInfo> &output_desc,
                                            std::vector<uint32_t> &inputFormats, std::vector<uint32_t> &outputFormats) {
  return SUCCESS;
}

Status ModelManager::GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                            std::vector<InputOutputDescInfo> &output_desc) {
  return SUCCESS;
}

Status ModelManager::GetInputOutputDescInfoForZeroCopy(const uint32_t model_id,
                                                       std::vector<InputOutputDescInfo> &input_desc,
                                                       std::vector<InputOutputDescInfo> &output_desc,
                                                       std::vector<uint32_t> &inputFormats,
                                                       std::vector<uint32_t> &outputFormats) {
  return SUCCESS;
}

Status ModelManager::Stop(uint32_t model_id) { return SUCCESS; }

Status ModelManager::Unload(uint32_t model_id) { return SUCCESS; }

Status ModelManager::LoadModelOnline(uint32_t &model_id, std::shared_ptr<ge::Model> &model,
                                     std::shared_ptr<ModelListener> listener) {
  return SUCCESS;
}

Status ModelManager::Start(uint32_t model_id) { return SUCCESS; }

Status ModelManager::GetMaxUsedMemory(const uint32_t model_id, uint64_t &max_size) { return SUCCESS; }

Status ModelManager::LoadModelOffline(uint32_t &model_id, const ModelData &model, shared_ptr<ModelListener> listener,
                                      void *dev_ptr, size_t memsize, void *weight_ptr, size_t weightsize) {
  return SUCCESS;
}

Status ModelManager::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                                    const std::vector<uint32_t> &input_queue_ids,
                                    const std::vector<uint32_t> &output_queue_ids) {
  return SUCCESS;
}

Status ModelManager::HandleCommand(const Command &command) { return SUCCESS; }

Status ModelManager::ExecuteModel(uint32_t model_id, rtStream_t stream, bool async_mode, const InputData &input_data,
                                  OutputData &output_data) {
  return SUCCESS;
}

}  // namespace ge

namespace ge {

enum JobState {
  JOBSTATE_WAITING = 1,
  JOBSTATE_RUNNING,
  JOBSTATE_KILLING,
  JOBSTATE_SUCCEED,
  JOBSTATE_FAILED,
  JOBSTATE_KILLED,
  JOBSTATE_UNKOWN
};

enum JobSubState {
  JOBSUBSTATE_ENV_INIT = 201,
  JOBSUBSTATE_ENV_FIN,
  JOBSUBSTATE_RESOUCE_ALLOC,
  JOBSUBSTATE_MODEL_COMPILE,
  JOBSUBSTATE_GRAPH_PREPARE,
  JOBSUBSTATE_GRAPH_SPLIT,
  JOBSUBSTATE_GRAPH_OPTIMIZE,
  JOBSUBSTATE_GRAPH_BUILD,
  JOBSUBSTATE_GRAPH_LOAD,
  JOBSUBSTATE_GRAPH_EXEC,
  JOBSUBSTATE_GRAPH_UNLOAD,
  JOBSUBSTATE_OTHER
};

enum ErrorModule {
  ERROR_MODULE_DRIVER = 0x01,
  ERROR_MODULE_RUNTIME = 0x04,
  ERROR_MODULE_CCE = 0x06,
  ERROR_MODULE_FMK = 0x08,
  ERROR_MODULE_HCCL = 0x12
};

class CsaInteract {
 public:
  CsaInteract &GetInstance();
  void WriteErrorCode(uint32_t module_ret_errcode, ErrorModule error_module, JobSubState job_sub_state);
  void Init(int32_t dev_index, int64_t job_id);
  Status WriteJobState(JobState job_state, JobSubState job_sub_state = JOBSUBSTATE_OTHER,
                       uint32_t module_ret_errcode = SUCCESS, ErrorModule error_module = ERROR_MODULE_FMK);
  // device index
  int32_t dev_index_;
  // job id
  int64_t job_id_;
  // is initialization complete
  bool is_init_;
  // current job state
  JobState curr_state_;
  // job state file
  std::string job_state_file_;
  // network connectivity detect file
  std::string hcom_detect_file_;
  // identification of internal errors that occurred during the training
  bool is_have_internal_error_;
};

CsaInteract &CsaInteract::GetInstance() {
  static CsaInteract instance;
  return instance;
}

void CsaInteract::Init(int32_t dev_index, int64_t job_id) {
  if (!is_init_) {
    dev_index_ = dev_index;
    job_id_ = job_id;
    string csa_path_prefix;
    if (std::getenv(FMK_STATUS_FILE_DIR_ENV) != nullptr) {
      csa_path_prefix = std::getenv(FMK_STATUS_FILE_DIR_ENV);
    }
    if (!csa_path_prefix.empty()) {
      std::string job_state_file = csa_path_prefix + std::to_string(dev_index_) + FILE_SEPARATE + JOBSTATE_FILE_NAME;
      std::string hcom_detect_file =
          csa_path_prefix + std::to_string(dev_index_) + FILE_SEPARATE + HCOM_DETECT_FILE_NAME;
      job_state_file_ = RealPath(job_state_file.c_str());
      hcom_detect_file_ = RealPath(hcom_detect_file.c_str());
    }
    is_init_ = true;
  }
}

void CsaInteract::WriteErrorCode(uint32_t module_ret_errcode, ErrorModule error_module, JobSubState job_sub_state) {}

}  // namespace ge

Status ModelParserBase::LoadFromFile(const char *model_path, const char *key, int32_t priority,
                                     ge::ModelData &model_data) {
  return SUCCESS;
}

Status CsaInteract::WriteJobState(JobState job_state, JobSubState job_sub_state, uint32_t module_ret_errcode,
                                  ErrorModule error_module) {
  return SUCCESS;
}

namespace ge {

static std::map<ge::DataType, uint32_t> DataTypeToLength = {
    {DT_BOOL, sizeof(bool)},      {DT_INT64, sizeof(int64_t)},   {DT_UINT64, sizeof(int64_t)},
    {DT_FLOAT, sizeof(float)},    {DT_INT32, sizeof(int32_t)},   {DT_UINT32, sizeof(int32_t)},
    {DT_INT8, sizeof(char)},      {DT_UINT8, sizeof(char)},      {DT_INT16, sizeof(int16_t)},
    {DT_UINT16, sizeof(int16_t)}, {DT_FLOAT16, sizeof(int16_t)}, {DT_DOUBLE, sizeof(double)},
};

class TypeUtils {
 public:
  static bool GetDataTypeLength(ge::DataType data_type, uint32_t &length);
  static bool CheckUint64MulOverflow(uint64_t a, uint32_t b);
};

bool TypeUtils::GetDataTypeLength(ge::DataType data_type, uint32_t &length) {
  auto it = DataTypeToLength.find(data_type);
  if (it != DataTypeToLength.end()) {
    length = it->second;
    return true;
  } else {
    return false;
  }
}

bool TypeUtils::CheckUint64MulOverflow(uint64_t a, uint32_t b) {
  // Not overflow
  if (a == 0) {
    return false;
  }
  if ((ULLONG_MAX / a) >= b) {
    return false;
  }
  return true;
}
}  // namespace ge
namespace domi {

///@ingroup fp16 basic parameter
///@brief   fp16 exponent bias
#define FP16_EXP_BIAS (15)

///@ingroup fp16 basic parameter
///@brief   the exponent bit length of fp16 is 5
#define FP16_EXP_LEN (5)

/// @ingroup fp16 basic parameter
/// @brief   the mantissa bit length of fp16 is 10
#define FP16_MAN_LEN (10)

/// @ingroup fp16 basic parameter
/// @brief   bit index of sign in fp16

#define FP16_SIGN_INDEX (15)

/// @ingroup fp16 basic parameter
/// @brief   sign mask of fp16         (1 00000 00000 00000)

#define FP16_SIGN_MASK (0x8000)

/// @ingroup fp16 basic parameter
/// @brief   exponent mask of fp16     (  11111 00000 00000)

#define FP16_EXP_MASK (0x7C00)

/// @ingroup fp16 basic parameter
/// @brief   mantissa mask of fp16     (        11111 11111)

#define FP16_MAN_MASK (0x03FF)

/// @ingroup fp16 basic parameter
/// @brief   hide bit of mantissa of fp16(   1 00000 00000)

#define FP16_MAN_HIDE_BIT (0x0400)

/// @ingroup fp16 basic parameter
/// @brief   maximum value            (0111 1011 1111 1111)

#define FP16_MAX (0x7BFF)

/// @ingroup fp16 basic parameter
/// @brief   minimum value            (1111 1011 1111 1111)

#define FP16_MIN (0xFBFF)

/// @ingroup fp16 basic parameter
/// @brief   absolute maximum value   (0111 1111 1111 1111)

#define FP16_ABS_MAX (0x7FFF)

/// @ingroup fp16 basic parameter
/// @brief   maximum exponent value of fp16 is 15(11111)

#define FP16_MAX_EXP (0x001F)

/// @ingroup fp16 basic parameter
/// @brief   maximum valid exponent value of fp16 is 14(11110)

#define FP16_MAX_VALID_EXP (0x001E)

/// @ingroup fp16 basic parameter
/// @brief   maximum mantissa value of fp16(11111 11111)

#define FP16_MAX_MAN (0x03FF)

/// @ingroup fp16 basic parameter
/// @brief   absolute minimum normal value of fp16
///          (E=1,M=0 D=2^(-14)=0.00006103515625)

#define FP16_MIN_NORMAL ((1.0f / (2 << 14)))

/// @ingroup fp16 basic operator
/// @brief   get sign of fp16

#define FP16_EXTRAC_SIGN(x) (((x) >> 15) & 1)

/// @ingroup fp16 basic operator
/// @brief   get exponent of fp16

#define FP16_EXTRAC_EXP(x) (((x) >> 10) & FP16_MAX_EXP)

/// @ingroup fp16 basic operator
/// @brief   get mantissa of fp16

#define FP16_EXTRAC_MAN(x) ((((x) >> 0) & 0x3FF) | (((((x) >> 10) & 0x1F) > 0 ? 1 : 0) * 0x400))

/// @ingroup fp16 basic operator
/// @brief   constructor of fp16 from sign exponent and mantissa

#define FP16_CONSTRUCTOR(s, e, m) (((s) << FP16_SIGN_INDEX) | ((e) << FP16_MAN_LEN) | ((m)&FP16_MAX_MAN))

/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is zero

#define FP16_IS_ZERO(x) (((x)&FP16_ABS_MAX) == 0)

/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is a denormalized value

#define FP16_IS_DENORM(x) ((((x)&FP16_EXP_MASK) == 0))

/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is infinite

#define FP16_IS_INF(x) (((x)&FP16_ABS_MAX) == FP16_ABS_MAX)

/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is NaN

#define FP16_IS_NAN(x) (((x & FP16_EXP_MASK) == FP16_EXP_MASK) && (x & FP16_MAN_MASK))

/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is invalid

#define FP16_IS_INVALID(x) ((x & FP16_EXP_MASK) == FP16_EXP_MASK)

/// @ingroup fp32 basic parameter
/// @brief   fp32 exponent bias

#define FP32_EXP_BIAS (127)

/// @ingroup fp32 basic parameter
/// @brief   the exponent bit length of float/fp32 is 8

#define FP32_EXP_LEN (8)

/// @ingroup fp32 basic parameter
/// @brief   the mantissa bit length of float/fp32 is 23

#define FP32_MAN_LEN (23)

/// @ingroup fp32 basic parameter
/// @brief   bit index of sign in float/fp32

#define FP32_SIGN_INDEX (31)

/// @ingroup fp32 basic parameter
/// @brief   sign mask of fp32         (1 0000 0000  0000 0000 0000 0000 000)

#define FP32_SIGN_MASK (0x80000000u)

/// @ingroup fp32 basic parameter
/// @brief   exponent mask of fp32     (  1111 1111  0000 0000 0000 0000 000)

#define FP32_EXP_MASK (0x7F800000u)

/// @ingroup fp32 basic parameter
/// @brief   mantissa mask of fp32     (             1111 1111 1111 1111 111)

#define FP32_MAN_MASK (0x007FFFFFu)

/// @ingroup fp32 basic parameter
/// @brief   hide bit of mantissa of fp32      (  1  0000 0000 0000 0000 000)

#define FP32_MAN_HIDE_BIT (0x00800000u)

/// @ingroup fp32 basic parameter
/// @brief   absolute maximum value    (0 1111 1111  1111 1111 1111 1111 111)

#define FP32_ABS_MAX (0x7FFFFFFFu)

/// @ingroup fp32 basic parameter
/// @brief   maximum exponent value of fp32 is 255(1111 1111)

#define FP32_MAX_EXP (0xFF)

/// @ingroup fp32 basic parameter
/// @brief   maximum mantissa value of fp32    (1111 1111 1111 1111 1111 111)

#define FP32_MAX_MAN (0x7FFFFF)

/// @ingroup fp32 special value judgment
/// @brief   whether a fp32 is NaN

#define FP32_IS_NAN(x) (((x & FP32_EXP_MASK) == FP32_EXP_MASK) && (x & FP32_MAN_MASK))

/// @ingroup fp32 special value judgment
/// @brief   whether a fp32 is infinite

#define FP32_IS_INF(x) (((x & FP32_EXP_MASK) == FP32_EXP_MASK) && (!(x & FP32_MAN_MASK)))

/// @ingroup fp32 special value judgment
/// @brief   whether a fp32 is a denormalized value

#define FP32_IS_DENORM(x) ((((x)&FP32_EXP_MASK) == 0))

/// @ingroup fp32 basic operator
/// @brief   get sign of fp32

#define FP32_EXTRAC_SIGN(x) (((x) >> FP32_SIGN_INDEX) & 1)

/// @ingroup fp32 basic operator
/// @brief   get exponent of fp16

#define FP32_EXTRAC_EXP(x) (((x)&FP32_EXP_MASK) >> FP32_MAN_LEN)

/// @ingroup fp32 basic operator
/// @brief   get mantissa of fp16

#define FP32_EXTRAC_MAN(x) \
  (((x)&FP32_MAN_MASK) | (((((x) >> FP32_MAN_LEN) & FP32_MAX_EXP) > 0 ? 1 : 0) * FP32_MAN_HIDE_BIT))

/// @ingroup fp32 basic operator
/// @brief   constructor of fp32 from sign exponent and mantissa

#define FP32_CONSTRUCTOR(s, e, m) (((s) << FP32_SIGN_INDEX) | ((e) << FP32_MAN_LEN) | ((m)&FP32_MAX_MAN))

/// @ingroup fp64 basic parameter
/// @brief   fp64 exponent bias

#define FP64_EXP_BIAS (1023)

/// @ingroup fp64 basic parameter
/// @brief   the exponent bit length of double/fp64 is 11

#define FP64_EXP_LEN (11)

/// @ingroup fp64 basic parameter
/// @brief   the mantissa bit length of double/fp64 is 52

#define FP64_MAN_LEN (52)

/// @ingroup fp64 basic parameter
/// @brief   bit index of sign in double/fp64 is 63

#define FP64_SIGN_INDEX (63)

/// @ingroup fp64 basic parameter
/// @brief   sign mask of fp64                 (1 000                   (total 63bits 0))

#define FP64_SIGN_MASK (0x8000000000000000LLu)

/// @ingroup fp64 basic parameter
/// @brief   exponent mask of fp64            (0 1 11111 11111  0000?-?-(total 52bits 0))

#define FP64_EXP_MASK (0x7FF0000000000000LLu)

/// @ingroup fp64 basic parameter
/// @brief   mantissa mask of fp64            (                 1111?-?-(total 52bits 1))

#define FP64_MAN_MASK (0x000FFFFFFFFFFFFFLLu)

/// @ingroup fp64 basic parameter
/// @brief   hide bit of mantissa of fp64     (               1 0000?-?-(total 52bits 0))

#define FP64_MAN_HIDE_BIT (0x0010000000000000LLu)

/// @ingroup fp64 basic parameter
/// @brief   absolute maximum value           (0 111?-?-(total 63bits 1))

#define FP64_ABS_MAX (0x7FFFFFFFFFFFFFFFLLu)

/// @ingroup fp64 basic parameter
/// @brief   maximum exponent value of fp64 is 2047(1 11111 11111)

#define FP64_MAX_EXP (0x07FF)

/// @ingroup fp64 basic parameter
/// @brief   maximum mantissa value of fp64  (111?-?-(total 52bits 1))

#define FP64_MAX_MAN (0xFFFFFFFFFFFLLu)

/// @ingroup fp64 special value judgment
/// @brief   whether a fp64 is NaN

#define FP64_IS_NAN(x) (((x & FP64_EXP_MASK) == FP64_EXP_MASK) && (x & FP64_MAN_MASK))

/// @ingroup fp64 special value judgment
/// @brief   whether a fp64 is infinite

#define FP64_IS_INF(x) (((x & FP64_EXP_MASK) == FP64_EXP_MASK) && (!(x & FP64_MAN_MASK)))

/// @ingroup integer special value judgment
/// @brief   maximum positive value of int8_t            (0111 1111)

#define INT8_T_MAX (0x7F)

/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 8 bits length  (1111 111)

#define BIT_LEN8_MAX (0xFF)

/// @ingroup integer special value judgment
/// @brief   maximum positive value of int16_t           (0111 1111 1111 1111)

#define INT16_T_MAX (0x7FFF)

/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 16 bits length (1111 1111 1111 1111)

#define BIT_LEN16_MAX (0xFFFF)

/// @ingroup integer special value judgment
/// @brief   maximum positive value of int32_t           (0111 1111 1111 1111 1111 1111 1111 1111)

#define INT32_T_MAX (0x7FFFFFFFu)

/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 32 bits length (1111 1111 1111 1111 1111 1111 1111 1111)

#define BIT_LEN32_MAX (0xFFFFFFFFu)

/// @ingroup print switch
/// @brief   print an error if input fp16 is overflow

typedef struct tagFp16 {
  uint16_t val;

 public:
  tagFp16(void) { val = 0x0u; }
  tagFp16(const uint16_t &uiVal) : val(uiVal) {}
  tagFp16(const tagFp16 &fp) : val(fp.val) {}
  float toFloat() const;
} fp16_t;
inline bool operator>(const tagFp16 &lhs, const tagFp16 &rhs) { return lhs.toFloat() > rhs.toFloat(); }
inline bool operator<(const tagFp16 &lhs, const tagFp16 &rhs) { return lhs.toFloat() < rhs.toFloat(); }
inline bool operator==(const tagFp16 &lhs, const tagFp16 &rhs) { return lhs.toFloat() == rhs.toFloat(); }
void ExtractFP16(const uint16_t &val, uint16_t *s, int16_t *e, uint16_t *m) {
  // 1.Extract
  *s = FP16_EXTRAC_SIGN(val);
  *e = FP16_EXTRAC_EXP(val);
  *m = FP16_EXTRAC_MAN(val);

  // Denormal
  if (0 == (*e)) {
    *e = 1;
  }
}

/// @ingroup fp16_t math conversion static method
/// @param [in] fpVal uint16_t value of fp16_t object
/// @brief   Convert fp16_t to float/fp32
/// @return  Return float/fp32 value of fpVal which is the value of fp16_t object

static float fp16ToFloat(const uint16_t &fpVal) {
  float ret;

  uint16_t hf_sign, hf_man;
  int16_t hf_exp;
  ExtractFP16(fpVal, &hf_sign, &hf_exp, &hf_man);

  while (hf_man && !(hf_man & FP16_MAN_HIDE_BIT)) {
    hf_man <<= 1;
    hf_exp--;
  }

  uint32_t s_ret, e_ret, m_ret, fVal;

  s_ret = hf_sign;
  if (!hf_man) {
    e_ret = 0;
    m_ret = 0;
  } else {
    e_ret = hf_exp - FP16_EXP_BIAS + FP32_EXP_BIAS;
    m_ret = hf_man & FP16_MAN_MASK;
    m_ret = m_ret << (FP32_MAN_LEN - FP16_MAN_LEN);
  }
  fVal = FP32_CONSTRUCTOR(s_ret, e_ret, m_ret);
  ret = *((float *)&fVal);

  return ret;
}

float fp16_t::toFloat() const { return fp16ToFloat(val); }
}  // namespace domi
