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

#include "graph/model.h"
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include "debug/ge_attr_define.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/model_serialize.h"
#include "proto/ge_ir.pb.h"
#include "utils/attr_utils.h"
#include "utils/ge_ir_utils.h"

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;

namespace {
const int DEFAULT_VERSION = 1;
const int ACCESS_PERMISSION_BITS = 0400;
}  // namespace

namespace ge {
void Model::Init() {
  (void)AttrUtils::SetInt(this, ATTR_MODEL_MEMORY_SIZE, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_STREAM_NUM, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_EVENT_NUM, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_LABEL_NUM, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_WEIGHT_SIZE, 0);
  (void)AttrUtils::SetStr(this, ATTR_MODEL_TARGET_TYPE, TARGET_TYPE_MINI);
  version_ = 0;
}

Model::Model() {
  attrs_.InitDefault();
  Init();
}

Model::Model(const string &name, const string &custom_version)
    : name_(name), version_(DEFAULT_VERSION), platform_version_(custom_version) {
  attrs_.InitDefault();
  Init();
}

string Model::GetName() const { return name_; }

void Model::SetName(const string &name) { name_ = name; }

uint32_t Model::GetVersion() const { return version_; }

string Model::GetPlatformVersion() const { return platform_version_; }

void Model::SetGraph(const ge::Graph &graph) { graph_ = graph; }

Graph Model::GetGraph() const { return graph_; }

graphStatus Model::Save(Buffer &buffer, bool is_dump) const {
  ModelSerialize serialize;
  buffer = serialize.SerializeModel(*this, is_dump);
  return buffer.GetSize() > 0 ? GRAPH_SUCCESS : GRAPH_FAILED;
}

void Model::SetAttr(const ProtoAttrMapHelper &attrs) { attrs_ = attrs; }

graphStatus Model::Load(const uint8_t *data, size_t len, Model &model) {
  ModelSerialize serialize;
  model = serialize.UnserializeModel(data, len);
  return model.IsValid() ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::SaveToFile(const string &file_name) const {
  Buffer buffer;
  if ((*this).Save(buffer) != GRAPH_SUCCESS) {
    GE_LOGE("save to file fail.");
    return GRAPH_FAILED;
  }
  // Write file
  ge::proto::ModelDef ge_proto;
  if (buffer.GetData() != nullptr) {
    std::string str((const char *)buffer.GetData(), buffer.GetSize());
    if (!ge_proto.ParseFromString(str)) {
      return GRAPH_FAILED;
    }
    char real_path[PATH_MAX] = {0x00};
    if (strlen(file_name.c_str()) >= PATH_MAX) {
      return GRAPH_FAILED;
    }
    if (realpath(file_name.c_str(), real_path) == nullptr) {
      GELOGI("file %s does not exit, it will be created.", file_name.c_str());
    }
    int fd = open(real_path, O_WRONLY | O_CREAT | O_TRUNC, ACCESS_PERMISSION_BITS);
    if (fd < 0) {
      GELOGE(GRAPH_FAILED, "open file failed, file path [%s], %s ", real_path, strerror(errno));
      return GRAPH_FAILED;
    }
    bool ret = ge_proto.SerializeToFileDescriptor(fd);
    if (!ret) {
      GELOGE(GRAPH_FAILED, "SerializeToFileDescriptor failed");
      if (close(fd) != 0) {
        GELOGE(GRAPH_FAILED, "close file descriptor fail.");
        return GRAPH_FAILED;
      }
      return GRAPH_FAILED;
    }
    if (close(fd) != 0) {
      GELOGE(GRAPH_FAILED, "close file descriptor fail.");
      return GRAPH_FAILED;
    }
    if (!ret) {
      GELOGE(GRAPH_FAILED, "function [SerializeToFileDescriptor] failed");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus Model::Load(ge::proto::ModelDef &model_def) {
  ModelSerialize serialize;
  *this = serialize.UnserializeModel(model_def);
  return this->IsValid() ? GRAPH_SUCCESS : GRAPH_FAILED;
}

bool Model::IsValid() const { return graph_.IsValid(); }

graphStatus Model::LoadFromFile(const string &file_name) {
  char real_path[PATH_MAX] = {0x00};
  if (strlen(file_name.c_str()) >= PATH_MAX) {
    return GRAPH_FAILED;
  }
  if (realpath(file_name.c_str(), real_path) == nullptr) {
    GELOGE(GRAPH_FAILED, "file %s does not exit, can not load.", file_name.c_str());
    return GRAPH_FAILED;
  }
  int fd = open(real_path, O_RDONLY);
  if (fd < 0) {
    GELOGE(GRAPH_FAILED, "open file failed, %s", strerror(errno));
    return GRAPH_FAILED;
  }

  ge::proto::ModelDef model_def;
  bool ret = model_def.ParseFromFileDescriptor(fd);
  if (!ret) {
    GELOGE(GRAPH_FAILED, "ParseFromFileDescriptor failed");
    if (close(fd) != 0) {
      GELOGE(GRAPH_FAILED, "close file descriptor fail.");
      return GRAPH_FAILED;
    }
    return GRAPH_FAILED;
  }
  if (close(fd) != 0) {
    GELOGE(GRAPH_FAILED, "close file descriptor fail.");
    return GRAPH_FAILED;
  }
  if (!ret) {
    GELOGE(GRAPH_FAILED, "function [ParseFromFileDescriptor] failed");
    return GRAPH_FAILED;
  }
  return Load(model_def);
}

ProtoAttrMapHelper Model::MutableAttrMap() { return attrs_; }

ConstProtoAttrMapHelper Model::GetAttrMap() const {
  return ConstProtoAttrMapHelper(attrs_.GetProtoOwner(), attrs_.GetProtoMsg());
}
}  // namespace ge
