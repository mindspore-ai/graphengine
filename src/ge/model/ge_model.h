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

#ifndef GE_MODEL_GE_MODEL_H_
#define GE_MODEL_GE_MODEL_H_

#include <securec.h>
#include <map>
#include <memory>
#include <string>
#include "common/tbe_kernel_store.h"
#include "framework/common/debug/log.h"
#include "framework/common/fmk_error_codes.h"
#include "graph/buffer.h"
#include "graph/graph.h"
#include "proto/task.pb.h"

namespace ge {
const uint32_t INVALID_MODEL_ID = 0xFFFFFFFFUL;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeModel : public AttrHolder {
 public:
  GeModel();
  ~GeModel() = default;
  GeModel(const GeModel &other) = delete;
  GeModel &operator=(const GeModel &other) = delete;

  const Graph &GetGraph() const;
  std::shared_ptr<domi::ModelTaskDef> GetModelTaskDefPtr() const;
  const TBEKernelStore &GetTBEKernelStore() const;
  Buffer GetWeight() const;

  std::string GetName() const;
  uint32_t GetVersion() const;
  std::string GetPlatformVersion() const;
  uint8_t GetPlatformType() const;

  void SetGraph(const Graph &graph);
  void SetModelTaskDef(const std::shared_ptr<domi::ModelTaskDef> &task);
  void SetTBEKernelStore(const TBEKernelStore &tbe_kernal_store);
  void SetWeight(const Buffer &weights_buffer);

  void SetName(const std::string &name);
  void SetVersion(uint32_t version);
  void SetPlatformVersion(const std::string &platform_version);
  void SetPlatformType(uint8_t platform_type);

  void SetAttr(const ProtoAttrMapHelper &attrs);

  ProtoAttrMapHelper MutableAttrMap() override;

  using AttrHolder::GetAllAttrNames;
  using AttrHolder::GetAllAttrs;
  using AttrHolder::SetAttr;

  void SetModelId(uint32_t model_id) { model_id_ = model_id; }
  uint32_t GetModelId() const { return model_id_; }

 protected:
  ConstProtoAttrMapHelper GetAttrMap() const override;

 private:
  void Init();

  ProtoAttrMapHelper attrs_; /*lint !e148*/

  Graph graph_;                              /*lint !e148*/
  std::shared_ptr<domi::ModelTaskDef> task_; /*lint !e148*/
  TBEKernelStore tbe_kernal_store_;
  Buffer weights_buffer_; /*lint !e148*/

  std::string name_;
  uint32_t version_ = {0};
  std::string platform_version_;
  uint8_t platform_type_ = {0};
  uint32_t model_id_ = INVALID_MODEL_ID;
};
};  // namespace ge
using GeModelPtr = std::shared_ptr<ge::GeModel>;
#endif  // GE_MODEL_GE_MODEL_H_
