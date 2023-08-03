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

#ifndef GE_MODEL_GE_MODEL_H_
#define GE_MODEL_GE_MODEL_H_

#include <map>
#include <memory>
#include <string>

#include "securec.h"
#include "runtime/rt.h"
#include "common/tbe_handle_store/tbe_kernel_store.h"
#include "common/tbe_handle_store/cust_aicpu_kernel_store.h"
#include "framework/common/debug/log.h"
#include "framework/common/fmk_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/buffer.h"
#include "graph/compute_graph.h"
#include "proto/task.pb.h"

namespace ge {
class GeModel : public std::enable_shared_from_this<GeModel>, public AttrHolder {
 public:
  GeModel();
  ~GeModel() override = default;
  GeModel(const GeModel &other) = delete;
  GeModel &operator=(const GeModel &other) & = delete;

  const ComputeGraphPtr &GetGraph() const;
  void SetGraph(const ComputeGraphPtr &graph);

  std::shared_ptr<domi::ModelTaskDef> GetModelTaskDefPtr() const;
  TBEKernelStore &GetTBEKernelStore();
  const CustAICPUKernelStore &GetCustAICPUKernelStore() const;
  Buffer GetWeight() const;
  uint8_t* GetWeightData() const;
  size_t GetWeightSize() const;
  void SetWeightDataBuf(const DataBuffer &data_buffer);
  void ClearWeightDataBuf();

  std::string GetName() const;
  uint32_t GetVersion() const;
  std::string GetPlatformVersion() const;
  uint8_t GetPlatformType() const;

  void SetModelTaskDef(const std::shared_ptr<domi::ModelTaskDef> &task);
  void SetTBEKernelStore(const TBEKernelStore &tbe_kernal_store);
  void SetCustAICPUKernelStore(const CustAICPUKernelStore &cust_aicpu_kernal_store);
  void SetWeight(const Buffer &weights_buffer);

  bool LoadTBEKernelStore(const uint8_t *const data, const size_t len);
  bool LoadAICPUKernelStore(const uint8_t *const data, const size_t len);

  void SetName(const std::string &name);
  void SetVersion(const uint32_t version);
  void SetPlatformVersion(const std::string &platform_version);
  void SetPlatformType(const uint8_t platform_type);

  void SetAttrMap(const ProtoAttrMap &attrs);

  ProtoAttrMap &MutableAttrMap() override;

  using AttrHolder::SetAttr;
  using AttrHolder::GetAllAttrs;
  using AttrHolder::GetAllAttrNames;

  void SetModelId(const uint32_t model_id) { model_id_ = model_id; }
  uint32_t GetModelId() const { return model_id_; }

  Status GetSessionId(const uint32_t model_id, uint64_t &session_id) const;

  void SetModelInOutInfo(const std::shared_ptr<uint8_t> &buff) { model_in_out_info_ = buff; }

 protected:
  ConstProtoAttrMap &GetAttrMap() const override;

 private:
  void Init();

  ProtoAttrMap attrs_;  /*lint !e148*/

  ComputeGraphPtr graph_;
  std::shared_ptr<domi::ModelTaskDef> task_;  /*lint !e148*/
  TBEKernelStore tbe_kernal_store_;  /*lint !e148*/
  CustAICPUKernelStore cust_aicpu_kernal_store_;  /*lint !e148*/
  Buffer weights_buffer_;  /*lint !e148*/
  // weight_data_buffer is high priority than weights_buffer_
  DataBuffer weight_data_buffer_;
  std::string name_;
  uint32_t version_ = {0U};
  std::string platform_version_;
  uint8_t platform_type_ = {0U};
  uint32_t model_id_ = INVALID_MODEL_ID;
  std::map<uint32_t, uint64_t> model_id_to_session_id_map_;
  std::shared_ptr<uint8_t> model_in_out_info_;
};
using GeModelPtr = std::shared_ptr<ge::GeModel>;
}  // namespace ge
#endif  // GE_MODEL_GE_MODEL_H_
