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
#ifndef GE_COMMON_MODEL_ENDPOINT_H_
#define GE_COMMON_MODEL_ENDPOINT_H_

#include <string>
#include "graph/node.h"
#include "ge/ge_api_types.h"
#include "proto/flow_model.pb.h"

namespace ge {
using AttrValueMap = ::google::protobuf::Map<string, ge::flow_model::proto::ModelRelationDef::AttrValue>;
enum class EndpointType : std::uint32_t { kQueue = 0, kEvent = 1, kMaxEndpointTypeNum };

class Endpoint {
 public:
  Endpoint(std::string name, EndpointType endpoint_type) : name_(std::move(name)), endpoint_type_(endpoint_type){};
  ~Endpoint() = default;
  EndpointType GetEndpointType() const;
  const std::string &GetName() const;
  std::string &MutableName();
  Endpoint &SetName(const std::string &name);
  Endpoint &SetEndpointType(EndpointType endpoint_type);
  template <typename T>
  void SetAttr(const std::string &name, const T &value);
  void SetAnyValueByName(const std::string &name, const AnyValue &value);
  template <typename T>
  const T GetAttr(const std::string &name) const;
  template <typename T>
  const T GetDefaultAttr() const;

  // function for serialize
  std::map<std::string, AnyValue> GetAllAttrs() const;
  using SetAttrFunc = Status (*)(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr);
  static const std::map<AnyValue::ValueType, SetAttrFunc> set_attr_funcs_;
  static Status SetStringAttr(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr);
  static Status SetIntAttr(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr);
  static Status SetBoolAttr(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr);

  // function for deserialize
  using SetAnyValueFunc = Status (*)(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value);
  static const std::map<ge::flow_model::proto::ModelRelationDef::AttrValue::ValueCase, SetAnyValueFunc>
      set_any_value_funcs_;
  static Status SetStringAnyValue(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value);
  static Status SetIntAnyValue(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value);
  static Status SetBoolAnyValue(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value);

  Status Serialize(ge::flow_model::proto::ModelRelationDef_Endpoint *proto_endpoint) const;
  Status Deserialize(const ge::flow_model::proto::ModelRelationDef_Endpoint &proto_endpoint);

 private:
  std::string name_;
  EndpointType endpoint_type_;
  AttrStore attrs_;
};

class P2pNodeUtils {
 public:
  explicit P2pNodeUtils(Endpoint &endpoint) : endpoint_(endpoint){};
  P2pNodeUtils &SetType(const std::string &event_type);
  P2pNodeUtils &SetGroupName(const std::string &group_name);
  P2pNodeUtils &SetTag(int64_t tag);
  P2pNodeUtils &SetPeerRank(int64_t peer_rank);
  P2pNodeUtils &SetLogicPeerRank(int64_t logic_peer_rank);
  P2pNodeUtils &SetDeviceIndices(const std::string &indices);
  P2pNodeUtils &SetIsOutput(bool is_output);
  // flow : need deploy
  // p2p_node : if send/recv nodes is in the graph, does not need deploy, else need deploy
  P2pNodeUtils &SetNeedDeploy(bool nee_deploy = false);
  std::string GetType() const;
  std::string GetGroupName() const;
  int64_t GetTag() const;
  int64_t GetPeerRank() const;
  int64_t GetLogicPeerRank() const;
  std::string GetDeviceIndices() const;
  bool IsOutput() const;
  bool GetNeedDeploy() const;
  // should called after wrapped partitioned call node
  Status GetNameByWrappedPartitionedCall(NodePtr &src_node, std::string &new_endpoint_name) const;

 private:
  Endpoint &endpoint_;
};

class QueueNodeUtils {
 public:
  explicit QueueNodeUtils(Endpoint &endpoint) : endpoint_(endpoint){};
  QueueNodeUtils &SetDepth(int64_t depth);
  QueueNodeUtils &SetEnqueuePolicy(const std::string &enqueue_policy);
  QueueNodeUtils &SetIsControl(bool is_control = true);
  int64_t GetDepth() const;
  std::string GetEnqueuePolicy() const;
  bool GetIsControl() const;

 private:
  Endpoint &endpoint_;
};
}  // namespace ge
#endif  // GE_COMMON_MODEL_ENDPOINT_H_
