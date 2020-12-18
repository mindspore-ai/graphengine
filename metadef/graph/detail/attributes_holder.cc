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

#include "detail/attributes_holder.h"
#include <map>
#include "debug/ge_log.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_attr_value.h"
#include "proto/ge_ir.pb.h"


namespace ge {
using std::map;
using std::unordered_set;
void AttrHolder::CopyAttrsFrom(const AttrHolder &holder) { MutableAttrMap().CopyValueFrom(holder.GetAttrMap()); }
graphStatus AttrHolder::SetAttr(const std::string &name, const GeAttrValue &value) {
  if (value.IsEmpty()) {
    GELOGE(GRAPH_FAILED, "value is empty, key of the attr is %s", name.c_str());
    return GRAPH_FAILED;
  }
  auto proto_map = MutableAttrMap().GetProtoMsg();
  auto proto_val = value.value_.GetProtoMsg();
  if (proto_map == nullptr || proto_val == nullptr) {
    return GRAPH_FAILED;
  }
  auto it = proto_map->find(name);
  if (it != proto_map->end()) {
    if (it->second.value_case() != proto::AttrDef::VALUE_NOT_SET &&
        it->second.value_case() != proto_val->value_case()) {
      return GRAPH_FAILED;
    }
  }
  (*proto_map)[name] = *proto_val;
  return GRAPH_SUCCESS;
}

graphStatus AttrHolder::AddRequiredAttr(const std::string &name) {
  if (HasAttr(name)) {
    return GRAPH_FAILED;
  }
  requiredAttrs_.push_back(name);
  return GRAPH_SUCCESS;
}

graphStatus AttrHolder::GetAttr(const std::string &name, GeAttrValue &value) const {
  auto proto_map = GetAttrMap().GetProtoMsg();
  auto proto_val = value.value_.GetProtoMsg();
  if (proto_map == nullptr || proto_val == nullptr) {
    return GRAPH_FAILED;
  }
  auto it = proto_map->find(name);
  if (it != proto_map->end()) {
    *proto_val = it->second;
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

bool AttrHolder::HasAttr(const std::string &name) const {
  auto proto_map = GetAttrMap().GetProtoMsg();
  if (proto_map != nullptr) {
    if (proto_map->find(name) != proto_map->end()) {
      return true;
    }
  }
  return std::find(requiredAttrs_.begin(), requiredAttrs_.end(), name) != requiredAttrs_.end();
}

graphStatus AttrHolder::DelAttr(const std::string &name) {
  auto proto_map = MutableAttrMap().GetProtoMsg();
  if (proto_map == nullptr) {
    return GRAPH_FAILED;
  }
  auto it = proto_map->find(name);
  if (it != proto_map->end()) {
    (void)proto_map->erase(it);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

const std::map<string, GeAttrValue> AttrHolder::GetAllAttrs() const {
  std::map<string, GeAttrValue> attr_value_map;
  auto proto_map = GetAttrMap().GetProtoMsg();
  if (proto_map != nullptr) {
    auto proto_owner = GetAttrMap().GetProtoOwner();
    GE_CHK_BOOL_EXEC(proto_owner != nullptr, return attr_value_map, "proto_owner is nullptr");
    for (const auto &it : *proto_map) {
      attr_value_map[it.first] = GeAttrValue(proto_owner, const_cast<proto::AttrDef *>(&it.second));
    }
  }
  return attr_value_map;
}

const std::unordered_set<string> AttrHolder::GetAllAttrNames() const {
  std::unordered_set<string> names;
  auto proto_map = GetAttrMap().GetProtoMsg();
  if (proto_map != nullptr) {
    for (const auto &it : *proto_map) {
      (void)names.insert(it.first);
    }
  }
  for (const string &it : requiredAttrs_) {
    (void)names.insert(it);
  }
  return names;
}

template <>
void GeIrProtoHelper<proto::AttrDef>::InitDefault() {
  std::shared_ptr<proto::AttrDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::AttrDef>();
  if (proto_owner == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::AttrDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::TensorDef>::InitDefault() {
  std::shared_ptr<proto::TensorDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::TensorDef>();
  if (proto_owner == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::TensorDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::TensorDescriptor>::InitDefault() {
  std::shared_ptr<proto::TensorDescriptor> proto_owner;
  proto_owner = ComGraphMakeShared<proto::TensorDescriptor>();
  if (proto_owner == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::TensorDescriptor make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::ShapeDef>::InitDefault() {
  std::shared_ptr<proto::ShapeDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::ShapeDef>();
  if (proto_owner == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::ShapeDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::NamedAttrs>::InitDefault() {
  std::shared_ptr<proto::NamedAttrs> proto_owner;
  proto_owner = ComGraphMakeShared<proto::NamedAttrs>();
  if (proto_owner == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::NamedAttrs make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::ModelDef>::InitDefault() {
  std::shared_ptr<proto::ModelDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::ModelDef>();
  if (proto_owner == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::ModelDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::OpDef>::InitDefault() {
  std::shared_ptr<proto::OpDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::OpDef>();
  if (proto_owner == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::OpDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::GraphDef>::InitDefault() {
  std::shared_ptr<proto::GraphDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::GraphDef>();
  if (proto_owner == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::GraphDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<ProtoAttrMap>::InitDefault() {
  std::shared_ptr<proto::TensorDescriptor> proto_owner;
  proto_owner = ComGraphMakeShared<proto::TensorDescriptor>();
  if (proto_owner == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::TensorDescriptor make shared failed");
    return;
  }
  protoMsg_ = proto_owner->mutable_attr();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<const ProtoAttrMap>::InitDefault() {
  std::shared_ptr<proto::TensorDescriptor> proto_owner;
  proto_owner = ComGraphMakeShared<proto::TensorDescriptor>();
  if (proto_owner == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::TensorDescriptor make shared failed");
    return;
  }
  protoMsg_ = &proto_owner->attr();
  protoOwner_ = proto_owner;
}
}  // namespace ge
