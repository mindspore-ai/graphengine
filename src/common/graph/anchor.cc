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

#include "graph/anchor.h"
#include <algorithm>
#include <cstring>
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/node.h"

namespace ge {
Anchor::Anchor(const NodePtr &owner_node, int idx) : owner_node_(owner_node), idx_(idx) {}

bool Anchor::IsTypeOf(TYPE type) const { return strcmp(Anchor::TypeOf<Anchor>(), type) == 0; }

size_t Anchor::GetPeerAnchorsSize() const { return peer_anchors_.size(); }

Anchor::Vistor<AnchorPtr> Anchor::GetPeerAnchors() const {
  vector<AnchorPtr> ret;
  for (const auto &anchor : peer_anchors_) {
    ret.push_back(anchor.lock());
  }
  return Anchor::Vistor<AnchorPtr>(shared_from_this(), ret);
}

AnchorPtr Anchor::GetFirstPeerAnchor() const {
  if (peer_anchors_.empty()) {
    return nullptr;
  } else {
    return Anchor::DynamicAnchorCast<Anchor>(peer_anchors_.begin()->lock());
  }
}

NodePtr Anchor::GetOwnerNode() const { return owner_node_.lock(); }

void Anchor::UnlinkAll() noexcept {
  if (!peer_anchors_.empty()) {
    do {
      auto peer_anchor_ptr = peer_anchors_.begin()->lock();
      if (Unlink(peer_anchor_ptr) != GRAPH_SUCCESS) {
        GELOGW("unlink peer_anchor_ptr failed.");
      }
    } while (!peer_anchors_.empty());
  }
}

graphStatus Anchor::Unlink(const AnchorPtr &peer) {
  if (peer == nullptr) {
    GELOGE(GRAPH_FAILED, "peer anchor is invalid.");
    return GRAPH_FAILED;
  }
  auto it = std::find_if(peer_anchors_.begin(), peer_anchors_.end(), [peer](const std::weak_ptr<Anchor> &an) {
    auto anchor = an.lock();
    return peer->Equal(anchor);
  });

  GE_IF_BOOL_EXEC(it == peer_anchors_.end(), GELOGW("this anchor is not connected to peer"); return GRAPH_FAILED);

  auto it_peer =
    std::find_if(peer->peer_anchors_.begin(), peer->peer_anchors_.end(), [this](const std::weak_ptr<Anchor> &an) {
      auto anchor = an.lock();
      return Equal(anchor);
    });

  GE_CHK_BOOL_RET_STATUS(it_peer != peer->peer_anchors_.end(), GRAPH_FAILED, "peer is not connected to this anchor");

  (void)peer_anchors_.erase(it);
  (void)peer->peer_anchors_.erase(it_peer);
  return GRAPH_SUCCESS;
}

graphStatus Anchor::ReplacePeer(const AnchorPtr &old_peer, const AnchorPtr &first_peer, const AnchorPtr &second_peer) {
  GE_CHK_BOOL_RET_STATUS(old_peer != nullptr, GRAPH_FAILED, "this old peer anchor is nullptr");
  GE_CHK_BOOL_RET_STATUS(first_peer != nullptr, GRAPH_FAILED, "this first peer anchor is nullptr");
  GE_CHK_BOOL_RET_STATUS(second_peer != nullptr, GRAPH_FAILED, "this second peer anchor is nullptr");
  auto this_it = std::find_if(peer_anchors_.begin(), peer_anchors_.end(), [old_peer](const std::weak_ptr<Anchor> &an) {
    auto anchor = an.lock();
    return old_peer->Equal(anchor);
  });

  GE_CHK_BOOL_RET_STATUS(this_it != peer_anchors_.end(), GRAPH_FAILED, "this anchor is not connected to old_peer");

  auto old_it = std::find_if(old_peer->peer_anchors_.begin(), old_peer->peer_anchors_.end(),
                             [this](const std::weak_ptr<Anchor> &an) {
                               auto anchor = an.lock();
                               return Equal(anchor);
                             });

  GE_CHK_BOOL_RET_STATUS(old_it != old_peer->peer_anchors_.end(), GRAPH_FAILED,
                         "old_peer is not connected to this anchor");
  *this_it = first_peer;
  first_peer->peer_anchors_.push_back(shared_from_this());
  *old_it = second_peer;
  second_peer->peer_anchors_.push_back(old_peer);
  return GRAPH_SUCCESS;
}

bool Anchor::IsLinkedWith(const AnchorPtr &peer) {
  auto it = std::find_if(peer_anchors_.begin(), peer_anchors_.end(), [peer](const std::weak_ptr<Anchor> &an) {
    auto anchor = an.lock();
    GE_CHK_BOOL_RET_STATUS(peer != nullptr, false, "this old peer anchor is nullptr");
    return peer->Equal(anchor);
  });
  return (it != peer_anchors_.end());
}

int Anchor::GetIdx() const { return idx_; }

void Anchor::SetIdx(int index) { idx_ = index; }

DataAnchor::DataAnchor(const NodePtr &owner_node, int idx) : Anchor(owner_node, idx) {}

bool DataAnchor::IsTypeOf(TYPE type) const {
  if (strcmp(Anchor::TypeOf<DataAnchor>(), type) == 0) {
    return true;
  }
  return Anchor::IsTypeOf(type);
}

InDataAnchor::InDataAnchor(const NodePtr &owner_node, int idx) : DataAnchor(owner_node, idx) {}

OutDataAnchorPtr InDataAnchor::GetPeerOutAnchor() const {
  if (peer_anchors_.empty()) {
    return nullptr;
  } else {
    return Anchor::DynamicAnchorCast<OutDataAnchor>(peer_anchors_.begin()->lock());
  }
}

graphStatus InDataAnchor::LinkFrom(const OutDataAnchorPtr &src) {
  // InDataAnchor must be only linkfrom once
  if (src == nullptr || !peer_anchors_.empty()) {
    GELOGE(GRAPH_FAILED, "src anchor is invalid or the peerAnchors is not empty.");
    return GRAPH_FAILED;
  }
  peer_anchors_.push_back(src);
  src->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

bool InDataAnchor::Equal(AnchorPtr anchor) const {
  auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor);
  if (in_data_anchor != nullptr) {
    if (GetOwnerNode() == in_data_anchor->GetOwnerNode() && GetIdx() == in_data_anchor->GetIdx()) {
      return true;
    }
  }
  return false;
}

bool InDataAnchor::IsTypeOf(TYPE type) const {
  if (strcmp(Anchor::TypeOf<InDataAnchor>(), type) == 0) {
    return true;
  }
  return DataAnchor::IsTypeOf(type);
}

OutDataAnchor::OutDataAnchor(const NodePtr &owner_node, int idx) : DataAnchor(owner_node, idx) {}

OutDataAnchor::Vistor<InDataAnchorPtr> OutDataAnchor::GetPeerInDataAnchors() const {
  vector<InDataAnchorPtr> ret;
  for (const auto &anchor : peer_anchors_) {
    auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
    if (in_data_anchor != nullptr) {
      ret.push_back(in_data_anchor);
    }
  }
  return OutDataAnchor::Vistor<InDataAnchorPtr>(shared_from_this(), ret);
}

uint32_t OutDataAnchor::GetPeerInDataNodesSize() const {
  uint32_t out_nums = 0;
  for (const auto &anchor : peer_anchors_) {
    auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
    if (in_data_anchor != nullptr && in_data_anchor->GetOwnerNode() != nullptr) {
      out_nums++;
    }
  }
  return out_nums;
}

OutDataAnchor::Vistor<InControlAnchorPtr> OutDataAnchor::GetPeerInControlAnchors() const {
  vector<InControlAnchorPtr> ret;
  for (const auto &anchor : peer_anchors_) {
    auto in_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor.lock());
    if (in_control_anchor != nullptr) {
      ret.push_back(in_control_anchor);
    }
  }
  return OutDataAnchor::Vistor<InControlAnchorPtr>(shared_from_this(), ret);
}

graphStatus OutDataAnchor::LinkTo(const InDataAnchorPtr &dest) {
  if (dest == nullptr || !dest->peer_anchors_.empty()) {
    GELOGE(GRAPH_FAILED, "dest anchor is invalid or the peerAnchors is not empty.");
    return GRAPH_FAILED;
  }
  peer_anchors_.push_back(dest);
  dest->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

graphStatus OutDataAnchor::LinkTo(const InControlAnchorPtr &dest) {
  if (dest == nullptr) {
    GELOGE(GRAPH_FAILED, "dest anchor is invalid.");
    return GRAPH_FAILED;
  }
  peer_anchors_.push_back(dest);
  dest->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

graphStatus OutControlAnchor::LinkTo(const InDataAnchorPtr &dest) {
  if (dest == nullptr) {
    GELOGE(GRAPH_FAILED, "dest anchor is invalid.");
    return GRAPH_FAILED;
  }
  peer_anchors_.push_back(dest);
  dest->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

bool OutDataAnchor::Equal(AnchorPtr anchor) const {
  CHECK_FALSE_EXEC(anchor != nullptr, return false);
  auto out_data_anchor = Anchor::DynamicAnchorCast<OutDataAnchor>(anchor);
  if (out_data_anchor != nullptr) {
    if (GetOwnerNode() == out_data_anchor->GetOwnerNode() && GetIdx() == out_data_anchor->GetIdx()) {
      return true;
    }
  }
  return false;
}

bool OutDataAnchor::IsTypeOf(TYPE type) const {
  if (strcmp(Anchor::TypeOf<OutDataAnchor>(), type) == 0) {
    return true;
  }
  return DataAnchor::IsTypeOf(type);
}

ControlAnchor::ControlAnchor(const NodePtr &owner_node) : Anchor(owner_node, -1) {}

ControlAnchor::ControlAnchor(const NodePtr &owner_node, int idx) : Anchor(owner_node, idx) {}

bool ControlAnchor::IsTypeOf(TYPE type) const {
  if (strcmp(Anchor::TypeOf<ControlAnchor>(), type) == 0) {
    return true;
  }
  return Anchor::IsTypeOf(type);
}

InControlAnchor::InControlAnchor(const NodePtr &owner_node) : ControlAnchor(owner_node) {}

InControlAnchor::InControlAnchor(const NodePtr &owner_node, int idx) : ControlAnchor(owner_node, idx) {}

InControlAnchor::Vistor<OutControlAnchorPtr> InControlAnchor::GetPeerOutControlAnchors() const {
  vector<OutControlAnchorPtr> ret;
  for (const auto &anchor : peer_anchors_) {
    auto out_control_anchor = Anchor::DynamicAnchorCast<OutControlAnchor>(anchor.lock());
    if (out_control_anchor != nullptr) {
      ret.push_back(out_control_anchor);
    }
  }
  return InControlAnchor::Vistor<OutControlAnchorPtr>(shared_from_this(), ret);
}

InControlAnchor::Vistor<OutDataAnchorPtr> InControlAnchor::GetPeerOutDataAnchors() const {
  vector<OutDataAnchorPtr> ret;
  for (const auto &anchor : peer_anchors_) {
    auto out_data_anchor = Anchor::DynamicAnchorCast<OutDataAnchor>(anchor.lock());
    if (out_data_anchor != nullptr) {
      ret.push_back(out_data_anchor);
    }
  }
  return InControlAnchor::Vistor<OutDataAnchorPtr>(shared_from_this(), ret);
}

graphStatus InControlAnchor::LinkFrom(const OutControlAnchorPtr &src) {
  if (src == nullptr) {
    GELOGE(GRAPH_FAILED, "src anchor is invalid.");
    return GRAPH_FAILED;
  }
  peer_anchors_.push_back(src);
  src->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

bool InControlAnchor::Equal(AnchorPtr anchor) const {
  CHECK_FALSE_EXEC(anchor != nullptr, return false);
  auto in_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor);
  if (in_control_anchor != nullptr) {
    if (GetOwnerNode() == in_control_anchor->GetOwnerNode()) {
      return true;
    }
  }
  return false;
}

bool InControlAnchor::IsTypeOf(TYPE type) const {
  if (strcmp(Anchor::TypeOf<InControlAnchor>(), type) == 0) {
    return true;
  }
  return ControlAnchor::IsTypeOf(type);
}

OutControlAnchor::OutControlAnchor(const NodePtr &owner_node) : ControlAnchor(owner_node) {}

OutControlAnchor::OutControlAnchor(const NodePtr &owner_node, int idx) : ControlAnchor(owner_node, idx) {}

OutControlAnchor::Vistor<InControlAnchorPtr> OutControlAnchor::GetPeerInControlAnchors() const {
  vector<InControlAnchorPtr> ret;
  for (const auto &anchor : peer_anchors_) {
    auto in_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor.lock());
    if (in_control_anchor != nullptr) {
      ret.push_back(in_control_anchor);
    }
  }
  return OutControlAnchor::Vistor<InControlAnchorPtr>(shared_from_this(), ret);
}

OutControlAnchor::Vistor<InDataAnchorPtr> OutControlAnchor::GetPeerInDataAnchors() const {
  vector<InDataAnchorPtr> ret;
  for (const auto &anchor : peer_anchors_) {
    auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
    if (in_data_anchor != nullptr) {
      ret.push_back(in_data_anchor);
    }
  }
  return OutControlAnchor::Vistor<InDataAnchorPtr>(shared_from_this(), ret);
}

graphStatus OutControlAnchor::LinkTo(const InControlAnchorPtr &dest) {
  if (dest == nullptr) {
    GELOGE(GRAPH_FAILED, "dest anchor is invalid.");
    return GRAPH_FAILED;
  }
  peer_anchors_.push_back(dest);
  dest->peer_anchors_.push_back(shared_from_this());
  return GRAPH_SUCCESS;
}

bool OutControlAnchor::Equal(AnchorPtr anchor) const {
  auto out_control_anchor = Anchor::DynamicAnchorCast<OutControlAnchor>(anchor);
  if (out_control_anchor != nullptr) {
    if (GetOwnerNode() == out_control_anchor->GetOwnerNode()) {
      return true;
    }
  }
  return false;
}

bool OutControlAnchor::IsTypeOf(TYPE type) const {
  if (strcmp(Anchor::TypeOf<OutControlAnchor>(), type) == 0) {
    return true;
  }
  return ControlAnchor::IsTypeOf(type);
}
}  // namespace ge
