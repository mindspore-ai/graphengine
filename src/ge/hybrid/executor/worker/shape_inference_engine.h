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

#ifndef GE_HYBRID_EXECUTOR_INFERSHAPE_SHAPE_INFERENCE_ENGINE_H_
#define GE_HYBRID_EXECUTOR_INFERSHAPE_SHAPE_INFERENCE_ENGINE_H_

#include <memory>
#include <thread>
#include <unordered_map>
#include "common/thread_pool.h"
#include "hybrid/executor/hybrid_execution_context.h"

namespace ge {
namespace hybrid {
class ShapeInferenceEngine {
 public:
  explicit ShapeInferenceEngine(GraphExecutionContext *context);

  ~ShapeInferenceEngine() = default;

  Status Start(ThreadPool &pool);

 private:
  class ShapeFuture {
   public:
    ShapeFuture(NodePtr src_node, uint32_t src_index, NodeDoneManager *node_done_manager);
    ~ShapeFuture() = default;
    Status Get(GeShape &ori_shape, GeShape &shape) {
      GELOGI("Start to wait node: %s for getting shape", src_node_->GetName().c_str());
      if (!node_done_manager_->Await(src_node_)) {
        GELOGE(INTERNAL_ERROR, "cancelled");
        return INTERNAL_ERROR;
      }

      shape = src_node_->GetOpDesc()->MutableOutputDesc(src_index_)->MutableShape();
      ori_shape = src_node_->GetOpDesc()->MutableOutputDesc(src_index_)->GetOriginShape();
      GELOGI("Get shape from %s:%u. shape = [%s]", src_node_->GetName().c_str(), src_index_, shape.ToString().c_str());
      return SUCCESS;
    }

   private:
    NodePtr src_node_;
    uint32_t src_index_;
    NodeDoneManager *node_done_manager_;
  };

  struct InferenceState {
    explicit InferenceState(const NodeItem &node_item);
    inline bool IsInputShapesReady() const { return num_pending_shapes == 0; }

    void UpdateInputShape(uint32_t idx, const GeShape &ori_shape, const GeShape &shape);

    Status AwaitShapeFutures(GraphExecutionContext *context);

    void UpdateInputShapeFuture(uint32_t idx, ShapeFuture &&future);

    const NodeItem &node_item;

   private:
    std::vector<std::pair<uint32_t, ShapeFuture>> shape_futures;
    int num_pending_shapes = 0;
  };

  InferenceState *GetOrCreateEntry(const NodeItem &node_item);

  Status InferShapeProcess();

  void InferenceDone(Status status);

  Status InferShape(InferenceState &entry);

  void PropagateOutputShapes(InferenceState &entry, std::queue<InferenceState *> &queue);

  GraphExecutionContext *context_;
  std::unordered_map<int64_t, std::unique_ptr<InferenceState>> inference_states_;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_EXECUTOR_INFERSHAPE_SHAPE_INFERENCE_ENGINE_H_
