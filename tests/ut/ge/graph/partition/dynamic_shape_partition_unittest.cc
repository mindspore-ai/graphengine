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

#include <gtest/gtest.h>
#include "graph/partition/dynamic_shape_partition.h"
#include "compute_graph.h"
#include "inc/framework/common/types.h"
#include "utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"


#define private public
#define protected public

namespace ge {

namespace {

GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, Format format = FORMAT_NCHW,
                                 DataType data_type = DT_FLOAT) {
  GeShape ge_shape{vector<int64_t>(shape)};
  GeTensorDescPtr tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(ge_shape);
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  return tensor_desc;
}

class NodeBuilder {
  public:
    NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

    NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                              DataType data_type = DT_FLOAT) {
      op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
      return *this;
    }

    NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                               DataType data_type = DT_FLOAT) {
      op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
      return *this;
    }

    NodeBuilder &AddOutputDesc(GeTensorDescPtr tensor_desc) {
      op_desc_->AddOutputDesc(tensor_desc->Clone());
      return *this;
    }

    NodePtr Build(const ComputeGraphPtr &graph) {
      NodePtr node = graph->AddNode(op_desc_);
      return node;
    }

  private:
    OpDescPtr op_desc_;
};
}  // namespace

class UtestDynamicShapePartition : public testing::Test {
  protected:
    void SetUp() {}

    void TearDown() {}
};

// test Init_EndGraphTaskInfo_failed
TEST_F(UtestDynamicShapePartition, single_op_scene_success) {
  ComputeGraphPtr graph = shared_ptr<ComputeGraph>("default");

  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  (void)AttrUtils::SetBool(add_n_node->GetOpDesc(), ATTR_SINGLE_OP_SCENE, true);

  DynamicShapePartitioner partitioner(computeGraph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
}

} // namespace ge