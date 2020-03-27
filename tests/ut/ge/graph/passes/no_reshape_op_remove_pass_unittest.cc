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

#include "graph/passes/no_reshape_op_remove_pass.h"

#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/debug/graph_debug.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/op_desc.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "inc/pass_manager.h"
#include "opskernel_manager/ops_kernel_manager.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestGraphNoReshapeOpRemovePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }
  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }
  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }
  ge::NodePtr Build(const ge::ComputeGraphPtr &graph) { return graph->AddNode(op_desc_); }

 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }
  ge::OpDescPtr op_desc_;
};

/// data->expanddim->reshape1->reshape2->reshape3->squeeze->reshape4->sinh
///                                                            /
///                                                          const
void make_graph(ComputeGraphPtr &graph) {
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr node_expanddim_1 = NodeBuilder("ExpandDim", EXPANDDIMS)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                     .Build(graph);

  ge::NodePtr node_reshape_1 = NodeBuilder("Reshape_1", RESHAPE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 1, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);

  ge::NodePtr node_reshape_2 = NodeBuilder("Reshape_2", RESHAPE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 1, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);

  ge::NodePtr node_reshape_3 = NodeBuilder("Reshape_3", RESHAPE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);

  ge::NodePtr node_squeeze_1 = NodeBuilder("Squeeze", SQUEEZE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 1, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);
  ge::NodePtr node_const =
      NodeBuilder("const", CONSTANT).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr node_reshape_4 = NodeBuilder("Reshape_4", RESHAPE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);

  ge::NodePtr node_sinh_1 = NodeBuilder("sinh", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 1, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);

  GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_expanddim_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_expanddim_1->GetOutDataAnchor(0), node_reshape_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_reshape_1->GetOutDataAnchor(0), node_reshape_2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_reshape_2->GetOutDataAnchor(0), node_reshape_3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_reshape_3->GetOutDataAnchor(0), node_squeeze_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_squeeze_1->GetOutDataAnchor(0), node_reshape_4->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_const->GetOutDataAnchor(0), node_reshape_4->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_reshape_4->GetOutDataAnchor(0), node_sinh_1->GetInDataAnchor(0));
}

// reshape->permute->transdata->correlation
void make_graph_for_sfc(ComputeGraphPtr &graph) {
  // Node4D
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  // reshape1
  ge::NodePtr node_reshape_1 = NodeBuilder("Reshape_3", RESHAPE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 1, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);
  // permute
  ge::NodePtr node_permute_1 = NodeBuilder("permute", PERMUTE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);
  // transdata
  ge::NodePtr node_transdata_1 = NodeBuilder("transdata", TRANSDATA)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                     .Build(graph);
  // transdata
  ge::NodePtr node_correlation_1 = NodeBuilder("correlation", CORRELATION)
                                       .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                       .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                       .Build(graph);
  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_reshape_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_reshape_1->GetOutDataAnchor(0), node_permute_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_permute_1->GetOutDataAnchor(0), node_transdata_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transdata_1->GetOutDataAnchor(0), node_correlation_1->GetInDataAnchor(0));
}

TEST_F(UtestGraphNoReshapeOpRemovePass, node_to_be_delete_success) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test");
  make_graph(compute_graph);

  NoReshapeOpRemovePass noreshapepass;
  ge::NodePtr expandDim1 = compute_graph->FindNode("ExpandDim");
  Status status = noreshapepass.Run(expandDim1);
  EXPECT_EQ(status, ge::SUCCESS);
  expandDim1 = compute_graph->FindNode("ExpandDim");
  EXPECT_EQ(expandDim1, nullptr);

  ge::NodePtr reshape1 = compute_graph->FindNode("Reshape_1");
  status = noreshapepass.Run(reshape1);
  EXPECT_EQ(status, ge::SUCCESS);
  reshape1 = compute_graph->FindNode("Reshape_1");
  EXPECT_EQ(reshape1, nullptr);

  ge::NodePtr reshape2 = compute_graph->FindNode("Reshape_2");
  EXPECT_EQ(reshape2, nullptr);

  ge::NodePtr reshape3 = compute_graph->FindNode("Reshape_3");
  EXPECT_EQ(reshape3, nullptr);

  ge::NodePtr reshape4 = compute_graph->FindNode("Reshape_4");
  status = noreshapepass.Run(reshape4);
  EXPECT_EQ(status, ge::SUCCESS);
  reshape4 = compute_graph->FindNode("Reshape_4");
  EXPECT_EQ(reshape4, nullptr);

  ge::NodePtr const1 = compute_graph->FindNode("const");
  auto output_size = const1->GetOutDataNodes().size();
  EXPECT_EQ(output_size, 0);
  ge::NodePtr sinh1 = compute_graph->FindNode("sinh");
  auto input_size = sinh1->GetInDataNodes().size();
  EXPECT_EQ(input_size, 1);
}
TEST_F(UtestGraphNoReshapeOpRemovePass, reshape_for_sfc_net_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  make_graph_for_sfc(graph);
  NoReshapeOpRemovePass noreshapepass;

  NodePtr reshape_node = graph->FindNode("Reshape_3");
  noreshapepass.Run(reshape_node);
  NodePtr permute_node = graph->FindNode("permute");
  bool flag = false;
  AttrUtils::GetBool(permute_node->GetOpDesc(), "reshape_correlation", flag);
  EXPECT_EQ(flag, true);
}
