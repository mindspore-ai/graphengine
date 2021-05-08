/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <gmock/gmock.h>
#include <vector>
#include <memory>

#define protected public
#define private public
#include "hybrid/node_executor/compiledsubgraph/known_node_executor.h"
#include "common/dump/dump_manager.h"
#undef private
#undef protected
#include "graph/manager/graph_mem_allocator.h"
#include "../graph/passes/graph_builder_utils.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace hybrid;

class UnknownNodeExecutorTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
class KnownNodeTaskMock : public KnownNodeTask {
 public:
  KnownNodeTaskMock(std::shared_ptr<DavinciModel> davinci_model): KnownNodeTask(davinci_model) {};
  ~KnownNodeTaskMock() override = default;
  MOCK_METHOD2(DoInitDavinciModel, Status(void *, size_t));
};
}

TEST_F(UnknownNodeExecutorTest, test_init_davinci_model) {
  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  davinci_model->SetDeviceId(0);
  davinci_model->SetKnownNode(true);

  auto ge_model = make_shared<GeModel>();
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  davinci_model->Assign(ge_model);

  HybridModel model(nullptr);
  KnownNodeTaskMock mock(davinci_model);
  DumpProperties dump_properties;
  dump_properties.enable_dump_ = "1";
  DumpManager::GetInstance().AddDumpProperties(model.GetSessionId(), dump_properties);
  EXPECT_CALL(mock, DoInitDavinciModel).WillRepeatedly(::testing::Return(SUCCESS));
  ASSERT_EQ(mock.InitDavinciModel(model, model.GetModelWeight("subgraph")), SUCCESS);

  int32_t buffer[8];
  model.weight_buffer_map_.emplace("subgraph", TensorBuffer::Create(buffer, sizeof(buffer)));
  ASSERT_EQ(mock.InitDavinciModel(model, model.GetModelWeight("subgraph")), SUCCESS);
}

TEST_F(UnknownNodeExecutorTest, TestParseAttrForAllocatingOutputs) {
  ut::GraphBuilder builder("test-graph");
  auto data_node = builder.AddNode("Data0", DATA, 1, 1);
  auto netoutput_node = builder.AddNode("NodeOutput", NETOUTPUT, 2, 2);
  builder.AddDataEdge(data_node, 0, netoutput_node, 0);
  auto const_node = builder.AddNode("Const0", CONSTANT, 0, 1);
  builder.AddDataEdge(const_node, 0, netoutput_node, 1);
  auto graph = builder.GetGraph();

  ut::GraphBuilder builder2("root-graph");
  auto partitioned_call = builder2.AddNode("Node0", PARTITIONEDCALL, 1, 2);
  NodeItem node_item(partitioned_call);
  ASSERT_EQ(KnownNodeExecutor::ParseAttrForAllocatingOutputs(node_item, *graph), SUCCESS);
  ASSERT_EQ(node_item.ref_outputs.size(), 1);
  ASSERT_EQ(node_item.ref_outputs[1], const_node);
  ASSERT_EQ(node_item.reuse_inputs.size(), 1);
  ASSERT_EQ(node_item.reuse_inputs[0], 0);
}