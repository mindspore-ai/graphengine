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
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "../utils/buffer_pool_graph_builder.h"
#include "graph/passes/buffer_pool_memory_pass.h"

#define protected public
#define private public
#include "graph/build/memory/buffer_pool_mem_assigner.h"
#include "graph/build/memory/graph_mem_assigner.h"
#include "graph/build/stream_allocator.h"
#undef protected
#undef private

namespace ge {
namespace {
const int64_t kMemoryTypeHBM = static_cast<int64_t>(RT_MEMORY_HBM);
const int64_t kMemoryTypeP2P = static_cast<int64_t>(RT_MEMORY_P2P_HBM);
const int64_t kMemoryTypeDDR = static_cast<int64_t>(RT_MEMORY_DDR);
const size_t kOffsetHBM = 10240;
const size_t kOffsetP2P = 20480;
const size_t kOffsetDDR = 30720;
const int64_t kMemAlignSize = 512;

int64_t AlignMemSize(int64_t mem_size, int64_t align_size = kMemAlignSize) {
  int64_t tmp = (mem_size + align_size - 1) / align_size * align_size;
  return tmp;
}
int64_t AlignOutputMemSize(int64_t mem_size) {
  int64_t tmp = (mem_size + kMemAlignSize - 1) / kMemAlignSize * kMemAlignSize;
  // hccl need alignment
  tmp = kMemAlignSize + tmp + kMemAlignSize;
  return tmp;
}
} // namespace
class UtestBufferPoolMemAssignerTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}

};

TEST_F(UtestBufferPoolMemAssignerTest, buffer_pool_normal_assign_success) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();
  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  std::map<int64_t, size_t> mem_type_to_offset = {{kMemoryTypeHBM, kOffsetHBM},
                                                  {kMemoryTypeP2P, kOffsetP2P}};
  int64_t offset_base = static_cast<int64_t>(kOffsetHBM + kMemAlignSize);
  std::vector<int64_t> expect_offset = {(offset_base + 0),
                                        (offset_base + AlignOutputMemSize(500)),
                                        (offset_base + (AlignOutputMemSize(500) * 2)),
                                        (offset_base + 0),
                                        (offset_base + AlignOutputMemSize(1024))};

  BufferPoolMemAssigner buffer_pool_mem_assigner(graph, mem_type_to_offset);
  ret = buffer_pool_mem_assigner.Assign();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(buffer_pool_mem_assigner.GetMemOffset(), offset_base +
                                                     AlignMemSize(5600, kMemAlignSize) + kMemAlignSize);

  {
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(0));
  }

  {
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(1));
  }

  {
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(2));
  }

  {
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(3));
  }

  {
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(4));
  }
}

TEST_F(UtestBufferPoolMemAssignerTest, buffer_pool_normal_graph_with_multi_buffer_pool_assign_success) {
  ut::BufferPoolGraphBuilder builder("NormalGraphWithMultiBufferPool");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraphWithMultiBufferPool();
  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  std::map<int64_t, size_t> mem_type_to_offset = {{kMemoryTypeHBM, kOffsetHBM},
                                                  {kMemoryTypeP2P, kOffsetP2P}};
  int64_t offset_base_0 = static_cast<int64_t>(kOffsetHBM + kMemAlignSize);
  int64_t offset_base_1 = static_cast<int64_t>(kOffsetHBM + kMemAlignSize) +
                          AlignMemSize(5000, kMemAlignSize) + kMemAlignSize;
  std::vector<int64_t> expect_offset = {(offset_base_0 + 0),
                                        (offset_base_1 + 0),
                                        (offset_base_0 + AlignOutputMemSize(500)),
                                        (offset_base_0 + 0),
                                        (offset_base_1 + AlignOutputMemSize(500))};

  BufferPoolMemAssigner buffer_pool_mem_assigner(graph, mem_type_to_offset);
  ret = buffer_pool_mem_assigner.Assign();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(buffer_pool_mem_assigner.GetMemOffset(), offset_base_1 +
                                                     AlignMemSize(5000, kMemAlignSize) + kMemAlignSize);

  {
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(0));
  }

  {
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(1));
  }

  {
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(2));
  }

  {
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(3));
  }

  {
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(4));
  }
}

TEST_F(UtestBufferPoolMemAssignerTest, buffer_pool_serial_graph_assign_success) {
  ut::BufferPoolGraphBuilder builder("SerialGraph");
  ge::ComputeGraphPtr graph = builder.BuildSerialGraph();
  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  std::map<int64_t, size_t> mem_type_to_offset = {{kMemoryTypeHBM, kOffsetHBM},
                                                  {kMemoryTypeP2P, kOffsetP2P}};
  int64_t offset_base = static_cast<int64_t>(kOffsetHBM + kMemAlignSize);
  std::vector<int64_t> expect_offset = {offset_base, offset_base, offset_base, offset_base, offset_base};

  BufferPoolMemAssigner buffer_pool_mem_assigner(graph, mem_type_to_offset);
  ret = buffer_pool_mem_assigner.Assign();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(buffer_pool_mem_assigner.GetMemOffset(), offset_base +
                                                     AlignMemSize(2048, kMemAlignSize) + kMemAlignSize);

  {
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(0));
  }

  {
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(1));
  }

  {
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(2));
  }

  {
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(3));
  }

  {
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(4));
  }
}

TEST_F(UtestBufferPoolMemAssignerTest, buffer_pool_subgraph_with_inner_dependency_assign_success) {
  ut::BufferPoolGraphBuilder builder("SubgraphWithInnerDependency");
  ge::ComputeGraphPtr graph = builder.BuildSubgraphWithInnerDependency();
  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  std::map<int64_t, size_t> mem_type_to_offset = {{kMemoryTypeHBM, kOffsetHBM},
                                                  {kMemoryTypeP2P, kOffsetP2P}};
  int64_t offset_base = static_cast<int64_t>(kOffsetHBM + kMemAlignSize);
  std::vector<int64_t> expect_offset = {(offset_base + 0),
                                        (offset_base + AlignOutputMemSize(500)),
                                        (offset_base + (AlignOutputMemSize(500) * 2)),
                                        (offset_base + 0),
                                        (offset_base + AlignOutputMemSize(1024))};

  BufferPoolMemAssigner buffer_pool_mem_assigner(graph, mem_type_to_offset);
  ret = buffer_pool_mem_assigner.Assign();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(buffer_pool_mem_assigner.GetMemOffset(), offset_base +
                                                     AlignMemSize(5600, kMemAlignSize) + kMemAlignSize);

  std::map<std::string, NodePtr> all_nodes;
  for (auto node : graph->GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    all_nodes[node->GetName()] = node;
  }

  {
    auto prefetch = all_nodes.at("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(0));
  }

  {
    auto prefetch = all_nodes.at("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(1));
  }

  {
    auto prefetch = all_nodes.at("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(2));
  }

  {
    auto prefetch = all_nodes.at("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(3));
  }

  {
    auto prefetch = all_nodes.at("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(4));
  }
}

TEST_F(UtestBufferPoolMemAssignerTest, buffer_pool_graph_with_multi_batch_assign_success) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiBatch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiBatch();
  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  std::map<int64_t, size_t> mem_type_to_offset = {{kMemoryTypeHBM, kOffsetHBM},
                                                  {kMemoryTypeP2P, kOffsetP2P}};
  int64_t offset_base = static_cast<int64_t>(kOffsetHBM + kMemAlignSize);
  std::vector<int64_t> expect_offset = {(offset_base + 0),
                                        (offset_base + AlignOutputMemSize(500)),
                                        (offset_base + (AlignOutputMemSize(500) * 2)),
                                        (offset_base + 0),
                                        (offset_base + AlignOutputMemSize(1024))};

  BufferPoolMemAssigner buffer_pool_mem_assigner(graph, mem_type_to_offset);
  ret = buffer_pool_mem_assigner.Assign();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(buffer_pool_mem_assigner.GetMemOffset(), offset_base +
                                                     AlignMemSize(5600, kMemAlignSize) + kMemAlignSize);

  {
    auto prefetch = graph->FindNode("batch_label_128/prefetch1");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(0));
  }

  {
    auto prefetch = graph->FindNode("batch_label_128/prefetch2");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(1));
  }

  {
    auto prefetch = graph->FindNode("batch_label_128/prefetch3");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(2));
  }

  {
    auto prefetch = graph->FindNode("batch_label_128/prefetch4");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(3));
  }

  {
    auto prefetch = graph->FindNode("batch_label_128/prefetch5");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(4));
  }

  {
    auto prefetch = graph->FindNode("batch_label_256/prefetch1");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(0));
  }

  {
    auto prefetch = graph->FindNode("batch_label_256/prefetch2");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(1));
  }

  {
    auto prefetch = graph->FindNode("batch_label_256/prefetch3");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(2));
  }

  {
    auto prefetch = graph->FindNode("batch_label_256/prefetch4");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(3));
  }

  {
    auto prefetch = graph->FindNode("batch_label_256/prefetch5");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> output_offset = prefetch->GetOpDesc()->GetOutputOffset();
    EXPECT_EQ(output_offset.size(), 1);
    EXPECT_EQ(output_offset.at(0), expect_offset.at(4));
  }
}

TEST_F(UtestBufferPoolMemAssignerTest, test_AssignBufferPoolMemory_success) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();
  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  std::map<int64_t, MemoryOffset> memory_offset = {{kMemoryTypeHBM, MemoryOffset(RT_MEMORY_HBM, kOffsetHBM)},
                                                   {kMemoryTypeP2P, MemoryOffset(RT_MEMORY_P2P_HBM, kOffsetP2P)}};

  GraphMemoryAssigner graph_memory_assigner(graph);
  graph_memory_assigner.memory_offset_ = memory_offset;
  ret = graph_memory_assigner.AssignBufferPoolMemory();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestBufferPoolMemAssignerTest, test_AssignBufferPoolMemory_fail) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();
  std::map<int64_t, MemoryOffset> memory_offset = {{kMemoryTypeHBM, MemoryOffset(RT_MEMORY_HBM, kOffsetHBM)},
                                                   {kMemoryTypeP2P, MemoryOffset(RT_MEMORY_P2P_HBM, kOffsetP2P)}};
  {
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    EXPECT_NE(prefetch->GetOpDesc(), nullptr);
    std::vector<int64_t> type_list = {static_cast<int64_t>(RT_MEMORY_P2P_HBM)};
    bool set_attr = ge::AttrUtils::SetListInt(prefetch->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, type_list);
    EXPECT_EQ(set_attr, true);

    GraphMemoryAssigner graph_memory_assigner(graph);
    graph_memory_assigner.memory_offset_ = memory_offset;
    Status ret = graph_memory_assigner.AssignBufferPoolMemory();
    EXPECT_EQ(ret, FAILED);
  }

  {
    std::vector<std::string> node_list = {"prefetch1", "prefetch2", "prefetch3", "prefetch4", "prefetch5"};
    std::vector<int64_t> type_list = {static_cast<int64_t>(RT_MEMORY_L1)};
    for (auto &node_name : node_list) {
      auto prefetch = graph->FindNode(node_name);
      EXPECT_NE(prefetch, nullptr);
      EXPECT_NE(prefetch->GetOpDesc(), nullptr);
      bool set_attr = ge::AttrUtils::SetListInt(prefetch->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, type_list);
      EXPECT_EQ(set_attr, true);
    }
    GraphMemoryAssigner graph_memory_assigner(graph);
    graph_memory_assigner.memory_offset_ = memory_offset;
    Status ret = graph_memory_assigner.AssignBufferPoolMemory();
    EXPECT_EQ(ret, FAILED);
  }
}

TEST_F(UtestBufferPoolMemAssignerTest, test_RefreshEventsWithReuse_success) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();
  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  std::map<std::string, NodePtr> all_nodes;
  for (auto node : graph->GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    all_nodes[node->GetName()] = node;
  }

  Graph2SubGraphInfoList sub_graphs;
  StreamAllocator stream_allocator(graph, sub_graphs);
  stream_allocator.event_num_ = 65520;

  // stream ctrl event
  stream_allocator.AddSendEventId(all_nodes.at("prefetch1"), 30);
  stream_allocator.AddRecvEventId(all_nodes.at("add1"), 30);

  stream_allocator.AddSendEventId(all_nodes.at("prefetch2"), 31);
  stream_allocator.AddRecvEventId(all_nodes.at("add2"), 31);

  stream_allocator.AddSendEventId(all_nodes.at("prefetch3"), 32);
  stream_allocator.AddRecvEventId(all_nodes.at("add3"), 32);

  stream_allocator.AddSendEventId(all_nodes.at("prefetch4"), 33);
  stream_allocator.AddRecvEventId(all_nodes.at("add4"), 33);

  stream_allocator.AddSendEventId(all_nodes.at("add2"), 34);
  stream_allocator.AddRecvEventId(all_nodes.at("prefetch4"), 34);

  stream_allocator.AddSendEventId(all_nodes.at("prefetch5"), 35);
  stream_allocator.AddRecvEventId(all_nodes.at("add5"), 35);

  stream_allocator.AddSendEventId(all_nodes.at("add3"), 36);
  stream_allocator.AddRecvEventId(all_nodes.at("prefetch5"), 36);

  // other event
  stream_allocator.AddSendEventId(all_nodes.at("prefetch1"), 37);
  stream_allocator.AddRecvEventId(all_nodes.at("add5"), 37);


  ret = stream_allocator.RefreshEventsWithReuse();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ((stream_allocator.node_to_send_events_.at(all_nodes.at("prefetch1"))).size(), 2);
  EXPECT_EQ((stream_allocator.node_to_send_events_.at(all_nodes.at("prefetch5"))).size(), 1);
  EXPECT_EQ((stream_allocator.node_to_recv_events_.at(all_nodes.at("prefetch5"))).size(), 1);
  EXPECT_EQ((stream_allocator.node_to_recv_events_.at(all_nodes.at("add5"))).size(), 2);
  EXPECT_EQ(stream_allocator.event_num_, 5);
}

TEST_F(UtestBufferPoolMemAssignerTest, test_RefreshEventsWithReuse_fail) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();

  std::map<std::string, NodePtr> all_nodes;
  for (auto node : graph->GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    all_nodes[node->GetName()] = node;
  }
  std::vector<std::vector<std::string>> event_info = {{"SendTo;add1;0"},
                                                      {"SendTo;add2;1"},
                                                      {"SendTo;add3;2"},
                                                      {"SendTo;add4;3", "RecvFrom;add2;0"},
                                                      {"SendTo;add5;0", "RecvFrom;add3;1"}};

  (void) AttrUtils::SetListStr(all_nodes.at("prefetch1")->GetOpDesc(), ATTR_NAME_EVENT_MULTIPLEXING, event_info[0]);
  (void) AttrUtils::SetListStr(all_nodes.at("prefetch2")->GetOpDesc(), ATTR_NAME_EVENT_MULTIPLEXING, event_info[1]);
  (void) AttrUtils::SetListStr(all_nodes.at("prefetch3")->GetOpDesc(), ATTR_NAME_EVENT_MULTIPLEXING, event_info[2]);
  (void) AttrUtils::SetListStr(all_nodes.at("prefetch4")->GetOpDesc(), ATTR_NAME_EVENT_MULTIPLEXING, event_info[3]);
  (void) AttrUtils::SetListStr(all_nodes.at("prefetch5")->GetOpDesc(), ATTR_NAME_EVENT_MULTIPLEXING, event_info[4]);

  Graph2SubGraphInfoList sub_graphs;
  StreamAllocator stream_allocator(graph, sub_graphs);
  stream_allocator.event_num_ = 65520;

  // Item num of raw event info is invalid
  event_info[0][0] = "SendTo;add1;0;1";
  (void) AttrUtils::SetListStr(all_nodes.at("prefetch1")->GetOpDesc(), ATTR_NAME_EVENT_MULTIPLEXING, event_info[0]);
  Status ret = stream_allocator.RefreshEventsWithReuse();
  EXPECT_EQ(ret, PARAM_INVALID);

  // Event id is invalid argument
  event_info[0][0] = "SendTo;add1;event_id";
  (void) AttrUtils::SetListStr(all_nodes.at("prefetch1")->GetOpDesc(), ATTR_NAME_EVENT_MULTIPLEXING, event_info[0]);
  ret = stream_allocator.RefreshEventsWithReuse();
  EXPECT_EQ(ret, PARAM_INVALID);

  // Event id is out of range
  event_info[0][0] = "SendTo;add1;666666666666666666666666666666666666666";
  (void) AttrUtils::SetListStr(all_nodes.at("prefetch1")->GetOpDesc(), ATTR_NAME_EVENT_MULTIPLEXING, event_info[0]);
  ret = stream_allocator.RefreshEventsWithReuse();
  EXPECT_EQ(ret, PARAM_INVALID);

  // Event id is negative
  event_info[0][0] = "SendTo;add1;-2";
  (void) AttrUtils::SetListStr(all_nodes.at("prefetch1")->GetOpDesc(), ATTR_NAME_EVENT_MULTIPLEXING, event_info[0]);
  ret = stream_allocator.RefreshEventsWithReuse();
  EXPECT_EQ(ret, PARAM_INVALID);

  // Key word is not supported
  event_info[0][0] = "SendToKey;add1;2";
  (void) AttrUtils::SetListStr(all_nodes.at("prefetch1")->GetOpDesc(), ATTR_NAME_EVENT_MULTIPLEXING, event_info[0]);
  ret = stream_allocator.RefreshEventsWithReuse();
  EXPECT_EQ(ret, PARAM_INVALID);
}
}  // namespace ge

