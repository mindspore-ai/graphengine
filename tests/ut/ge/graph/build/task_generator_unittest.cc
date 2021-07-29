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
#include <memory>

#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "omg/omg_inner_types.h"
#include "../passes/graph_builder_utils.h"

#define protected public
#define private public
#include "init/gelib.h"
#include "ge/opskernel_manager/ops_kernel_builder_manager.h"
#include "graph/build/task_generator.h"
#include "graph/manager/graph_mem_manager.h"
#include "graph/manager/graph_var_manager.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
using namespace ge;
namespace {
const char *const kIsInputVar = "INPUT_IS_VAR";
const char *const kIsOutputVar = "OUTPUT_IS_VAR";
const char *const kKernelInfoNameHccl = "ops_kernel_info_hccl";
}  // namespace
class UtestTaskGeneratorTest : public testing::Test {
 public:
  struct FakeOpsKernelBuilder : OpsKernelBuilder {
    FakeOpsKernelBuilder(){};

   private:
    Status Initialize(const map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    Status CalcOpRunningParam(Node &node) override {
      return SUCCESS;
    };
    Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
      domi::TaskDef task_def;
      tasks.push_back(task_def);
      return SUCCESS;
    };
  };

  struct FakeOpsKernelInfoStore : OpsKernelInfoStore {
    FakeOpsKernelInfoStore() = default;

   private:
    Status Initialize(const std::map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override {
      return true;
    };
    void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override{};
  };

  ge::ComputeGraphPtr BuildGraphFpProfiling() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "phony", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto netoutput = builder.AddNode("netoutput", "NetOutput", 2, 0);
    auto op_desc = data->GetOpDesc();
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
    op_desc->SetOpKernelLibName("GE");
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddDataEdge(addn1, 0, netoutput, 0);
    return builder.GetGraph();
  }
  ge::ComputeGraphPtr BuildGraphBpProfiling() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "phony", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto netoutput = builder.AddNode("Node_Output", "NetOutput", 2, 0);
    auto data_desc = data->GetOpDesc();
    (void)AttrUtils::SetStr(data_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
    data_desc->SetOpKernelLibName("GE");
    auto output_desc = netoutput->GetOpDesc();
    output_desc->SetOpKernelLibName("output");
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddControlEdge(addn1, netoutput);
    return builder.GetGraph();
  }
  ge::ComputeGraphPtr BuildGraphWithVar(int64_t session_id) {
    // init
    MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
    VarManager::Instance(session_id)->Init(0, 0, 0, 0);
    ge::ut::GraphBuilder builder("graph");
    auto var_input = builder.AddNode("var", "Variable", 1, 1);
    auto const_input = builder.AddNode("const", "Const", 1, 1);
    auto assign = builder.AddNode("assgin", "Assign", 2, 1);
    // add link
    builder.AddDataEdge(var_input, 0, assign, 0);
    builder.AddDataEdge(const_input, 0, assign, 1);
    // set offset
    var_input->GetOpDesc()->SetOutputOffset({10000});
    const_input->GetOpDesc()->SetOutputOffset({1000});
    assign->GetOpDesc()->SetInputOffset({10100, 1000});
    assign->GetOpDesc()->SetOutputOffset({10100});
    // set inner offset
    int64_t inner_offset = 100;
    ge::AttrUtils::SetInt(assign->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset);
    ge::AttrUtils::SetInt(assign->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset);
    // add var addr
    VarManager::Instance(session_id)->var_resource_->var_offset_map_.emplace(10000, RT_MEMORY_HBM);

    return builder.GetGraph();
  }
  ge::ComputeGraphPtr BuildHcclGraph() {
    ge::ut::GraphBuilder builder("graph");
    auto hccl_node = builder.AddNode("hccl_phony_node", "HCCL_PHONY", 0, 0);
    auto op_desc = hccl_node->GetOpDesc();
    op_desc->SetOpKernelLibName(kKernelInfoNameHccl);
    op_desc->SetStreamId(0);
    return builder.GetGraph();
  }

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTaskGeneratorTest, AutoFindFpOpIndex) {
  auto graph = BuildGraphFpProfiling();
  TaskGenerator task_generator(nullptr, 0);
  ProfilingPoint profiling_point;
  profiling_point.fp_index = -1;
  EXPECT_EQ(task_generator.AutoFindFpOpIndex(graph, profiling_point), SUCCESS);
  // addn1 is fp
  EXPECT_EQ(profiling_point.fp_index, 2);
}

TEST_F(UtestTaskGeneratorTest, FindLastBpFromBpNode) {
  auto graph = BuildGraphBpProfiling();
  TaskGenerator task_generator(nullptr, 0);
  auto net_output = graph->FindNode("Node_Output");
  // netoutput has no data input, return default value 0
  uint32_t bp_index = 0;
  EXPECT_EQ(task_generator.FindLastBpFromBpNode(graph, net_output, bp_index), 0);
  EXPECT_EQ(bp_index, 2);
}

TEST_F(UtestTaskGeneratorTest, UpdateOpIsVarAttr) {
  int64_t session_id = 0;
  ge::ComputeGraphPtr graph = BuildGraphWithVar(session_id);
  graph->SetSessionID(session_id);
  TaskGenerator task_generator(nullptr, 0);
  auto assign = graph->FindNode("assgin");
  task_generator.UpdateOpIsVarAttr(assign->GetOpDesc(), session_id);
  // input
  vector<bool> input_var;
  AttrUtils::GetListBool(assign->GetOpDesc(), kIsInputVar, input_var);
  EXPECT_EQ(input_var.size(), 2);
  EXPECT_EQ(input_var[0], true);
  EXPECT_EQ(input_var[1], false);
  // output
  vector<bool> output_var;
  AttrUtils::GetListBool(assign->GetOpDesc(), kIsOutputVar, output_var);
  EXPECT_EQ(output_var.size(), 1);
  EXPECT_EQ(output_var[0], true);

  MemManager::Instance().Finalize();
}

TEST_F(UtestTaskGeneratorTest, AutoFindBpOpIndex) {
  auto graph = BuildGraphBpProfiling();
  TaskGenerator task_generator(nullptr, 0);
  auto net_output = graph->FindNode("Node_Output");
  ProfilingPoint profiling_point;
  vector<uint32_t> all_reduce_nodes;
  EXPECT_EQ(task_generator.AutoFindBpOpIndex(graph, profiling_point, all_reduce_nodes), SUCCESS);

  auto output_desc = net_output->GetOpDesc();
  output_desc->SetType("HcomAllReduce");
  output_desc->SetName("hcom");
  EXPECT_EQ(task_generator.AutoFindBpOpIndex(graph, profiling_point, all_reduce_nodes), SUCCESS);
}

TEST_F(UtestTaskGeneratorTest, GenerateTask) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);

  OpsKernelInfoStorePtr ops_kernel_info_store_ptr = MakeShared<FakeOpsKernelInfoStore>();
  instance_ptr->opsManager_.ops_kernel_store_.insert(make_pair(kKernelInfoNameHccl, ops_kernel_info_store_ptr));

  OpsKernelBuilderManager &builder_manager_instance_ptr = ge::OpsKernelBuilderManager::Instance();
  OpsKernelBuilderPtr fake_builder = MakeShared<FakeOpsKernelBuilder>();
  builder_manager_instance_ptr.ops_kernel_builders_[kKernelInfoNameHccl] = fake_builder;

  auto graph = BuildHcclGraph();
  TaskGenerator task_generator(nullptr, 0);
  RunContext run_context;
  run_context.graphStreamList.push_back(static_cast<void *>(ops_kernel_info_store_ptr.get()));
  vector<uint32_t> all_reduce_nodes;
  vector<domi::TaskDef> task_def_list;
  map<uint32_t, string> op_name_map;

  EXPECT_EQ(task_generator.GenerateTask(run_context, graph, task_def_list, op_name_map), SUCCESS);
  EXPECT_EQ(task_def_list.size(), 1);
  EXPECT_EQ(task_def_list[0].ops_kernel_store_ptr(), reinterpret_cast<uintptr_t>(ops_kernel_info_store_ptr.get()));
}