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
#include "graph/build/task_generator.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
using namespace ge;

class UtestTaskGeneratorTest : public testing::Test {
 public:
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
