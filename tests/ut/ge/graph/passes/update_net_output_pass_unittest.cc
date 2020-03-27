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

#define protected public
#define private public
#include "graph/passes/update_net_output_pass.h"

#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_builder_utils.h"
#include "omg/omg_inner_types.h"
#undef protected
#undef private

using namespace testing;

namespace ge {
class UtestNodePassesUpdateNetoutputPass : public Test {
 protected:
  UtestNodePassesUpdateNetoutputPass() = default;
};

namespace {
///     net_output1
///       |
///     addn
///    /    \
///  /       \
/// const1   const2
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1);
  auto addn1 = builder.AddNode("addn1", ADDN, 2, 1);
  auto net_output1 = builder.AddNode("net_output", NETOUTPUT, 1, 1);

  builder.AddDataEdge(const1, 0, addn1, 0);
  builder.AddDataEdge(const2, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0, net_output1, 0);
  return builder.GetGraph();
}
}  // namespace

TEST_F(UtestNodePassesUpdateNetoutputPass, update_netoutput_succ) {
  auto graph = BuildGraph1();
  auto net_output = graph->FindNode("net_output");
  EXPECT_NE(net_output, nullptr);

  auto tensor = net_output->GetOpDesc()->GetOutputDesc(0);
  EXPECT_EQ(tensor.GetDataType(), DT_FLOAT);
  EXPECT_EQ(tensor.GetFormat(), FORMAT_NCHW);

  ge::NodePtr node = nullptr;
  ReUpdateNetOutputPass re_update_net_output_pass;
  Status status = re_update_net_output_pass.Run(node);
  EXPECT_EQ(FAILED, status);

  status = re_update_net_output_pass.Run(net_output);
  EXPECT_EQ(SUCCESS, status);

  domi::GetContext().output_type = "FP17";
  status = re_update_net_output_pass.Run(net_output);
  EXPECT_EQ(SUCCESS, status);

  domi::GetContext().output_type = "FP16";
  status = re_update_net_output_pass.Run(net_output);
  EXPECT_EQ(SUCCESS, status);
  auto in_desc = net_output->GetOpDesc()->GetInputDesc(0);
  EXPECT_EQ(in_desc.GetDataType(), DT_FLOAT16);
  auto out_desc = net_output->GetOpDesc()->GetOutputDesc(0);
  EXPECT_EQ(out_desc.GetDataType(), DT_FLOAT16);
}
}  // namespace ge
