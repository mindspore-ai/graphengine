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

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"
#include "new_op_test_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/detail/model_serialize_imp.h"
#include "proto/ge_ir.pb.h"

#define private public
#define protected public
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/model_serialize.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "common/properties_manager.h"
#include "common/op/ge_op_utils.h"
#include <cce/taskdown_api.h>
#include "runtime/dev.h"
#include "runtime/kernel.h"
#include "cce/fwk_adpt_struct.h"
#undef private
#undef protected

using namespace std;
using namespace testing;

namespace ge {
class UtestModelManagerTaskBuilder : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  /// data    weight
  ///  |      |  | |
  ///  |-conv-|  | |
  ///      |     | |
  ///      conv2d  |
  ///         |    |
  ///         |-resApply

  void BuildGraph(ComputeGraphPtr graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA1", "data");
    OpDescPtr weight = std::make_shared<OpDesc>("WEIGHT", "weight");
    OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "conv");
    OpDescPtr conv_2D = std::make_shared<OpDesc>("conv_2D", "conv2d");
    OpDescPtr res_apply_op = std::make_shared<OpDesc>("res_apply_op", "resapply");
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    int32_t blockSize = 4096;

    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 1);
    data->AddOutputDesc(out_desc);

    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 2);
    weight->AddOutputDesc(out_desc);

    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 1);
    conv_op->AddInputDesc(out_desc);
    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 2);
    conv_op->AddInputDesc(out_desc);
    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 3);
    conv_op->AddOutputDesc(out_desc);

    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 3);
    conv_2D->AddInputDesc(out_desc);
    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 2);
    conv_2D->AddInputDesc(out_desc);
    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 4);
    conv_2D->AddOutputDesc(out_desc);

    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 4);
    res_apply_op->AddInputDesc(out_desc);
    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 1);
    res_apply_op->AddInputDesc(out_desc);
    ge::TensorUtils::SetDataOffset(out_desc, blockSize * 5);
    res_apply_op->AddOutputDesc(out_desc);

    NodePtr data_node = graph->AddNode(data);
    NodePtr weigth_node = graph->AddNode(weight);
    NodePtr conv_node = graph->AddNode(conv_op);
    NodePtr conv_2D_node = graph->AddNode(conv_2D);
    NodePtr res_node = graph->AddNode(res_apply_op);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weigth_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), conv_2D_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weigth_node->GetOutDataAnchor(0), conv_2D_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_2D_node->GetOutDataAnchor(0), res_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weigth_node->GetOutDataAnchor(0), res_node->GetInDataAnchor(1));
    return;
  }
};
}  // namespace ge
