/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "graph/passes/infershape_pass.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "analyzer/analyzer.h"
#include "framework/common/util.h"
#include "graph/shape_refiner.h"
#include "graph/utils/graph_utils.h"

namespace ge {
Status InferShapePass::Run(NodePtr &node) {
  // kOptimizeAfterSubGraph exist means after subgraph
  auto ret = ShapeRefiner::InferShapeAndType(node, !OptionExists(kOptimizeAfterSubGraph));
  if (ret != GRAPH_SUCCESS) {
    // select INFERSHAPE failed info
    auto graph = node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(graph);
    auto root_graph = ge::GraphUtils::FindRootGraph(graph);
    GE_CHECK_NOTNULL(root_graph);
    analyzer::DataInfo analyze_info{root_graph->GetSessionID(), root_graph->GetGraphID(),
                                  analyzer::INFER_SHAPE, node, "InferShapeFailed!"};
    (void)Analyzer::GetInstance()->DoAnalyze(analyze_info);
    (void)Analyzer::GetInstance()->SaveAnalyzerDataToFile(root_graph->GetSessionID(),
                                                          root_graph->GetGraphID());

    GELOGE(GE_GRAPH_INFERSHAPE_FAILED, "infershape failed. node: %s", node->GetName().c_str());
    return GE_GRAPH_INFERSHAPE_FAILED;
  }
  bool need_repass = false;
  auto has_attr = AttrUtils::GetBool(node->GetOpDesc(), "need_infer_again_", need_repass);
  if (has_attr) {
    if (!OptionExists(kOptimizeAfterSubGraph)) {
      return SUCCESS;
    }
    if (need_repass) {
      AddImmediateRePassNode(node);
      GELOGD("Node %s need repass immediately.", node->GetName().c_str());
    } else {
      // clear attr on while
      node->GetOpDesc()->DelAttr("need_infer_again_");
    }
  }
  return SUCCESS;
}
}  // namespace ge
