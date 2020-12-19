#ifndef MAIN_TUNING_UTILS_H
#define MAIN_TUNING_UTILS_H

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <queue>
#include <mutex>

#include <graph/anchor.h>
#include <graph/detail/attributes_holder.h>
#include <graph/ge_tensor.h>
#include <graph/graph.h>
#include <graph/model.h>
#include <graph/node.h>
#include <graph/utils/graph_utils.h>
#include <graph/utils/type_utils.h>

#include "framework/common/debug/ge_log.h"
#include "utils/attr_utils.h"
#include "utils/node_utils.h"
#include "external/ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
namespace ge {
// Configure build mode, default value is "normal"
const char *const BUILD_MODE = "ge.buildMode";
const char *const BUILD_STEP = "ge.buildStep";
// Configure tuning path
const char *const TUNING_PATH = "ge.tuningPath";
// for interface: aclgrphBuildModel
const std::set<std::string> ir_builder_supported_options_for_lx_fusion = {
  BUILD_MODE,
  BUILD_STEP,
  TUNING_PATH
};

// Build model
const char *const BUILD_MODE_NORMAL = "normal";
const char *const BUILD_MODE_TUNING = "tuning";
const char *const BUILD_MODE_BASELINE = "baseline";
const std::set<std::string> build_mode_options = {
    BUILD_MODE_NORMAL,
    BUILD_MODE_TUNING,
    BUILD_MODE_BASELINE
};

// Build step
const char *const BUILD_STEP_BEFORE_UB_MATCH = "before_ub_match";
const char *const BUILD_STEP_AFTER_UB_MATCH = "after_ub_match";
const char *const BUILD_STEP_AFTER_BUILDER = "after_builder";
const char *const BUILD_STEP_AFTER_BUILDER_SUB = "after_builder_sub";
const char *const BUILD_STEP_AFTER_MERGE = "after_merge";
const std::set<std::string> build_step_options = {
    BUILD_STEP_BEFORE_UB_MATCH,
    BUILD_STEP_AFTER_UB_MATCH,
    BUILD_STEP_AFTER_BUILDER,
    BUILD_STEP_AFTER_BUILDER_SUB,
    BUILD_STEP_AFTER_MERGE
};

using SubgraphCreateOutNode = std::unordered_map<ComputeGraphPtr, NodePtr>;
using NodetoNodeMap = std::unordered_map<NodePtr, NodePtr>;
using NodeVec = std::vector<NodePtr>;
using NodeNametoNodeNameMap = std::unordered_map<std::string, std::string>;
using NodetoNodeNameMap = std::unordered_map<NodePtr, std::string>;
class TuningUtils {
 public:
  TuningUtils() = default;
  ~TuningUtils() = default;
  // Dump all the subgraphs and modify
  // the subgraphs in them to be executable subgraphs if exe_flag is true
  // `tuning_path` means path to save the graphs
  static graphStatus ConvertGraphToFile(std::vector<ComputeGraphPtr> tuning_subgraphs,
                                        std::vector<ComputeGraphPtr> non_tuning_subgraphs = {},
                                        bool exe_flag = false,
                                        const std::string &path = "",
                                        const std::string &user_path = "");
  // Recovery `graph` from graph dump files configured in options
  static graphStatus ConvertFileToGraph(const map<int64_t, string> &options, ge::Graph &graph);

 private:
  // part 1
  struct HelpInfo {
    int64_t index;
    bool exe_flag;
    bool is_tuning_graph;
    const std::string &path;
    const std::string &user_path;
  };
  static graphStatus MakeExeGraph(ComputeGraphPtr &exe_graph,
                                  const HelpInfo& help_info);
  static graphStatus HandlePld(NodePtr &node);
  static graphStatus HandleEnd(NodePtr &node);
  static graphStatus ChangePld2Data(NodePtr &node, NodePtr &data_node);
  static graphStatus ChangeEnd2NetOutput(NodePtr &node, NodePtr &out_node);
  static graphStatus LinkEnd2NetOutput(NodePtr &node, NodePtr &out_node);
  static graphStatus CreateDataNode(NodePtr &node, NodePtr &data_node);
  static graphStatus CreateNetOutput(NodePtr &node, NodePtr &out_node);
  static graphStatus AddAttrToDataNodeForMergeGraph(const NodePtr &pld, NodePtr &data_node);
  static graphStatus AddAttrToNetOutputForMergeGraph(const NodePtr &end, NodePtr &out_node);
  static void DumpGraphToPath(ComputeGraphPtr &exe_graph, int64_t index,
                              bool is_tuning_graph, std::string path);

  static SubgraphCreateOutNode create_output_;
  // part 2
  static graphStatus MergeAllSubGraph(std::vector<ComputeGraphPtr> &graphs,
                                      ComputeGraphPtr &graph);
  static graphStatus MergeSubGraph(ComputeGraphPtr &graph);
  // Deletes new data and output nodes added by call `MakeExeGraph()` func in part 1
  static graphStatus RemoveDataNetoutputEdge(ComputeGraphPtr &graph);
  static graphStatus GetInAndOutAnchorPair(NodePtr &data_node,
                                           NodePtr &out_node,
                                           AnchorPtr &dest_in_anchor,
                                           AnchorPtr &src_out_anchor);
  static graphStatus HandleContinuousInputNodeNextData(NodePtr &node);
  static NodeNametoNodeNameMap data_2_netoutput_;
  static NodetoNodeNameMap data_node_2_netoutput_;
  static NodetoNodeMap data_node_2_netoutput_node_;
  static NodeVec netoutput_nodes_;
  static NodeVec merged_graph_nodes_;
  static std::mutex mutex_;
  // for debug
  static std::string PrintCheckLog();
  static std::string GetNodeNameByAnchor(const Anchor *anchor);
};
}
#endif //MAIN_TUNING_UTILS_H
