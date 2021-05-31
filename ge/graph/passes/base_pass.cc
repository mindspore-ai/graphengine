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

#include "graph/passes/base_pass.h"

#include <queue>
#include <unordered_set>

#include "common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace {
constexpr int kMaxRePassTimes = 10000;
constexpr size_t kMaxOneInNodes = 1000;
// Each iteration, we take about 0.3k memory on the stack, we should change the recursion to loop later
constexpr int kMaxRecursiveDepth = 20;
struct DuringPassNodeSets {
  std::unordered_set<Node *> nodes_seen;
  std::unordered_set<NodePtr> nodes_deleted;
  std::unordered_set<NodePtr> nodes_re_pass;
  std::unordered_set<NodePtr> nodes_re_pass_immediately;
  std::unordered_set<NodePtr> nodes_last;
  std::unordered_set<NodePtr> nodes_suspend;
  std::unordered_set<NodePtr> nodes_resume;
};

void GetAllNodesNoInputEdge(const ComputeGraphPtr &graph, std::deque<NodePtr> &input_edge_nodes,
                            std::unordered_set<Node *> &nodes_seen, std::unordered_set<NodePtr> &nodes_last) {
  nodes_last.clear();
  for (auto &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    size_t in_nums = node->GetInNodes().size();
    if (in_nums == 0) {
      input_edge_nodes.push_back(node);
      nodes_seen.insert(node.get());
    } else if (in_nums > kMaxOneInNodes) {
      nodes_last.insert(node);
    }
  }
}

bool IsAllInNodesAlive(const Node::Vistor<NodePtr> &nodes, const std::unordered_set<NodePtr> &nodes_suspend) {
  return !std::any_of(nodes.begin(), nodes.end(), [&](const NodePtr &n) { return nodes_suspend.count(n) > 0; });
}

void AddNextIterNodes(const Node::Vistor<NodePtr> &nodes, std::deque<NodePtr> &nodes_to_pass,
                      DuringPassNodeSets &during_pass_node_set) {
  auto &nodes_seen = during_pass_node_set.nodes_seen;
  const auto &nodes_last = during_pass_node_set.nodes_last;
  const auto &nodes_suspend = during_pass_node_set.nodes_suspend;
  for (auto &node : nodes) {
    if (node == nullptr) {
      continue;
    }
    if (nodes_last.count(node) != 0) {
      continue;
    }
    if (nodes_suspend.count(node) > 0) {
      GELOGD("The node %s has suspend by pass, skip it.", node->GetName().c_str());
      continue;
    }

    bool all_in_nodes_alive = IsAllInNodesAlive(node->GetInAllNodes(), nodes_suspend);
    bool all_in_nodes_seen = node->IsAllInNodesSeen(nodes_seen);
    if (all_in_nodes_seen && all_in_nodes_alive && nodes_seen.insert(node.get()).second) {
      nodes_to_pass.push_back(node);
    }
  }
}

void AddRepassNodes(DuringPassNodeSets &during_pass_node_set, std::deque<NodePtr> &nodes) {
  for (const auto &node : during_pass_node_set.nodes_re_pass_immediately) {
    GELOGD("The node %s will be re-pass immediately.", node->GetName().c_str());
    nodes.push_front(node);
  }
  during_pass_node_set.nodes_re_pass_immediately.clear();
}

void AddResumeNodes(DuringPassNodeSets &during_pass_node_set, std::deque<NodePtr> &nodes) {
  for (auto &node : during_pass_node_set.nodes_resume) {
    const auto &it = during_pass_node_set.nodes_suspend.find(node);
    if (it != during_pass_node_set.nodes_suspend.end()) {
      during_pass_node_set.nodes_suspend.erase(node);
      GELOGD("The node %s resumed by pass.", node->GetName().c_str());
      nodes.push_back(node);
    } else {
      GELOGW("The node %s not suspend, drop from resumed", node->GetName().c_str());
    }
  }
  during_pass_node_set.nodes_resume.clear();
}

void PushToSuspendNodes(DuringPassNodeSets &during_pass_node_set, const std::string &pass_name,
                        const std::unordered_set<NodePtr> &nodes_suspend,
                        const std::unordered_set<NodePtr> &nodes_resume) {
  for (const auto &node : nodes_suspend) {
    GELOGD("The iteration suspend of node %s has been set by pass %s", node->GetName().c_str(), pass_name.c_str());
    during_pass_node_set.nodes_suspend.emplace(node);
  }

  for (const auto &node : nodes_resume) {
    GELOGD("The iteration suspend of node %s has been resumed by pass %s", node->GetName().c_str(), pass_name.c_str());
    during_pass_node_set.nodes_resume.emplace(node);
  }
}

void PushToRePassIfSeen(NodePtr &node, const std::pair<std::string, BaseNodePass *> &name_to_pass,
                        std::unordered_set<Node *> &nodes_seen, const std::unordered_set<NodePtr> &nodes_to_re_pass,
                        std::unordered_set<NodePtr> &nodes_re_pass) {
  for (const auto &node_to_re_pass : nodes_to_re_pass) {
    if (node_to_re_pass == nullptr) {
      GELOGW("Found null re-pass node when executing %s on node %s type %s", name_to_pass.first.c_str(),
             node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    if (nodes_seen.count(node_to_re_pass.get()) > 0 || node_to_re_pass->IsAllInNodesSeen(nodes_seen)) {
      GELOGD("The node %s will be re-pass.", node_to_re_pass->GetName().c_str());
      nodes_re_pass.insert(node_to_re_pass);
    } else {
      GELOGD("The node %s are not all seen, don't set repass this time", node_to_re_pass->GetName().c_str());
    }
  }
}

Status RunPasses(NodePtr &node, const NamesToPass &names_to_passes, DuringPassNodeSets &during_pass_node_set) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return FAILED;
  }
  GELOGD("Begin to run pass for node %s", node->GetName().c_str());
  for (const auto &name_to_pass : names_to_passes) {
    if (name_to_pass.second == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] There is null pointer in passes(%s), skip it", name_to_pass.first.c_str());
      continue;
    }

    GELOGD("Begin to run pass %s for node %s", name_to_pass.first.c_str(), node->GetName().c_str());
    name_to_pass.second->init();
    auto result = name_to_pass.second->Run(node);
    if (result != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "process pass %s on node:%s failed, ret:%u",
                        name_to_pass.first.c_str(), node->GetName().c_str(), result);
      GELOGE(INTERNAL_ERROR, "[Process][Pass] %s on node %s failed, result "
             "%u, the passes will be terminated immediately.",
             name_to_pass.first.c_str(), node->GetName().c_str(), result);
      return result;
    }

    const auto &nodes_to_re_pass = name_to_pass.second->GetNodesNeedRePass();
    PushToRePassIfSeen(node, name_to_pass, during_pass_node_set.nodes_seen, nodes_to_re_pass,
                       during_pass_node_set.nodes_re_pass);

    const auto &nodes_to_re_pass_immediately = name_to_pass.second->GetNodesNeedRePassImmediately();
    PushToRePassIfSeen(node, name_to_pass, during_pass_node_set.nodes_seen, nodes_to_re_pass_immediately,
                       during_pass_node_set.nodes_re_pass_immediately);

    PushToSuspendNodes(during_pass_node_set, name_to_pass.first,
                       name_to_pass.second->GetNodesSuspend(), name_to_pass.second->GetNodesResume());

    const auto &nodes_deleted_by_pass = name_to_pass.second->GetNodesDeleted();
    during_pass_node_set.nodes_deleted.insert(nodes_deleted_by_pass.begin(), nodes_deleted_by_pass.end());
    if (nodes_deleted_by_pass.count(node) > 0) {
      GELOGD("The node %s was deleted by pass %s, stop the remain passes", node->GetName().c_str(),
             name_to_pass.first.c_str());
      break;
    }
  }

  return SUCCESS;
}

void SetFlagOption(NodePassOption option, NamesToPass names_to_pass) {
  for (auto &name_to_pass : names_to_pass) {
    name_to_pass.second->SetOption(option, "");
  }
}

void ClearOption(NamesToPass names_to_pass) {
  for (auto &name_to_pass : names_to_pass) {
    name_to_pass.second->ClearOptions();
  }
}
}  // namespace

Status BaseNodePass::IsolateAndDeleteNode(NodePtr &node, const std::vector<int> &io_map) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return FAILED;
  }
  GELOGI("Prepare to isolate and delete node, name:%s, type:%s.", node->GetName().c_str(),
         node->GetType().c_str());
  ComputeGraphPtr graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "The owner graph of node:%s must not be null.", node->GetName().c_str());
    GELOGE(FAILED, "[Get][OwnerComputeGraph] failed, The owner graph of node:%s must not be null.",
           node->GetName().c_str());
    return FAILED;
  }

  AddRePassNodesWithInOut(node);

  if (GraphUtils::IsolateNode(node, io_map) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Isolate Node:%s failed", node->GetName().c_str());
    GELOGE(FAILED, "[Isolate][Node] %s failed.", node->GetName().c_str());
    return FAILED;
  }

  if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "call RemoveNodeWithoutRelink for node:%s failed.", node->GetName().c_str());
    GELOGE(FAILED, "[Call][RemoveNodeWithoutRelink] for node:%s failed.", node->GetName().c_str());
    return FAILED;
  }

  AddNodeDeleted(node);
  return SUCCESS;
}

Status GEPass::Run(const NamesToPass &names_to_passes) {
  if (graph_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph_ is nullptr, check invalid.");
    GELOGE(INTERNAL_ERROR, "[Check][Param] The graph is nullptr");
    return INTERNAL_ERROR;
  }
  if (names_to_passes.empty()) {
    GELOGW("No passes input, the GEPass will do nothing");
    return INTERNAL_ERROR;
  }

  if (depth_ > kMaxRecursiveDepth) {
    GELOGE(PARAM_INVALID,
        "[Check][Param] The pass for root graph %s will be terminated because too many nesting"
        " levels(%d) of subgraphs, last subgraph is %s",
        root_graph_->GetName().c_str(), depth_, graph_->GetName().c_str());
    return PARAM_INVALID;
  }

  return RunPassesOneGraph(names_to_passes);
}

Status GEPass::RunPassesOneGraph(const NamesToPass &names_to_passes) {
  GELOGD("Begin to run pass on graph, passes count %zu", names_to_passes.size());
  std::deque<NodePtr> nodes;
  DuringPassNodeSets during_pass_node_set;
  GetAllNodesNoInputEdge(graph_, nodes, during_pass_node_set.nodes_seen, during_pass_node_set.nodes_last);
  GELOGD("Start points count %zu", nodes.size());
  int re_pass_times = 0;

  do {
    for (auto &node : during_pass_node_set.nodes_re_pass) {
      nodes.push_back(node);
      during_pass_node_set.nodes_seen.insert(node.get());
    }
    during_pass_node_set.nodes_re_pass.clear();

    while (!nodes.empty()) {
      NodePtr node = nodes.front();
      nodes.pop_front();

      (void)during_pass_node_set.nodes_re_pass.erase(node);
      GE_IF_BOOL_EXEC(node == nullptr, GELOGW("node is null"); continue);
      if (during_pass_node_set.nodes_deleted.count(node) > 0) {
        GELOGD("The node %s was deleted before, skip it.", node->GetName().c_str());
        continue;
      }
      if (during_pass_node_set.nodes_suspend.count(node) > 0) {
        GELOGD("The node %s has been added to suspend-iteration nodes list, the iteration of it will be suspend.",
               node->GetName().c_str());
        continue;
      }

      AddNextIterNodes(node->GetOutNodes(), nodes, during_pass_node_set);

      auto ret = RunPasses(node, names_to_passes, during_pass_node_set);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Process][Passes] on node %s type %s failed, error code:%u",
               node->GetName().c_str(), node->GetType().c_str(), ret);
        return ret;
      }

      bool has_sub_graph = false;
      ret = RunPassesOnSubGraph(node, names_to_passes, has_sub_graph);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Run][Passes] on the sub graph of node %s failed", node->GetName().c_str());
        return ret;
      }

      if (has_sub_graph) {
        GELOGD("There are subgraphs on node %s, run passes for for the second time", node->GetName().c_str());
        SetFlagOption(kOptimizeAfterSubGraph, names_to_passes);
        ret = RunPasses(node, names_to_passes, during_pass_node_set);
        if (ret != SUCCESS) {
          GELOGE(ret, "[Process][Passes] on node %s type %s failed, error code: %u",
                 node->GetName().c_str(), node->GetType().c_str(), ret);
          return ret;
        }

        // There is only one option scene, so set and clear options around the `RunPasses` func.
        // if there are more than one scene to set options, the `ClearOption` function
        // should be called each time at the begin of the iteration
        ClearOption(names_to_passes);
      }

      AddRepassNodes(during_pass_node_set, nodes);
      AddResumeNodes(during_pass_node_set, nodes);
    }

    for (auto &node : during_pass_node_set.nodes_last) {
      bool all_in_nodes_seen = node->IsAllInNodesSeen(during_pass_node_set.nodes_seen);
      if (all_in_nodes_seen && during_pass_node_set.nodes_seen.insert(node.get()).second) {
        nodes.push_back(node);
      }
    }
    during_pass_node_set.nodes_last.clear();
  } while ((!during_pass_node_set.nodes_re_pass.empty() || !nodes.empty()) && ++re_pass_times < kMaxRePassTimes);

  if (re_pass_times == kMaxRePassTimes) {
    GELOGW("re_pass_times should not come to %d", kMaxRePassTimes);
  }
  GELOGD("All passes runs end");

  return SUCCESS;
}
Status GEPass::RunPassesOnSubGraph(const NodePtr &node, const NamesToPass &names_to_passes, bool &has_sub_graph) {
  auto sub_graph_names = node->GetOpDesc()->GetSubgraphInstanceNames();
  has_sub_graph = false;
  for (const auto &name : sub_graph_names) {
    auto graph = root_graph_->GetSubgraph(name);
    if (graph == nullptr) {
      GELOGW("Can not find the sub graph %s from node %s, the pass-process will skip it",
          name.c_str(), node->GetName().c_str());
      continue;
    }
    has_sub_graph = true;
    GELOGI("Begin to run passes on the sub graph %s of node %s", name.c_str(), node->GetName().c_str());
    GEPass pass(graph, root_graph_, depth_ + 1);
    auto ret = pass.Run(names_to_passes);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Run][Passes] for sub graph:%s from node:%s failed", name.c_str(), node->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}
}  // namespace ge
