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

#include "graph/build/graph_builder.h"
#include "common/ge/ge_util.h"
#include "common/helper/model_helper.h"
#include "graph/build/logical_stream_allocator.h"
#include "graph/build/run_context.h"
#include "graph/build/stream_graph_optimizer.h"
#include "graph/common/ge_call_wrapper.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/passes/mark_same_addr_pass.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "model/ge_model.h"
#include "graph/ge_context.h"
#include "opskernel_manager/ops_kernel_builder_manager.h"
#include "graph/utils/op_desc_utils.h"

using domi::BuildMode;

namespace {
const int32_t kInvalidPerfLevel = -1;
enum NodeType { kSubgraphData, kSubgraphNode, kOthers };
}  // namespace
namespace ge {
NodeType TransferNodeType(const NodePtr &node) {
  const std::string type = node->GetType();
  if (type == ge::DATA) {
    if (node->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
      GELOGD("access src data node:%s", node->GetName().c_str());
      return kOthers;
    }
    GELOGD("access subgraph input node:%s", node->GetName().c_str());
    return kSubgraphData;
  } else if (type == PARTITIONEDCALL) {
    GELOGD("access subgraph node:%s", node->GetName().c_str());
    return kSubgraphNode;
  }
  GELOGD("access other node:%s", node->GetName().c_str());
  return kOthers;
}

Status HandleSubgraphNode(NodePtr &src_node, OutDataAnchorPtr &src_out_anchor) {
  auto subgraph = NodeUtils::GetSubgraph(*src_node, 0);
  GE_CHECK_NOTNULL(subgraph);
  const NodePtr &net_output_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  GE_CHECK_NOTNULL(net_output_node);
  const InDataAnchorPtr &in_data_anchor = net_output_node->GetInDataAnchor(src_out_anchor->GetIdx());
  GE_CHECK_NOTNULL(in_data_anchor);
  const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);

  src_node = peer_out_anchor->GetOwnerNode();
  src_out_anchor = peer_out_anchor;
  return SUCCESS;
}

Status HandleSubgraphDataNode(NodePtr &src_node, OutDataAnchorPtr &src_out_anchor) {
  uint32_t index = 0;
  if (!AttrUtils::GetInt(src_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, index)) {
    GELOGE(FAILED, "Get attr ATTR_NAME_PARENT_NODE_INDEX failed, node:%s.", src_node->GetName().c_str());
    return FAILED;
  }
  const NodePtr &parent_node = src_node->GetOwnerComputeGraph()->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  const InDataAnchorPtr &in_data_anchor = parent_node->GetInDataAnchor(index);
  GE_CHECK_NOTNULL(in_data_anchor);
  const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);

  src_node = peer_out_anchor->GetOwnerNode();
  src_out_anchor = peer_out_anchor;
  return SUCCESS;
}

GraphBuilder::GraphBuilder() : build_mode_(BuildMode::GEN_TASK_WITH_FUSION), hcom_parallel_(false) {}

void GraphBuilder::SetOptions(const ge::GraphManagerOptions &options) {
  stream_max_parallel_num_ = options.stream_max_parallel_num;
  hcom_parallel_ = options.hcom_parallel;

  if (options.perf_level == kInvalidPerfLevel) {
    build_mode_ = static_cast<int>(BuildMode::GEN_TASK_WITH_FUSION);
  } else {
    build_mode_ = options.perf_level;
  }
}

Status GraphBuilder::CalcOpParam(const ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  auto instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "GraphBuilder: GE is not initialized");
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  for (const auto &node_ptr : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    GE_CHECK_NOTNULL(node_ptr->GetOpDesc());
    std::string kernel_lib_name = node_ptr->GetOpDesc()->GetOpKernelLibName();
    if (kernel_lib_name.empty()) {
      // reset op kernel lib
      (void)instance_ptr->DNNEngineManagerObj().GetDNNEngineName(node_ptr);
      kernel_lib_name = node_ptr->GetOpDesc()->GetOpKernelLibName();
      if (kernel_lib_name.empty()) {
        GELOGE(INTERNAL_ERROR, "Get node:%s(%s) kernel lib failed.", node_ptr->GetName().c_str(),
               node_ptr->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }

    auto ret = SetInputSize(node_ptr);
    if (ret != SUCCESS) {
      GELOGE(ret, "Set node inputDesc size failed, node name is %s", node_ptr->GetName().c_str());
      return ret;
    }

    ret = OpsKernelBuilderManager::Instance().CalcOpRunningParam(*node_ptr);
    if (ret != SUCCESS) {
      GELOGE(ret, "Calculate op running param failed, node name is %s", node_ptr->GetName().c_str());
      return ret;
    }
    GE_CHK_STATUS_RET(AddOutputMemTypeForNode(node_ptr));
  }

  auto parent_node = graph->GetParentNode();
  if (parent_node == nullptr) {
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(UpdateParentNodeOutputSize(graph, parent_node));
  GELOGI("Success to calculate op running param.");
  return SUCCESS;
}

Status GraphBuilder::UpdateParentNodeOutputSize(const ge::ComputeGraphPtr &graph, ge::NodePtr &parent_node_ptr) {
  GELOGI("Begin to update parent node[%s] of graph[%s] output size.", parent_node_ptr->GetName().c_str(),
         graph->GetName().c_str());
  auto parent_op_desc = parent_node_ptr->GetOpDesc();
  GE_CHECK_NOTNULL(parent_op_desc);
  bool is_unknown_shape = graph->GetGraphUnknownFlag();
  if (is_unknown_shape) {
    GELOGI("Current graph[%s] is unknown, no need to update parent node[%s] output size.", graph->GetName().c_str(),
           parent_node_ptr->GetName().c_str());
    return SUCCESS;
  }
  for (const auto &node_ptr : graph->GetDirectNode()) {
    if (node_ptr->GetType() != NETOUTPUT) {
      continue;
    }
    auto op_desc = node_ptr->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (const auto &in_data_anchor : node_ptr->GetAllInDataAnchors()) {
      auto index = in_data_anchor->GetIdx();
      ge::GeTensorDesc desc_temp = op_desc->GetInputDesc(index);
      uint32_t parent_index = 0;
      if (!AttrUtils::GetInt(desc_temp, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
        GELOGI("NetOutput input tensor %d, attr %s not found.", index, ATTR_NAME_PARENT_NODE_INDEX.c_str());
        continue;
      }

      int64_t size = 0;
      GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(desc_temp, size) != SUCCESS, GELOGI("Get size failed!"));
      ge::GeTensorDesc parent_desc_temp = parent_op_desc->GetOutputDesc(parent_index);
      ge::TensorUtils::SetSize(parent_desc_temp, size);
      GE_CHK_STATUS_RET(parent_op_desc->UpdateOutputDesc(parent_index, parent_desc_temp));
      GELOGI("Update parent node[%s] output index[%u] to size[%ld].", parent_node_ptr->GetName().c_str(), parent_index,
             size);
    }
  }
  return SUCCESS;
}

Status GraphBuilder::Build(ComputeGraphPtr &comp_graph, std::vector<SubGraphInfoPtr> &subgraph_ptr_list,
                           GeRootModelPtr &ge_root_model_ptr, uint64_t session_id) {
  if (comp_graph == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "Graph build comp_graph is null.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  ge_root_model_ptr = MakeShared<ge::GeRootModel>(comp_graph);
  if (ge_root_model_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }
  GeModelPtr ge_model_ptr = nullptr;
  bool is_dynamic_shape = false;
  // To be compatible with the old process, do not verify the return value temporarily.
  (void)AttrUtils::GetBool(comp_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  if (is_dynamic_shape) {
    GE_CHK_STATUS_RET(
        BuildForDynamicShapeGraph(comp_graph, subgraph_ptr_list, ge_root_model_ptr, ge_model_ptr, session_id),
        "Build for dynamic shape graph failed.");
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(BuildForKnownShapeGraph(comp_graph, subgraph_ptr_list, ge_model_ptr, session_id),
                    "Build for known shape graph failed.");
  ge_root_model_ptr->SetSubgraphInstanceNameToModel(comp_graph->GetName(), ge_model_ptr);
  return SUCCESS;
}

Status GraphBuilder::BuildForKnownShapeGraph(ComputeGraphPtr &comp_graph, std::vector<SubGraphInfoPtr> &subgraph_list,
                                             GeModelPtr &ge_model_ptr, uint64_t session_id) {
  if (ge::GetContext().GetHostExecFlag()) {
    GE_CHK_STATUS_RET(BuildForHostCpuGraph(comp_graph, ge_model_ptr, session_id), "Build for host-cpu graph failed.");
    return SUCCESS;
  }

  GELOGI("Begin to build known shape graph[%s].", comp_graph->GetName().c_str());
  Status ret = SecondPartition(comp_graph, subgraph_list);
  GE_CHK_STATUS_RET(ret, "Graph[%s] second partition Failed.", comp_graph->GetName().c_str());
  auto subgraph_map = graph_partitioner_.GetSubGraphMap();

  GE_TIMESTAMP_START(BuildSubgraph);
  ge::ModelBuilder builder(session_id, comp_graph, subgraph_map, stream_max_parallel_num_, hcom_parallel_, build_mode_);
  GE_DUMP(comp_graph, "BeforePreBuildModel");
  GE_TIMESTAMP_START(PreBuildModel);
  GE_CHK_STATUS_RET(builder.PreBuildModel(), "Graph[%s] builder PreBuildModel() return fail.",
                    comp_graph->GetName().c_str());
  GE_TIMESTAMP_END(PreBuildModel, "GraphBuilder::PreBuildModel");

  GE_DUMP(comp_graph, "AfterPreBuildModel");
  GE_TIMESTAMP_START(CalcOpParam);
  GE_CHK_STATUS_RET(CalcOpParam(comp_graph), "Graph[%s] builder CalcOpParam() return fail.",
                    comp_graph->GetName().c_str());
  GE_TIMESTAMP_END(CalcOpParam, "GraphBuilder::CalcOpParam");
  GE_DUMP(comp_graph, "AfterCalcOpParam");

  ModelPtr model_ptr = MakeShared<ge::Model>();
  if (model_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }
  GE_TIMESTAMP_START(BuildModelForGetTask);
  GE_CHK_STATUS_RET(builder.BuildModelForGetTask(*model_ptr), "Graph[%s] builder BuildModelForGetTask() return fail.",
                    comp_graph->GetName().c_str());
  GE_TIMESTAMP_END(BuildModelForGetTask, "GraphBuilder::BuildModelForGetTask");
  GE_DUMP(comp_graph, "AfterBuildModel");

  GE_TIMESTAMP_START(GetTaskInfo);
  ret = GetTaskInfo(builder, model_ptr, comp_graph, subgraph_map, session_id);
  GE_TIMESTAMP_END(GetTaskInfo, "GraphBuilder::GetTaskInfo");
  GE_DUMP(comp_graph, "AfterGetTask");
  if (ret != SUCCESS) {
    GELOGE(ret, "Graph[%s] builder GetTaskInfo() return fail.", comp_graph->GetName().c_str());
    return ret;
  }

  ge_model_ptr = MakeShared<ge::GeModel>();
  if (ge_model_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }
  GE_CHK_STATUS_RET(builder.SaveDataToModel(*model_ptr, *ge_model_ptr),
                    "Graph[%s] builder SaveDataToModel() return fail.", comp_graph->GetName().c_str());
  GELOGD("Success to build graph[%s] model.", comp_graph->GetName().c_str());
  GE_TIMESTAMP_END(BuildSubgraph, "GraphBuilder::Build");
  return SUCCESS;
}

Status GraphBuilder::BuildForUnknownShapeGraph(ComputeGraphPtr &comp_graph, GeModelPtr &ge_model_ptr,
                                               uint64_t session_id) {
  GELOGI("Begin to build unknown shape graph[%s].", comp_graph->GetName().c_str());
  GE_TIMESTAMP_START(CalcOpParam);
  GE_CHK_STATUS_RET(CalcOpParam(comp_graph), "Graph[%s] builder CalcOpParam() return fail.",
                    comp_graph->GetName().c_str());
  GE_TIMESTAMP_END(CalcOpParam, "GraphBuilder::CalcOpParam");
  GE_DUMP(comp_graph, "AfterCalcOpParam");
  Graph2SubGraphInfoList subgraph_map;
  ge::ModelBuilder builder(session_id, comp_graph, subgraph_map, stream_max_parallel_num_, hcom_parallel_, build_mode_);
  ModelPtr model_ptr = MakeShared<ge::Model>();
  if (model_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }
  GE_TIMESTAMP_START(BuildModelForGetDynShapeTask);
  GE_CHK_STATUS_RET(builder.BuildModelForGetDynShapeTask(*model_ptr),
                    "Graph[%s] builder BuildModelForGetDynShapeTask() return fail.", comp_graph->GetName().c_str());
  GE_TIMESTAMP_END(BuildModelForGetDynShapeTask, "GraphBuilder::BuildModelForGetDynShapeTask");
  GE_TIMESTAMP_START(GetTaskInfo);
  Status ret = GetTaskInfo(builder, model_ptr, comp_graph, subgraph_map, session_id);
  GE_TIMESTAMP_END(GetTaskInfo, "GraphBuilder::GetTaskInfo");

  GraphUtils::DumpGEGraph(comp_graph, "AfterGetTask");
  GraphUtils::DumpGEGraphToOnnx(*comp_graph, "AfterGetTask");
  if (ret != SUCCESS) {
    GELOGE(ret, "Graph[%s] builder GetTaskInfo() return fail.", comp_graph->GetName().c_str());
    return ret;
  }
  ge_model_ptr = MakeShared<ge::GeModel>();
  if (ge_model_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }
  GE_CHK_STATUS_RET(builder.SaveDataToModel(*model_ptr, *ge_model_ptr),
                    "Graph[%s] builder SaveDataToModel() return fail.", comp_graph->GetName().c_str());
  GELOGD("Success to build graph[%s] model.", comp_graph->GetName().c_str());
  return SUCCESS;
}

Status GraphBuilder::BuildForHostCpuGraph(ComputeGraphPtr &comp_graph, GeModelPtr &ge_model_ptr, uint64_t session_id) {
  return BuildForUnknownShapeGraph(comp_graph, ge_model_ptr, session_id);
}

static Status InsertMemcpyNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                               const std::vector<InDataAnchorPtr> &in_anchors, const std::string &name) {
  GE_CHECK_NOTNULL(out_anchor);
  NodePtr in_node = out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(in_node);
  OpDescBuilder op_desc_builder(name, MEMCPYADDRASYNC);
  OpDescPtr op_desc = op_desc_builder.AddInput("x", in_node->GetOpDesc()->GetOutputDesc(0))
                                     .AddOutput("y", in_node->GetOpDesc()->GetOutputDesc(0))
                                     .Build();
  (void)AttrUtils::SetBool(op_desc, ATTR_NO_NEED_CONSTANT_FOLDING, false);
  if (GraphUtils::InsertNodeAfter(out_anchor, in_anchors, graph->AddNode(op_desc)) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Insert IDENTITY node %s after %s failed.", name.c_str(), in_node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

static Status GenerateTaskForConstant(const std::shared_ptr<ComputeGraph> &graph) {
  for (auto &node : graph->GetDirectNode()) {
    // CONSTANT not generate task, so insert IDENTITY between CONSTANT and NETOUTPUT
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    auto op_type = op_desc->GetType();
    if (op_type == NETOUTPUT) {
      for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
        const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
        GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
        NodePtr in_node = peer_out_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(in_node);

        std::string in_node_op_type = in_node->GetType();
        if (in_node_op_type == CONSTANT) {
          GELOGD("Insert MemcpyAsync node between %s and %s.", in_node->GetName().c_str(), node->GetName().c_str());
          std::string name = node->GetName() + "_input_" + std::to_string(in_data_anchor->GetIdx()) + "_Memcpy";
          if (InsertMemcpyNode(graph, peer_out_anchor, {in_data_anchor}, name) != SUCCESS) {
            GELOGE(FAILED, "Insert memcpy between %s and %s failed.",
                   in_node->GetName().c_str(), node->GetName().c_str());
            return FAILED;
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status GraphBuilder::BuildForDynamicShapeGraph(ComputeGraphPtr &comp_graph,
                                               std::vector<SubGraphInfoPtr> &subgraph_ptr_list,
                                               GeRootModelPtr &ge_root_model_ptr, GeModelPtr &ge_model_ptr,
                                               uint64_t session_id) {
  GELOGI("Start to build BuildForDynamicShape for dynamic shape.");
  // Update Root Graph Data size
  for (auto &node : comp_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    op_desc->SetStreamId(kInvalidStream);
    if (node->GetType() == DATA) {
      GE_CHK_STATUS_RET(CalcDynShapeRootGraphDataSize(op_desc), "Calc dynamic shape root graph data[%s] size failed.",
                        op_desc->GetName().c_str());
    }
  }
  //
  for (auto &sub_graph : comp_graph->GetAllSubgraphs()) {
    // exclude functional subgraph in known subgraph
    if (sub_graph->GetParentGraph() != comp_graph && !sub_graph->GetParentGraph()->GetGraphUnknownFlag()) {
      continue;
    }

    GE_CHK_STATUS_RET(GenerateTaskForConstant(sub_graph), "Generate task For constant node in subgraph failed.");

    if (sub_graph->GetGraphUnknownFlag()) {
      // unknown shape build flow
      GE_CHK_STATUS_RET(BuildForUnknownShapeGraph(sub_graph, ge_model_ptr, session_id),
                        "Build for unknown shape graph failed.");
    } else {
      // reset functional subgraph parent graph as known subgraph
      for (const auto &node : sub_graph->GetDirectNode()) {
        for (const auto &sub_graph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
          auto sub_sub_graph = comp_graph->GetSubgraph(sub_graph_name);
          GE_CHK_STATUS_RET(sub_graph->AddSubgraph(sub_sub_graph), "Failed add subgraph to known graph.");
        }
      }
      // known shape build flow
      GE_CHK_STATUS_RET(BuildForKnownShapeGraph(sub_graph, subgraph_ptr_list, ge_model_ptr, session_id),
                        "Build for known shape graph failed.");
    }
    ge_root_model_ptr->SetSubgraphInstanceNameToModel(sub_graph->GetName(), ge_model_ptr);
  }

  return SUCCESS;
}

Status GraphBuilder::GetTaskInfo(const ge::ModelBuilder &builder, const ModelPtr &model_ptr,
                                 ComputeGraphPtr &comp_graph, Graph2SubGraphInfoList &subgraph_map,
                                 uint64_t session_id) {
  GE_CHECK_NOTNULL(model_ptr);
  GE_CHECK_NOTNULL(comp_graph);

  int64_t memory_size = 0;
  if (!AttrUtils::GetInt(model_ptr, ATTR_MODEL_MEMORY_SIZE, memory_size)) {
    GELOGE(INTERNAL_ERROR, "Get memory size fail.");
    return INTERNAL_ERROR;
  }
  int64_t p2p_memory_size = 0;
  if (!AttrUtils::GetInt(model_ptr, ATTR_MODEL_P2P_MEMORY_SIZE, p2p_memory_size)) {
    GELOGE(INTERNAL_ERROR, "Get p2p memory size fail.");
    return INTERNAL_ERROR;
  }
  int64_t weight_size = 0;
  if (!AttrUtils::GetInt(model_ptr, ATTR_MODEL_WEIGHT_SIZE, weight_size)) {
    GELOGE(INTERNAL_ERROR, "Get weight memory size fail.");
    return INTERNAL_ERROR;
  }

  auto var_manager = VarManager::Instance(session_id);
  // since var_mem_logic_base_ = graph_mem_max_size_ + kGraphMemoryBuffer in graph_var_manager.cc,
  // get_mem_base should not bigger than kGraphMemoryBuffer
  auto *get_mem_base = reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(kGraphMemoryBuffer>>1));
  uint8_t *get_weight_mem_base = get_mem_base;
  if (weight_size > 0) {
    get_weight_mem_base = get_mem_base + memory_size + p2p_memory_size;
  }
  std::map<int64_t, uint8_t *> mem_type_to_data_mem_base;
  mem_type_to_data_mem_base[RT_MEMORY_HBM] = get_mem_base;
  if (p2p_memory_size == 0) {
    mem_type_to_data_mem_base[RT_MEMORY_P2P_DDR] = nullptr;
  } else {
    mem_type_to_data_mem_base[RT_MEMORY_P2P_DDR] = get_mem_base + memory_size;
  }
  std::map<int64_t, uint64_t> mem_type_to_data_mem_size;
  mem_type_to_data_mem_size[RT_MEMORY_HBM] = memory_size;
  mem_type_to_data_mem_size[RT_MEMORY_P2P_DDR] = p2p_memory_size;
  RunContextUtil run_context;
  Status ret = run_context.InitMemInfo(get_mem_base, memory_size, mem_type_to_data_mem_base, mem_type_to_data_mem_size,
                                       get_weight_mem_base, weight_size);
  if (ret != SUCCESS) {
    GELOGE(ret, "task_generator init mem info fail.");
    return ret;
  }
  auto weight_buffer = builder.GetWeightBuffer();
  ret = run_context.CreateRunContext(*model_ptr, comp_graph, weight_buffer, session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "runContext create run context fail.");
    return ret;
  }

  StreamGraphOptimizer stream_optimizer;
  ret = stream_optimizer.OptimizeStreamedSubGraph(comp_graph, subgraph_map, run_context.GetRunContext());
  if (ret != SUCCESS) {
    GELOGE(ret, "Optimize streamed subGraph fail.");
    return ret;
  }
  GE_DUMP(comp_graph, "AfterOptimizeStreamedSubGraph");
  auto *get_var_mem_base = reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(var_manager->GetVarMemLogicBase()));
  uint64_t var_size = (var_manager->GetVarMemSize(RT_MEMORY_HBM) > 0) ? var_manager->GetVarMemMaxSize() : 0;
  TaskGenerator task_generator(get_var_mem_base, var_size);
  ret = task_generator.GetTaskInfo(*model_ptr, comp_graph, session_id, run_context.GetRunContext());

  return ret;
}

Status GraphBuilder::SetInputSize(const ge::NodePtr &node_ptr) {
  // Set the size of input_desc to 'src_node.output_desc.size'
  if (node_ptr->GetType() == DATA) {
    bool is_unknown_shape = false;
    GE_CHK_STATUS_RET(ge::NodeUtils::GetNodeUnknownShapeStatus(*node_ptr, is_unknown_shape),
                      "Get data node[%s] shape status failed!", node_ptr->GetName().c_str());
    if (is_unknown_shape) {
      GELOGD("data node: %s is unknown shape, do not set input size!", node_ptr->GetName().c_str());
      return SUCCESS;
    }
    if (UpdateDataInputSize(node_ptr) != SUCCESS) {
      GELOGE(FAILED, "Update data input size failed.");
      return FAILED;
    }
  }

  for (const auto &in_data_anchor : node_ptr->GetAllInDataAnchors()) {
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    const auto &src_op = src_node->GetOpDesc();
    GE_IF_BOOL_EXEC(src_op == nullptr, continue);
    auto node_op_desc = node_ptr->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    // Set the input_desc of dst_node to 'src_node.output_desc'
    auto output_desc = src_op->GetOutputDescPtr(peer_out_anchor->GetIdx());
    int64_t size = 0;
    GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(*output_desc, size) != SUCCESS, GELOGI("Get size failed!"));
    GELOGD("src node %s output desc, dim_size: %zu, mem_size: %ld, format: %s, type: %s.", src_node->GetName().c_str(),
           output_desc->GetShape().GetDimNum(), size, TypeUtils::FormatToSerialString(output_desc->GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(output_desc->GetDataType()).c_str());
    for (size_t i = 0; i < output_desc->GetShape().GetDimNum(); ++i) {
      GELOGD("dims[%zu]: %ld", i, output_desc->GetShape().GetDim(i));
    }

    auto input_desc = node_op_desc->MutableInputDesc(in_data_anchor->GetIdx());
    GE_CHECK_NOTNULL(input_desc);
    (void) ge::TensorUtils::SetSize(*input_desc, size);
    GELOGD("%s input desc, dim_size: %zu, mem_size: %ld, format: %s, type: %s.", node_ptr->GetName().c_str(),
           input_desc->GetShape().GetDimNum(), size, TypeUtils::FormatToSerialString(input_desc->GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(input_desc->GetDataType()).c_str());
    // inherit some attr
    int64_t tensor_size_attr;
    if (AttrUtils::GetInt(output_desc, ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size_attr) && (tensor_size_attr > 0)) {
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(*input_desc, ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size_attr),
                      GELOGW("Set size attr failed!"); continue);
      GELOGD("node[%s] [%d]th output has sepcial size[%ld], and update to node[%s] [%d]th input",
             src_op->GetName().c_str(), peer_out_anchor->GetIdx(), tensor_size_attr,
             node_op_desc->GetName().c_str(), in_data_anchor->GetIdx());
    }
  }

  return SUCCESS;
}

Status GraphBuilder::UpdateDataInputSize(const ge::NodePtr &node_ptr) {
  const auto &op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Op desc is nullptr.");
    return FAILED;
  }
  // data op only has one output anchor
  ge::GeTensorDesc output_desc = op_desc->GetOutputDesc(0);
  int64_t output_size = 0;
  if (ge::TensorUtils::GetSize(output_desc, output_size) != SUCCESS) {
    GELOGW("Get size failed!");
  }

  if (output_size > 0) {
    GELOGI("No need to update data input size.");
    return SUCCESS;
  } else {
    int64_t real_dim_size = 0;
    ge::graphStatus graph_status = TensorUtils::GetTensorSizeInBytes(output_desc, real_dim_size);
    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Get tensor size in bytes failed.");
      return FAILED;
    }
    // data op only has one input anchor
    ge::GeTensorDesc input_desc = op_desc->GetInputDesc(0);
    ge::TensorUtils::SetSize(input_desc, real_dim_size);
    if (op_desc->UpdateInputDesc(0, input_desc) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Update input desc size failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GraphBuilder::CalcDynShapeRootGraphDataSize(const ge::OpDescPtr &op_desc) {
  GELOGI("Begin to calc dynamic shape graph data[%s] size.", op_desc->GetName().c_str());
  // data op only has one output anchor
  ge::GeTensorDesc output_desc = op_desc->GetOutputDesc(0);
  if (output_desc.MutableShape().IsUnknownShape()) {
    GELOGI("No need to update dynamic shape graph data output size for unknown shape data.");
    return SUCCESS;
  }

  int64_t output_size = 0;
  if (ge::TensorUtils::GetSize(output_desc, output_size) != SUCCESS) {
    GELOGW("Get size failed!");
  }

  if (output_size > 0) {
    GELOGI("No need to update dynamic shape graph data output size[%ld].", output_size);
    return SUCCESS;
  } else {
    int64_t real_dim_size = 0;
    ge::graphStatus graph_status = TensorUtils::GetTensorSizeInBytes(output_desc, real_dim_size);
    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Get tensor size in bytes failed.");
      return FAILED;
    }

    ge::TensorUtils::SetSize(output_desc, real_dim_size);
    GELOGI("Update dynamic shape graph data output size to [%ld].", real_dim_size);
    if (op_desc->UpdateOutputDesc(0, output_desc) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Update dynamic shape graph data output desc size failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GraphBuilder::SecondPartition(ge::ComputeGraphPtr &comp_graph, vector<ge::SubGraphInfoPtr> &subgraph_ptr_list) {
  GE_TIMESTAMP_START(GraphPartition2);
  auto ret = graph_partitioner_.Partition(comp_graph, GraphPartitioner::kSecondPartitioning);
  if (ret != SUCCESS) {
    GELOGE(ret, "Graph partition Failed");
    return ret;
  }
  GE_CHK_STATUS_RET(ret, "Graph partition Failed.");
  auto graph_2_subgraphlist = graph_partitioner_.GetSubGraphMap();
  if (graph_2_subgraphlist.find(comp_graph) != graph_2_subgraphlist.end()) {
    subgraph_ptr_list = graph_2_subgraphlist[comp_graph];
  } else {
    GELOGE(FAILED, "Find subgraph failed.");
    return FAILED;
  }
  GE_TIMESTAMP_END(GraphPartition2, "GraphPartitioner::Partition2");
  return ret;
}

Status GraphBuilder::AddOutputMemTypeForNode(const NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  uint32_t mem_type;
  if (!AttrUtils::GetInt(op_desc, ATTR_INPUT_MEMORY_TYPE, mem_type)) {
    return SUCCESS;
  }
  GELOGD("[%s] has attr input_memory_type %ld", op_desc->GetName().c_str(), mem_type);
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    bool valid_flag = false;
    auto src_node = peer_out_anchor->GetOwnerNode();
    auto src_out_anchor = peer_out_anchor;
    while (true) {
      const auto &src_desc = src_node->GetOpDesc();
      GE_IF_BOOL_EXEC(src_desc == nullptr, continue);
      GELOGD("[%s:%u] set attr output_memory_type %ld", src_desc->GetName().c_str(), src_out_anchor->GetIdx(),
             mem_type);
      if (!AttrUtils::SetInt(src_desc->MutableOutputDesc(src_out_anchor->GetIdx()), ATTR_OUTPUT_MEMORY_TYPE,
                             mem_type)) {
        GELOGE(INTERNAL_ERROR, "Set out_memory_type attr for [%s:%d] failed.", src_desc->GetName().c_str(),
               src_out_anchor->GetIdx());
        return INTERNAL_ERROR;
      }
      switch (TransferNodeType(src_node)) {
        case kSubgraphNode:
          GE_CHK_STATUS_RET(HandleSubgraphNode(src_node, src_out_anchor), "Handle subgraph node %s failed",
                            src_node->GetName().c_str());
          break;
        case kSubgraphData:
          GE_CHK_STATUS_RET(HandleSubgraphDataNode(src_node, src_out_anchor), "Handle Data node %s in subgraph failed",
                            src_node->GetName().c_str());
          break;
        case kOthers:
        default:
          valid_flag = true;
          break;
      }
      if (valid_flag) {
        break;
      }
    }
  }

  return SUCCESS;
}
}  // namespace ge
