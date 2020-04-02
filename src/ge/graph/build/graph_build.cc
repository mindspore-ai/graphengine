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

#include "graph/build/graph_build.h"
#include "common/ge/ge_util.h"
#include "common/helper/model_helper.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/build/optimize_stream_graph.h"
#include "graph/build/run_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "model/ge_model.h"

using domi::BuildMode;

namespace {
const int32_t kInvalidPerfLevel = -1;
}  // namespace
namespace ge {
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
  GELOGI("Begin to calculate op running param.");
  GE_CHECK_NOTNULL(graph);
  auto instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "GraphBuilder: GE is not initialized");
    return GE_CLI_GE_NOT_INITIALIZED;
  }
  for (const auto &node_ptr : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr->GetOpDesc());
    std::string kernel_lib_name = node_ptr->GetOpDesc()->GetOpKernelLibName();
    if (kernel_lib_name.empty()) {
      // reset op kernel lib
      (void)instance_ptr->DNNEngineManagerObj().GetDNNEngineName(node_ptr->GetOpDesc());
      kernel_lib_name = node_ptr->GetOpDesc()->GetOpKernelLibName();
      if (kernel_lib_name.empty()) {
        GELOGE(INTERNAL_ERROR, "Get node:%s(%s) kernel lib failed.", node_ptr->GetName().c_str(),
               node_ptr->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }

    OpsKernelInfoStorePtr kernel_info = instance_ptr->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
    if (kernel_info != nullptr) {
      auto ret = SetInputSize(node_ptr);
      if (ret != SUCCESS) {
        GELOGE(ret, "Set node inputDesc size failed, node name is %s", node_ptr->GetName().c_str());
        return ret;
      }
      ret = kernel_info->CalcOpRunningParam(*node_ptr);
      if (ret != SUCCESS) {
        GELOGE(ret, "Calculate op running param failed, node name is %s", node_ptr->GetName().c_str());
        return ret;
      }
    } else {
      GELOGE(GE_GRAPH_PARAM_NULLPTR, "Get op %s ops kernel info store failed", node_ptr->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }
  GELOGI("Success to calculate op running param.");
  return SUCCESS;
}

Status GraphBuilder::Build(ComputeGraphPtr &comp_graph, std::vector<SubGraphInfoPtr> &subgraph_ptr_list,
                           GeModelPtr &ge_model_ptr, uint64_t session_id) {
  GELOGI("Start to build model.");
  if (comp_graph == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "Graph build comp_graph is null.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  Status ret = SecondPartition(comp_graph, subgraph_ptr_list);
  GE_CHK_STATUS_RET(ret, "Graph second partition Failed.");

  GE_TIMESTAMP_START(BuildSubgraph);
  ge::ModelBuilder builder(comp_graph, subgraph_ptr_list, stream_max_parallel_num_, hcom_parallel_, build_mode_);

  GELOGI("[Build] invoke the other opskernel to generate task.");

  GraphUtils::DumpGEGraph(comp_graph, "BeforePreBuildModel");
  GraphUtils::DumpGEGraphToOnnx(*comp_graph, "BeforePreBuildModel");

  GE_TIMESTAMP_START(PreBuildModel);
  GE_CHK_STATUS_RET(builder.PreBuildModel(), "Builder PreBuildModel() return fail.");
  GE_TIMESTAMP_END(PreBuildModel, "GraphBuilder::PreBuildModel");

  GraphUtils::DumpGEGraph(comp_graph, "AfterPrebuildmodel");
  GraphUtils::DumpGEGraphToOnnx(*comp_graph, "AfterPrebuildmodel");

  GE_TIMESTAMP_START(CalcOpParam);
  GE_CHK_STATUS_RET(CalcOpParam(comp_graph), "Builder CalcOpParam() return fail.");
  GE_TIMESTAMP_END(CalcOpParam, "GraphBuilder::CalcOpParam");
  GraphUtils::DumpGEGraph(comp_graph, "AfterCalcOpParam");
  GraphUtils::DumpGEGraphToOnnx(*comp_graph, "AfterCalcOpParam");

  ModelPtr model_ptr = MakeShared<ge::Model>();
  if (model_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }
  GE_TIMESTAMP_START(BuildModelForGetTask);
  GE_CHK_STATUS_RET(builder.BuildModelForGetTask(*model_ptr), "Builder BuildModelForGetTask() return fail.");
  GE_TIMESTAMP_END(BuildModelForGetTask, "GraphBuilder::BuildModelForGetTask");

  GraphUtils::DumpGEGraph(comp_graph, "AfterBuildModel");
  GraphUtils::DumpGEGraphToOnnx(*comp_graph, "AfterBuildModel");

  GE_TIMESTAMP_START(GetTaskInfo);
  ret = GetTaskInfo(builder, model_ptr, comp_graph, subgraph_ptr_list, session_id);
  GE_TIMESTAMP_END(GetTaskInfo, "GraphBuilder::GetTaskInfo");

  GraphUtils::DumpGEGraph(comp_graph, "AfterGetTask");
  GraphUtils::DumpGEGraphToOnnx(*comp_graph, "AfterGetTask");
  if (ret != SUCCESS) {
    GELOGE(ret, "Builder GetTaskInfo() return fail.");
    return ret;
  }
  ge_model_ptr = MakeShared<ge::GeModel>();
  if (ge_model_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }
  GE_CHK_STATUS_RET(builder.SaveDataToModel(*model_ptr, *ge_model_ptr), "model builder SaveDataToModel() return fail.");
  GELOGI("Success to build model.");
  GE_TIMESTAMP_END(BuildSubgraph, "GraphBuilder::Build");
  return SUCCESS;
}

Status GraphBuilder::GetTaskInfo(const ge::ModelBuilder &builder, const ModelPtr &model_ptr,
                                 ComputeGraphPtr &comp_graph, std::vector<SubGraphInfoPtr> &subgraph_ptr_list,
                                 uint64_t session_id) {
  GE_CHECK_NOTNULL(model_ptr);
  GE_CHECK_NOTNULL(comp_graph);

  int64_t memory_size = 0;
  if (!AttrUtils::GetInt(model_ptr, ATTR_MODEL_MEMORY_SIZE, memory_size)) {
    GELOGE(INTERNAL_ERROR, "Get memory size fail.");
    return INTERNAL_ERROR;
  }
  int64_t weight_size = 0;
  if (!AttrUtils::GetInt(model_ptr, ATTR_MODEL_WEIGHT_SIZE, weight_size)) {
    GELOGE(INTERNAL_ERROR, "Get weight memory size fail.");
    return INTERNAL_ERROR;
  }
  auto *get_mem_base = reinterpret_cast<uint8_t *>(ge::VarManager::Instance(0)->GetVarMemMaxSize());
  uint8_t *get_weight_mem_base = get_mem_base;
  if (weight_size > 0) {
    get_weight_mem_base = get_mem_base + memory_size;
  }

  RunContextUtil run_context;
  Status ret = run_context.InitMemInfo(get_mem_base, memory_size, get_weight_mem_base, weight_size);
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

  OptimizeStreamGraph optimize_stream;
  ret = optimize_stream.OptimizeStreamedSubGraph(comp_graph, subgraph_ptr_list, run_context.GetRunContext());
  if (ret != SUCCESS) {
    GELOGE(ret, "Optimize streamed subGraph fail.");
    return ret;
  }

  GraphUtils::DumpGEGraph(comp_graph, "AfterOptimizeStreamedSubGraph");
  GraphUtils::DumpGEGraphToOnnx(*comp_graph, "AfterOptimizeStreamedSubGraph");

  auto *get_var_mem_base = reinterpret_cast<uint8_t *>(ge::VarManager::Instance(0)->GetVarMemLogicBase());
  uint64_t var_size = (ge::VarManager::Instance(session_id)->GetVarMemSize(RT_MEMORY_HBM) > 0)
                        ? ge::VarManager::Instance(0)->GetVarMemMaxSize()
                        : 0;
  TaskGenerator task_generator(get_var_mem_base, var_size);
  ret = task_generator.GetTaskInfo(*model_ptr, comp_graph, session_id, run_context.GetRunContext());

  return ret;
}

Status GraphBuilder::SetInputSize(const ge::NodePtr &node_ptr) {
  // set input_desc.size = src_node.output_desc.size
  for (const auto &in_data_anchor : node_ptr->GetAllInDataAnchors()) {
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    const auto &src_op = src_node->GetOpDesc();
    GE_IF_BOOL_EXEC(src_op == nullptr, continue);
    auto node_op_desc = node_ptr->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    // set dst_node.input_desc = src_node.output_desc
    ge::GeTensorDesc desc_temp(src_op->GetOutputDesc(peer_out_anchor->GetIdx()));

    uint32_t size = 0;
    GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(desc_temp, size) != SUCCESS, GELOGI("Get size failed!"));
    GELOGD("src node %s output desc, dim_size: %zu, mem_size: %u, format: %s, type: %s.", src_node->GetName().c_str(),
           desc_temp.GetShape().GetDimNum(), size, TypeUtils::FormatToSerialString(desc_temp.GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(desc_temp.GetDataType()).c_str());
    for (size_t i = 0; i < desc_temp.GetShape().GetDimNum(); ++i) {
      GELOGD("dims[%zu]: %ld", i, desc_temp.GetShape().GetDim(i));
    }

    auto input_desc = node_op_desc->GetInputDescPtr(in_data_anchor->GetIdx());
    GE_CHECK_NOTNULL(input_desc);
    ge::TensorUtils::SetSize(const_cast<GeTensorDesc &>(*input_desc), size);
    GE_CHK_STATUS_RET(node_op_desc->UpdateInputDesc(in_data_anchor->GetIdx(), *input_desc));
    GELOGD("%s input desc, dim_size: %zu, mem_size: %u, format: %s, type: %s.", node_ptr->GetName().c_str(),
           input_desc->GetShape().GetDimNum(), size, TypeUtils::FormatToSerialString(input_desc->GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(input_desc->GetDataType()).c_str());
  }

  return SUCCESS;
}

Status GraphBuilder::SecondPartition(ge::ComputeGraphPtr &comp_graph,
                                     std::vector<ge::SubGraphInfoPtr> &subgraph_ptr_list) {
  GELOGI("[SecondPartition] second partition.");
  subgraph_ptr_list.clear();
  GE_TIMESTAMP_START(GraphPartition2);
  Status ret = graph_partitioner_.Partition(comp_graph, subgraph_ptr_list, GraphPartitioner::kSecondPartitioning);
  GE_CHK_STATUS_RET(ret, "Graph partition Failed.");
  GE_TIMESTAMP_END(GraphPartition2, "GraphPartitioner::Partition2");
  return ret;
}
}  // namespace ge
