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

#include "utils/graph_utils.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "./ge_context.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "proto/ge_ir.pb.h"
#include "utils/attr_utils.h"
#include "utils/ge_ir_utils.h"
#include "utils/node_utils.h"

using google::protobuf::io::FileOutputStream;

namespace ge {
enum DumpGraphLevel {
  kDumpLevel1 = 1,
  kDumpLevel2 = 2,
  kDumpLevel3 = 3,
  kDumpLevelOther,
};

namespace {
const int32_t kBaseOfIntegerValue = 10;
#ifdef FMK_SUPPORT_DUMP
const char *const kDumpGeGraph = "DUMP_GE_GRAPH";
#endif
const char *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
const char *const kDumpStrBuild = "Build";
const char *const kDumpStrPartition = "partition";
const char *const kDumpStrOptimizeSubgraph = "OptimizeSubGraph";
const char *const kDumpStrAicpu = "Aicpu";
};  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,
                                                                               const InDataAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Add edge Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const AnchorPtr &src,
                                                                               const AnchorPtr &dst) {
  OutDataAnchorPtr src_data = Anchor::DynamicAnchorCast<OutDataAnchor>(src);
  InDataAnchorPtr dst_data = Anchor::DynamicAnchorCast<InDataAnchor>(dst);
  OutControlAnchorPtr src_control = Anchor::DynamicAnchorCast<OutControlAnchor>(src);
  InControlAnchorPtr dst_control = Anchor::DynamicAnchorCast<InControlAnchor>(dst);
  if ((src_data != nullptr) && (dst_data != nullptr) && (src_data->LinkTo(dst_data) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  if ((src_data != nullptr) && (dst_control != nullptr) && (src_data->LinkTo(dst_control) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  if ((src_control != nullptr) && (dst_control != nullptr) && (src_control->LinkTo(dst_control) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  if ((src_control != nullptr) && (dst_data != nullptr) && (src_control->LinkTo(dst_data) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Add edge Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,
                                                                               const Format &src_format,
                                                                               const InDataAnchorPtr &dst,
                                                                               const Format &dst_format) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    (void)AnchorUtils::SetFormat(src, src_format);
    (void)AnchorUtils::SetFormat(dst, dst_format);
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Add edge Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutControlAnchorPtr &src,
                                                                               const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Add edge Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,
                                                                               const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Add edge Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutDataAnchorPtr &src,
                                                                                  const InDataAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Remove edge Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const AnchorPtr &src,
                                                                                  const AnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Remove edge Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutControlAnchorPtr &src,
                                                                                  const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Remove edge Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutDataAnchorPtr &src,
                                                                                  const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Remove edge Failed.");
  return GRAPH_FAILED;
}

graphStatus GraphUtils::ReplaceEdgeDst(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                       const InDataAnchorPtr &new_dst) {
  if (RemoveEdge(src, dst) == GRAPH_SUCCESS && AddEdge(src, new_dst) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Replace edge dst Failed.");
  return GRAPH_FAILED;
}

graphStatus GraphUtils::ReplaceEdgeDst(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst,
                                       const InControlAnchorPtr &new_dst) {
  if (RemoveEdge(src, dst) == GRAPH_SUCCESS && AddEdge(src, new_dst) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Replace edge dst Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::InsertNodeBetweenDataAnchors(
    const OutDataAnchorPtr &src, const InDataAnchorPtr &dst, const NodePtr &new_node) {
  GE_CHECK_NOTNULL(src);
  GE_CHECK_NOTNULL(dst);
  GE_CHECK_NOTNULL(new_node);

  InDataAnchorPtr node_in_anchor = new_node->GetInDataAnchor(0);
  GE_CHK_BOOL_RET_STATUS(node_in_anchor != nullptr, GRAPH_FAILED, "this node has not inDataAnchor");
  OutDataAnchorPtr node_out_anchor = new_node->GetOutDataAnchor(0);
  GE_CHK_BOOL_RET_STATUS(node_out_anchor != nullptr, GRAPH_FAILED, "this node has not outDataAnchor");
  GE_CHK_STATUS_RET(src->ReplacePeer(dst, node_in_anchor, node_out_anchor), "ReplacePeer Failed");
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RemoveNodeWithoutRelink(const ComputeGraphPtr &compute_graph, const NodePtr &node) {
  GE_CHECK_NOTNULL(compute_graph);
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  // If the node save as input node, delete it
  (void)compute_graph->RemoveInputNode(node);

  // If the node save as output node, delete it
  (void)compute_graph->RemoveOutputNode(node);

  auto iter = find(compute_graph->nodes_.begin(), compute_graph->nodes_.end(), node);
  if (iter != compute_graph->nodes_.end()) {
    compute_graph->nodes_.erase(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

/// Add two edges to the new node, respectively connecting the SRC and DST
/// associated with the original edge
/// A ---> B transferred to  A ---> N ---> B
graphStatus InsertTransNode(ComputeGraph &compute_graph, const InDataAnchorPtr &in_data_anchor,
                            const std::vector<OpDescPtr> &vec_op_desc) {
  for (auto &op_desc : vec_op_desc) {
    GE_CHECK_NOTNULL(op_desc);

    auto ret = op_desc->AddInputDesc(GeTensorDesc());
    GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS, return GRAPH_FAILED, "Add input desc failed");
    ret = op_desc->AddOutputDesc(GeTensorDesc());
    GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS, return GRAPH_FAILED, "Add input desc failed");
    auto node_to_insert = compute_graph.AddNode(op_desc);

    GE_CHECK_NOTNULL(node_to_insert);
    GE_CHECK_NOTNULL(in_data_anchor->GetPeerOutAnchor());

    auto src = in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
    if (!src) {
      GELOGE(GRAPH_FAILED, "src nullptr error.");
      return GRAPH_FAILED;
    }

    auto src_out_index = in_data_anchor->GetPeerOutAnchor()->GetIdx();

    auto dst = in_data_anchor->GetOwnerNode();
    if (!dst) {
      GELOGE(GRAPH_FAILED, "dst nullptr error.");
      return GRAPH_FAILED;
    }

    auto dst_in_index = in_data_anchor->GetIdx();

    auto in_data_anchor_src_format = AnchorUtils::GetFormat(in_data_anchor->GetPeerOutAnchor());
    auto in_data_anchor_dst_format = AnchorUtils::GetFormat(in_data_anchor);

    GE_CHECK_NOTNULL(src->GetOutDataAnchor(src_out_index));
    GE_CHECK_NOTNULL(dst->GetInDataAnchor(dst_in_index));

    ret = GraphUtils::RemoveEdge(src->GetOutDataAnchor(src_out_index), dst->GetInDataAnchor(dst_in_index));
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Remove edge failed");
      return GRAPH_FAILED;
    }

    GE_CHECK_NOTNULL(node_to_insert->GetInDataAnchor(0));
    GE_CHECK_NOTNULL(node_to_insert->GetOutDataAnchor(0));

    ret = GraphUtils::AddEdge(src->GetOutDataAnchor(src_out_index), node_to_insert->GetInDataAnchor(0));
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Add edge failed");
      return ret;
    }
    ret = GraphUtils::AddEdge(node_to_insert->GetOutDataAnchor(0), dst->GetInDataAnchor(dst_in_index));
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Add edge failed");
      return ret;
    }

    if (op_desc->HasAttr("input_format")) {
      int64_t input_format = 0;
      int64_t output_format = 0;
      if (!AttrUtils::GetInt(op_desc, "input_format", input_format)) {
        GELOGW("get attr input_format failed");
      }
      if (!AttrUtils::GetInt(op_desc, "output_format", output_format)) {
        GELOGW("get attr output_format failed");
      }

      GE_CHECK_NOTNULL(node_to_insert->GetInDataAnchor(0)->GetPeerOutAnchor());
      GE_CHK_BOOL_RET_STATUS(node_to_insert->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty(), GRAPH_FAILED,
                             "Vistor<InDataAnchorPtr> is empty");
      GE_CHECK_NOTNULL(node_to_insert->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0));

      (void)AnchorUtils::SetFormat(node_to_insert->GetInDataAnchor(0)->GetPeerOutAnchor(), in_data_anchor_src_format);
      (void)AnchorUtils::SetFormat(node_to_insert->GetInDataAnchor(0), (Format)input_format);
      (void)AnchorUtils::SetFormat(node_to_insert->GetOutDataAnchor(0), (Format)output_format);
      (void)AnchorUtils::SetFormat(node_to_insert->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0),
                                   in_data_anchor_dst_format);
    }
    std::vector<ge::NodePtr> original_nodes;
    GraphUtils::RecordOriginalNames(original_nodes, node_to_insert);
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::InsertTransNode(
    ComputeGraphPtr compute_graph, const InDataAnchorPtr &in_data_anchor, const std::vector<OpDescPtr> &vec_op_desc) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(in_data_anchor);
  graphStatus ret =
      ge::InsertTransNode(*compute_graph, in_data_anchor, vec_op_desc) == GRAPH_SUCCESS ? GRAPH_SUCCESS : GRAPH_FAILED;
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveJustNode(ComputeGraph &compute_graph,
                                                                                      const NodePtr &node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "The node ptr should be not null.");
    return GRAPH_FAILED;
  }
  auto iter = find(compute_graph.nodes_.begin(), compute_graph.nodes_.end(), node);
  if (iter != compute_graph.nodes_.end()) {
    compute_graph.nodes_.erase(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveJustNode(ComputeGraphPtr compute_graph,
                                                                                      const NodePtr &node) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(node);
  graphStatus ret = (RemoveJustNode(*compute_graph, node) == GRAPH_SUCCESS ? GRAPH_SUCCESS : GRAPH_FAILED);
  return ret;
}

void GraphUtils::RecordOriginalNames(std::vector<ge::NodePtr> original_nodes, const ge::NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, return, "node is null.");
  std::vector<std::string> original_names;
  for (const auto &node_tmp : original_nodes) {
    std::vector<std::string> names_tmp;
    ge::OpDescPtr opdesc_tmp = node_tmp->GetOpDesc();
    (void)ge::AttrUtils::GetListStr(opdesc_tmp, "original_op_names", names_tmp);
    if (names_tmp.size() != 0) {
      original_names.insert(original_names.end(), names_tmp.begin(), names_tmp.end());
    } else {
      original_names.push_back(opdesc_tmp->GetName());
    }
  }
  if (original_names.size() == 0) {
    std::string tmp;
    original_names.push_back(tmp);
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(node->GetOpDesc(), "original_op_names", original_names), return,
                   "Set original_op_names fail.");
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::RecordOriginalNames(std::vector<std::string> names_tmp,
                                                                                    const ge::NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, return, "node is null.");
  std::vector<std::string> original_names;
  if (names_tmp.size() != 0) {
    original_names.insert(original_names.end(), names_tmp.begin(), names_tmp.end());
  } else {
    std::string tmp;
    original_names.push_back(tmp);
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(node->GetOpDesc(), "original_op_names", original_names), return,
                   "Set original_op_names fail.");
}

// Check global_step Node has IsVariable and Read.
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::CheckGlobalStepNode(const ge::NodePtr &node) {
  GE_CHK_BOOL_EXEC(
      node != nullptr, { return false; }, "node is null.");
  bool has_variable = false;
  bool has_cond_read = false;
  for (const auto &out : node->GetOutDataNodes()) {
    if ((out->GetType() == "VarIsInitializedOp") && (out->GetName() == "global_step/IsVariableInitialized")) {
      has_variable = true;
    } else if ((out->GetType() == "FrameworkOp") && (out->GetName() == "global_step/cond/read/Switch")) {
      has_cond_read = true;
    }
  }
  return (has_variable && has_cond_read);
}

// Check origin ComputeGraph is TrainGraph.
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::CheckIsTrainGraph(
    const ge::ComputeGraphPtr &compute_graph) {
  GE_CHK_BOOL_EXEC(
      compute_graph != nullptr, { return false; }, "compute_graph is nullptr");

  bool is_iterator_v2 = false;
  bool is_train_graph = false;
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "ApplyMomentum") {
      return true;
    }
    // Check global_step has IsVariable and Read.
    if ((node->GetType() == "Variable") && (node->GetName() == "global_step")) {
      is_train_graph = CheckGlobalStepNode(node);
    } else if ((node->GetType() == "FrameworkOp") && (node->GetName() == "IteratorGetNext")) {
      // Train Graph must has GetNext.
      is_iterator_v2 = true;
    }
    if (is_iterator_v2 && is_train_graph) {
      break;
    }
  }
  GELOGI("Generate: compute_graph is_iterator_v2[%d], is_train_graph[%d].", is_iterator_v2, is_train_graph);
  return (is_iterator_v2 && is_train_graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::MatchDumpStr(const std::string &suffix) {
  char *dump_level = std::getenv(kDumpGraphLevel);
  int64_t dump_graph_level =
      (dump_level != nullptr) ? std::strtol(dump_level, nullptr, kBaseOfIntegerValue) : kDumpLevel2;
  if (dump_graph_level == kDumpLevel1) {
    return false;
  }

  if (dump_graph_level == kDumpLevel2 && ((suffix.find(kDumpStrPartition) != std::string::npos) ||
                                          (suffix.find(kDumpStrOptimizeSubgraph) != std::string::npos) ||
                                          (suffix.find(kDumpStrAicpu) != std::string::npos))) {
    return true;
  }

  if (dump_graph_level == kDumpLevel3 && suffix.compare(kDumpStrBuild) != 0) {
    return true;
  }

  return false;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraph(const ge::ComputeGraphPtr &graph,
                                                                            const std::string &suffix,
                                                                            bool is_always_dump) {
#ifdef FMK_SUPPORT_DUMP
  char *dump_ge_graph = std::getenv(kDumpGeGraph);
  GE_IF_BOOL_EXEC(dump_ge_graph == nullptr && !is_always_dump, return;);

  // dump the graph according to different graph level
  if (GraphUtils::MatchDumpStr(suffix)) {
    return;
  }

  // file name
  static int file_idx = 0;
  const int dump_graph_index_width = 5;
  file_idx++;
  GELOGD("Start to dump om txt: %d", file_idx);

  static int max_dumpfile_num = 0;
  if (max_dumpfile_num == 0) {
    string opt = "0";
    (void)GetContext().GetOption("ge.maxDumpFileNum", opt);
    max_dumpfile_num = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
  }
  if (max_dumpfile_num != 0 && file_idx > max_dumpfile_num) {
    GELOGW("dump graph file cnt > maxDumpFileNum, maxDumpFileCnt=%d.", max_dumpfile_num);
    return;
  }

  std::stringstream stream_file_name;
  stream_file_name << "ge_proto_" << std::setw(dump_graph_index_width) << std::setfill('0') << file_idx;
  stream_file_name << "_" << suffix << ".txt";
  std::string proto_file = stream_file_name.str();

  // Create buffer
  ge::Model model("", "");
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(std::const_pointer_cast<ComputeGraph>(graph)));
  Buffer buffer;
  model.Save(buffer);

  // Write file
  ge::proto::ModelDef ge_proto;
  if (buffer.GetData() != nullptr) {
    std::string str(reinterpret_cast<const char *>(buffer.GetData()), buffer.GetSize());
    if (!ge_proto.ParseFromString(str)) {
      GELOGE(GRAPH_FAILED, "parse from string failed.");
      return;
    }
    char real_path[PATH_MAX] = {0x00};
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strlen(proto_file.c_str()) >= PATH_MAX, return, "file path is too longer!");
    GE_IF_BOOL_EXEC(realpath(proto_file.c_str(), real_path) == nullptr,
                    GELOGI("file %s does not exist, it will be created.", proto_file.c_str()));

    GraphUtils::WriteProtoToTextFile(ge_proto, real_path);
  }
#else
  GELOGW("need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::LoadGEGraph(const char *file,
                                                                            ge::ComputeGraph &compute_graph) {
  ge::proto::ModelDef model_def;
  // Get ModelDef object from file generated by DumpGEGraph()
  if (!ReadProtoFromTextFile(file, &model_def)) {
    GELOGE(GRAPH_FAILED, "Get ModelDef failed from file");
    return false;
  }
  ge::Model model;
  // Get Model object from ModelDef by deserialize ModelDef
  if (model.Load(model_def) == GRAPH_SUCCESS) {
    compute_graph = *(GraphUtils::GetComputeGraph(model.GetGraph()));
    return true;
  } else {
    GELOGE(GRAPH_FAILED, "Get Model failed from ModelDef");
    return false;
  }
}

// Printing protocol messages in text format is useful for debugging and human editing of messages.
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::WriteProtoToTextFile(
    const google::protobuf::Message &proto, const char *real_path) {
#ifdef FMK_SUPPORT_DUMP
  const int FILE_AUTHORITY = 0600;
  int fd = open(real_path, O_WRONLY | O_CREAT | O_TRUNC, FILE_AUTHORITY);
  if (fd < 0) {
    GELOGE(GRAPH_FAILED, "fail to open the file: %s", real_path);
    return;
  }
  google::protobuf::io::FileOutputStream *output = new (std::nothrow) FileOutputStream(fd);
  if (output == nullptr) {
    GELOGE(GRAPH_FAILED, "Output is nullptr");
    if (close(fd) != 0) {
      GELOGE(GRAPH_FAILED, "Close fileoutputstream failed");
    }
    return;
  }
  bool ret = google::protobuf::TextFormat::Print(proto, output);
  if (!ret) {
    GELOGE(GRAPH_FAILED, "Fail to write the file: %s", real_path);
    delete output;
    output = nullptr;
    GE_CHK_BOOL_EXEC(close(fd) == 0, return, "Close fileoutputstream failed");
    return;
  }
  delete output;
  output = nullptr;
  GE_CHK_BOOL_EXEC(close(fd) == 0, return, "Close fileoutputstream failed");

  FILE *file = fopen(real_path, "rb");
  if (file == nullptr) {
    return;
  }
  if (fseek(file, 0L, SEEK_END) == 0) {
    int64_t fileSize = ftell(file);
    static int64_t maxDumpFileSize = 0;
    if (maxDumpFileSize == 0) {
      string opt = "0";
      (void)GetContext().GetOption("ge.maxDumpFileSize", opt);
      maxDumpFileSize = atol(opt.c_str());
    }
    if (maxDumpFileSize != 0 && fileSize != -1 && fileSize > maxDumpFileSize) {
      GELOGW("dump graph file size > maxDumpFileSize, maxDumpFileSize=%ld.", maxDumpFileSize);
      GE_IF_BOOL_EXEC(std::remove(real_path) != 0, GELOGW("remove %s failed", real_path));
      GE_CHK_BOOL_EXEC(fclose(file) == 0, return, "Fclose %s failed", real_path);
      return;
    }
  }
  GE_CHK_BOOL_EXEC(fclose(file) == 0, return, "Fclose fileoutputstream failed");
#else
  GELOGW("need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::ReadProtoFromTextFile(
    const char *file, google::protobuf::Message *proto) {
  if (file == nullptr || proto == nullptr) {
    GELOGE(GRAPH_FAILED, "incorrect parameter. file path or message is invalid");
    return false;
  }
  std::ifstream fs(file, std::ifstream::in);
  if (!fs.is_open()) {
    GELOGE(GRAPH_FAILED, "proto file '%s' open fail.", file);
    return false;
  }
  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, proto);
  if (!ret) {
    GELOGE(GRAPH_FAILED, "parse proto from text ret fail, please check your text file '%s'.", file);
  }
  fs.close();
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraphToOnnx(const ge::ComputeGraph &compute_graph,
                                                                                  const std::string &suffix) {
#ifdef FMK_SUPPORT_DUMP
  char *dump_ge_graph = std::getenv(kDumpGeGraph);
  int64_t dump_ge_graph_level =
      (dump_ge_graph != nullptr) ? std::strtol(dump_ge_graph, nullptr, kBaseOfIntegerValue) : OnnxUtils::NO_DUMP;
  if ((dump_ge_graph_level == OnnxUtils::NO_DUMP) || (dump_ge_graph_level >= OnnxUtils::DUMP_LEVEL_END)) {
    GELOGD("Skip DumpGEGraphToOnnx with dump_ge_graph_level %ld.", dump_ge_graph_level);
    return;
  }

  // dump the graph according to different graph level
  if (GraphUtils::MatchDumpStr(suffix)) {
    return;
  }

  // 1.Get onnx::ModelProto from ge::Model
  ge::Model model("GE", "");
  std::shared_ptr<ge::ComputeGraph> compute_graph_ptr = ComGraphMakeShared<ge::ComputeGraph>(compute_graph);
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(std::const_pointer_cast<ComputeGraph>(compute_graph_ptr)));
  onnx::ModelProto model_proto;
  if (!OnnxUtils::ConvertGeModelToModelProto(model, model_proto)) {
    GELOGE(GRAPH_FAILED, "DumpGEGraphToOnnx failed.");
    return;
  }

  // 2.Set file name
  static int file_index = 0;
  file_index++;
  GELOGD("Start to dump ge onnx file: %d", file_index);

  static int max_dumpfile_num = 0;
  if (max_dumpfile_num == 0) {
    string opt = "0";
    (void)GetContext().GetOption("ge.maxDumpFileNum", opt);
    max_dumpfile_num = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
  }
  if (max_dumpfile_num != 0 && file_index > max_dumpfile_num) {
    GELOGW("dump graph file cnt > maxDumpFileNum, maxDumpFileNum=%d.", max_dumpfile_num);
    return;
  }

  /// 99999 graphs can be dumped at most at one time
  /// setw(5) is for formatted sort
  std::stringstream stream_file_name;
  stream_file_name << "ge_onnx_" << std::setw(5) << std::setfill('0') << file_index;
  stream_file_name << "_" << suffix << ".pbtxt";
  std::string proto_file = stream_file_name.str();
  if ((proto_file.length()) >= NAME_MAX) {
    GELOGE(GRAPH_FAILED, "File name is too longer!");
    return;
  }
  std::unique_ptr<char> real_path(new (std::nothrow) char[PATH_MAX]{0});
  if (real_path == nullptr) {
    GELOGE(GRAPH_FAILED, "New real_path failed.");
    return;
  }
  /// Returning nullptr means 3 case as follows:
  /// a.path is PATH_MAX chars or more
  /// b.the file does not exist
  /// c.the path has no permissions
  /// Distinguish between last the two cases in the function WriteProtoToTextFile call open()
  if (realpath(proto_file.c_str(), real_path.get()) == nullptr) {
    // For case a
    if (errno == ENAMETOOLONG) {
      GELOGE(GRAPH_FAILED, "Call realpath failed: path is PATH_MAX chars or more.");
      return;
    }
  }

  // 3. Serialize to file in current path
  GraphUtils::WriteProtoToTextFile(model_proto, real_path.get());
#else
  GELOGW("need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::LoadGEGraphFromOnnx(const char *file,
                                                                                    ge::ComputeGraph &compute_graph) {
  if (file == nullptr) {
    GELOGE(GRAPH_FAILED, "incorrect parameter. file path is invalid");
    return false;
  }
  onnx::ModelProto model_proto;
  // 1. Get ModelDef object from file generated by DumpGEGraphToOnnx()
  if (!ReadProtoFromTextFile(file, &model_proto)) {
    GELOGE(GRAPH_FAILED, "Get ModelDef from file failed");
    return false;
  }
  // 2.Convert onnx::ModelProto To ge::Model
  ge::Model model;
  if (!OnnxUtils::ConvertModelProtoToGeModel(model_proto, model)) {
    GELOGE(GRAPH_FAILED, "Convert ModelDef to Model failed");
    return false;
  }
  auto compute_graph_ptr = GraphUtils::GetComputeGraph(model.GetGraph());
  if (compute_graph_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "Get compute graph from Model failed");
    return false;
  }
  compute_graph = *(compute_graph_ptr);
  return true;
}

namespace {
using InNodesToOut = std::unordered_map<NodePtr, std::unordered_set<NodePtr>>;

inline std::string GetNodeNameByAnchor(const Anchor *anchor) {
  if (anchor == nullptr) {
    GELOGE(GRAPH_FAILED, "Anchor is nullptr");
    return "Null";
  }
  auto node = anchor->GetOwnerNode();
  return node == nullptr ? "Null" : node->GetName();
}

graphStatus ReplaceOutDataAnchor(const OutDataAnchorPtr &new_anchor, const OutDataAnchorPtr &old_anchor,
                                 InNodesToOut *in_nodes_to_out = nullptr) {
  if (new_anchor == nullptr || old_anchor == nullptr) {
    GELOGE(GRAPH_FAILED, "new_anchor or old_anchor is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  auto new_node = new_anchor->GetOwnerNode();
  for (const auto &peer_in_anchor : old_anchor->GetPeerInDataAnchors()) {
    auto ret = peer_in_anchor->Unlink(old_anchor);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Failed to unlink old anchor link from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(old_anchor.get()).c_str(), old_anchor->GetIdx(),
             GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
    ret = peer_in_anchor->LinkFrom(new_anchor);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Failed to relink new anchors from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(new_anchor.get()).c_str(), new_anchor->GetIdx(),
             GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }

    if (in_nodes_to_out != nullptr) {
      (*in_nodes_to_out)[new_node].insert(peer_in_anchor->GetOwnerNode());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus RelinkDataIO(const NodePtr &node, const std::vector<int> &io_map, InNodesToOut &in_nodes_to_out) {
  GE_CHECK_NOTNULL(node);
  auto in_data_anchors = node->GetAllInDataAnchors();
  auto out_data_anchors = node->GetAllOutDataAnchors();
  if (out_data_anchors.size() < io_map.size()) {
    GELOGE(GRAPH_FAILED, "The io_map specified for node %s type %s is larger %zu than the actual size %zu",
           node->GetName().c_str(), node->GetType().c_str(), io_map.size(), out_data_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  for (size_t i = 0; i < out_data_anchors.size(); ++i) {
    auto out_data_anchor = out_data_anchors.at(i);
    if (out_data_anchor == nullptr) {
      GELOGE(GRAPH_FAILED, "Failed to relink for node %s type %s, the out data anchor at index %zu is null",
             node->GetName().c_str(), node->GetType().c_str(), i);
      return GRAPH_FAILED;
    }

    int in_index = -1;
    if (i < io_map.size()) {
      in_index = io_map.at(i);
    }
    if (in_index < 0) {
      out_data_anchor->UnlinkAll();
      continue;
    }

    if (in_index >= static_cast<int>(in_data_anchors.size())) {
      GELOGE(GRAPH_PARAM_INVALID, "Failed to relink for node %s type %s, invalid index %d specified for input(%zu)",
             node->GetName().c_str(), node->GetType().c_str(), in_index, in_data_anchors.size());
      return GRAPH_PARAM_INVALID;
    }
    auto in_anchor = in_data_anchors.at(in_index);
    if (in_anchor == nullptr) {
      GELOGW("Invalid in data anchors(null) found at node %s type %s index %d, ignore it.", node->GetName().c_str(),
             node->GetType().c_str(), in_index);
      continue;
    }
    auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    if (peer_out_anchor->Unlink(in_anchor) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED,
             "Failed relink node %s type %s, failed to unlink the data link"
             " from %s(%d) to it at input-index %d",
             node->GetName().c_str(), node->GetType().c_str(), GetNodeNameByAnchor(peer_out_anchor.get()).c_str(),
             peer_out_anchor->GetIdx(), in_index);
      return GRAPH_FAILED;
    }
    auto ret = ReplaceOutDataAnchor(peer_out_anchor, out_data_anchor, &in_nodes_to_out);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Failed to relink node %s type %s for relinking data anchors", node->GetName().c_str(),
             node->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    in_anchor->UnlinkAll();
  }
  return GRAPH_SUCCESS;
}

InNodesToOut GetFullConnectIONodes(const NodePtr &node) {
  InNodesToOut in_nodes_to_out;
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "Node is nullptr,node is %s", node->GetName().c_str());
    return in_nodes_to_out;
  }
  auto in_nodes_list = node->GetInNodes();
  auto out_nodes_list = node->GetOutNodes();
  auto out_nodes = std::unordered_set<NodePtr>(out_nodes_list.begin(), out_nodes_list.end());

  for (const auto &in_node : in_nodes_list) {
    in_nodes_to_out.insert(std::make_pair(in_node, out_nodes));
  }
  return in_nodes_to_out;
}

graphStatus RelinkControlNodeIfNeed(const NodePtr &node, InNodesToOut &in_nodes_to_out,
                                    InNodesToOut &connected_data_in_to_out) {
  GE_CHECK_NOTNULL(node);
  for (const auto &in_node_to_out : in_nodes_to_out) {
    auto &in_node = in_node_to_out.first;
    GE_CHECK_NOTNULL(in_node);
    auto &connected_data_out = connected_data_in_to_out[in_node];
    for (const auto &out_node : in_node_to_out.second) {
      GE_CHECK_NOTNULL(out_node);
      if (connected_data_out.count(out_node) == 0) {
        GE_CHECK_NOTNULL(in_node->GetOutControlAnchor());
        if (in_node->GetOutControlAnchor()->IsLinkedWith(out_node->GetInControlAnchor())) {
          continue;
        }
        auto ret = GraphUtils::AddEdge(in_node->GetOutControlAnchor(), out_node->GetInControlAnchor());
        if (ret != GRAPH_SUCCESS) {
          GELOGE(GRAPH_FAILED, "Failed to add control edge from %s to %s when isolating node %s type %s",
                 in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str(),
                 node->GetType().c_str());
          return GRAPH_FAILED;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ReplaceOutDataAnchors(const Node::Vistor<OutDataAnchorPtr> &new_outs,
                                  const Node::Vistor<OutDataAnchorPtr> &old_outs, const std::vector<int> &outputs_map) {
  auto new_out_size = new_outs.size();
  if (new_out_size < outputs_map.size()) {
    GELOGE(GRAPH_PARAM_INVALID,
           "Failed to replace out data anchors, the actual size %zu is less than the mapping size %zu", new_out_size,
           outputs_map.size());
    return GRAPH_PARAM_INVALID;
  }
  for (size_t i = 0; i < new_out_size; ++i) {
    auto &new_out_anchor = new_outs.at(i);
    if (new_out_anchor == nullptr) {
      GELOGE(GRAPH_FAILED, "Failed to replace out data anchors, the out data anchor on new node is null, index %zu", i);
      return GRAPH_FAILED;
    }
    if (i >= outputs_map.size()) {
      continue;
    }
    auto old_index = outputs_map.at(i);
    if (old_index < 0) {
      continue;
    }

    const OutDataAnchorPtr &old_out_anchor = old_outs.at(old_index);
    if (old_out_anchor == nullptr) {
      GELOGE(GRAPH_FAILED, "Failed to replace out data anchors, the out data anchor on old node is null, index %d",
             old_index);
      return GRAPH_FAILED;
    }
    auto ret = ReplaceOutDataAnchor(new_out_anchor, old_out_anchor);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus ReplaceInDataAnchors(const Node::Vistor<InDataAnchorPtr> &new_ins,
                                 const Node::Vistor<InDataAnchorPtr> &old_ins, const std::vector<int> &inputs_map) {
  auto new_in_size = new_ins.size();
  if (new_in_size < inputs_map.size()) {
    GELOGE(GRAPH_FAILED, "Failed to replace in data anchors, the actual size %zu is less than the mapping size %zu",
           new_in_size, inputs_map.size());
    return GRAPH_PARAM_INVALID;
  }

  for (size_t i = 0; i < new_in_size; ++i) {
    auto &new_in_anchor = new_ins.at(i);
    if (new_in_anchor == nullptr) {
      GELOGE(GRAPH_FAILED, "Failed to replace in data anchors, the out data anchor on new node is null, index %zu", i);
      return GRAPH_FAILED;
    }
    if (i >= inputs_map.size()) {
      continue;
    }
    auto old_index = inputs_map.at(i);
    if (old_index < 0) {
      continue;
    }
    const InDataAnchorPtr &old_in_anchor = old_ins.at(old_index);
    if (old_in_anchor == nullptr) {
      GELOGE(GRAPH_FAILED, "Failed to replace in data anchors, the out data anchor on old node is null, index %d",
             old_index);
      return GRAPH_FAILED;
    }

    auto peer_out_anchor = old_in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      GELOGW("Peer out anchor is nullptr");
      continue;
    }
    auto ret = peer_out_anchor->Unlink(old_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Failed to unlink old anchors, unlink from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
             GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
    ret = peer_out_anchor->LinkTo(new_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Failed to link new anchors, link from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
             GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ReplaceControlAnchors(const NodePtr &new_node, const NodePtr &old_node) {
  GE_CHECK_NOTNULL(new_node);
  GE_CHECK_NOTNULL(old_node);
  GE_CHECK_NOTNULL(new_node->GetInControlAnchor());
  GE_CHECK_NOTNULL(old_node->GetInControlAnchor());
  auto peer_out_anchors = old_node->GetInControlAnchor()->GetPeerAnchors();
  auto new_in_control_anchor = new_node->GetInControlAnchor();
  for (const auto &peer_out_anchor : peer_out_anchors) {
    if (peer_out_anchor != nullptr) {
      auto ret = GraphUtils::AddEdge(peer_out_anchor, new_in_control_anchor);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "Add edge failed");
        return GRAPH_FAILED;
      }
    }
  }
  auto old_out_control_anchor = old_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(old_out_control_anchor);
  auto peer_in_anchors = old_out_control_anchor->GetPeerAnchors();
  auto new_out_control_anchor = new_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(new_out_control_anchor);
  for (const auto &peer_in_anchor : peer_in_anchors) {
    if (peer_in_anchor != nullptr) {
      auto ret = GraphUtils::AddEdge(new_out_control_anchor, peer_in_anchor);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "Add edge failed");
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::IsolateNode(const NodePtr &node,
                                                                                   const std::vector<int> &io_map) {
  if (node == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "Failed to isolate node(null)");
    return GRAPH_PARAM_INVALID;
  }
  /// We must get full connections info before re-link data io, because the data
  /// edges may be unlinked when relink data io
  auto in_nodes_to_out = GetFullConnectIONodes(node);
  InNodesToOut data_in_to_out;
  auto ret = RelinkDataIO(node, io_map, data_in_to_out);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Failed to isolate node %s type %s when relink data IO", node->GetName().c_str(),
           node->GetType().c_str());
    return ret;
  }

  ret = RelinkControlNodeIfNeed(node, in_nodes_to_out, data_in_to_out);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  NodeUtils::UnlinkAll(*node);

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::IsolateNode(const NodePtr &node, const std::initializer_list<int> &io_map) {
  return IsolateNode(node, std::vector<int>(io_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::IsolateNodeOneIO(const NodePtr &node) {
  if (node == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "incorrect parameter. node is invalid");
    return GRAPH_PARAM_INVALID;
  }
  if (node->GetAllInDataAnchorsSize() != 1) {
    return GRAPH_PARAM_INVALID;
  }
  if (node->GetAllOutDataAnchorsSize() != 1) {
    return GRAPH_PARAM_INVALID;
  }
  return IsolateNode(node, {0});
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeAnchors(const NodePtr &new_node, const NodePtr &old_node, const std::vector<int> &inputs_map,
                               const std::vector<int> &outputs_map) {
  if ((new_node == nullptr) || (old_node == nullptr)) {
    GELOGE(GRAPH_FAILED, "Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  auto ret = ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  if (ret != GRAPH_SUCCESS) {
    // The error log was printed in `ReplaceNodeDataAnchors`
    return GRAPH_FAILED;
  }
  ret = ReplaceControlAnchors(new_node, old_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED,
           "Failed to replace control anchors when replace node from old node %s type %s to new node %s type %s",
           old_node->GetName().c_str(), old_node->GetType().c_str(), new_node->GetName().c_str(),
           new_node->GetType().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ReplaceNodeAnchors(
    const NodePtr &new_node, const NodePtr &old_node, const std::initializer_list<int> inputs_map,
    const std::initializer_list<int> outputs_map) {
  return ReplaceNodeAnchors(new_node, old_node, std::vector<int>(inputs_map), std::vector<int>(outputs_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                   std::initializer_list<int> inputs_map, std::initializer_list<int> outputs_map) {
  return ReplaceNodeDataAnchors(new_node, old_node, std::vector<int>(inputs_map), std::vector<int>(outputs_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node, const std::vector<int> &inputs_map,
                                   const std::vector<int> &outputs_map) {
  if (new_node == nullptr || old_node == nullptr) {
    GELOGE(GRAPH_FAILED, "Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }

  auto ret = ReplaceOutDataAnchors(new_node->GetAllOutDataAnchors(), old_node->GetAllOutDataAnchors(), outputs_map);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED,
           "Failed to replace out data anchors when replace node from old node %s type %s to new node %s type %s",
           old_node->GetName().c_str(), old_node->GetType().c_str(), new_node->GetName().c_str(),
           new_node->GetType().c_str());
    return GRAPH_FAILED;
  }
  ret = ReplaceInDataAnchors(new_node->GetAllInDataAnchors(), old_node->GetAllInDataAnchors(), inputs_map);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED,
           "Failed to replace in data anchors when replace node from old node %s type %s to new node %s type %s",
           old_node->GetName().c_str(), old_node->GetType().c_str(), new_node->GetName().c_str(),
           new_node->GetType().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyInCtrlEdges(const NodePtr &src_node,
                                                                                       NodePtr &dst_node) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    GELOGE(GRAPH_FAILED, "Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  auto src_ctrl_in_nodes = src_node->GetInControlNodes();
  if (src_ctrl_in_nodes.empty()) {
    return GRAPH_SUCCESS;
  }

  std::unordered_set<NodePtr> exist_in_ctrl_nodes_set;
  auto exist_in_ctrl_nodes = dst_node->GetInControlNodes();
  if (!exist_in_ctrl_nodes.empty()) {
    exist_in_ctrl_nodes_set.insert(exist_in_ctrl_nodes.begin(), exist_in_ctrl_nodes.end());
  }

  auto dst_ctrl = dst_node->GetInControlAnchor();
  for (const auto &in_node : src_ctrl_in_nodes) {
    if (exist_in_ctrl_nodes_set.count(in_node) > 0) {
      continue;
    }
    auto ret = GraphUtils::AddEdge(in_node->GetOutControlAnchor(), dst_ctrl);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Failed to add control edge from %s to %s when copy control dependencies from %s to %s",
             in_node->GetName().c_str(), dst_node->GetName().c_str(), src_node->GetName().c_str(),
             dst_node->GetName().c_str());
      return ret;
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::MoveInCtrlEdges(const NodePtr &src_node,
                                                                                       NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    GELOGE(GRAPH_FAILED, "Parameter is nullptr");
    return GRAPH_FAILED;
  }
  auto ret = CopyInCtrlEdges(src_node, dst_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Copy in ctrl edges failed");
    return ret;
  }
  GE_CHECK_NOTNULL(src_node->GetInControlAnchor());
  src_node->GetInControlAnchor()->UnlinkAll();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyOutCtrlEdges(const NodePtr &src_node,
                                                                                        NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    GELOGE(GRAPH_FAILED, "Parameter is nullptr");
    return GRAPH_FAILED;
  }
  auto out_ctrl_nodes = src_node->GetOutControlNodes();
  if (out_ctrl_nodes.empty()) {
    return GRAPH_SUCCESS;
  }

  std::unordered_set<Node *> exists_out_ctrl_nodes_set;
  for (const auto &node : dst_node->GetOutControlNodes()) {
    exists_out_ctrl_nodes_set.insert(node.get());
  }

  auto dst_out_ctrl = dst_node->GetOutControlAnchor();
  for (const auto &node : out_ctrl_nodes) {
    if (exists_out_ctrl_nodes_set.count(node.get()) > 0) {
      continue;
    }
    auto ret = GraphUtils::AddEdge(dst_out_ctrl, node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Failed to add control edge from %s to %s when copy control dependencies from %s to %s",
             dst_node->GetName().c_str(), node->GetName().c_str(), src_node->GetName().c_str(),
             dst_node->GetName().c_str());
      return ret;
    }
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::MoveOutCtrlEdges(NodePtr &src_node,
                                                                                        NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    GELOGE(GRAPH_FAILED, "Parameter is nullptr");
    return GRAPH_FAILED;
  }
  auto ret = CopyOutCtrlEdges(src_node, dst_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Copyout ctrl edges failed");
    return ret;
  }
  GE_CHECK_NOTNULL(src_node->GetOutControlAnchor());
  src_node->GetOutControlAnchor()->UnlinkAll();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AppendInputNode(const ComputeGraphPtr &graph,
                                                                                       const NodePtr &node) {
  if (graph->AddInputNode(node) == nullptr) {
    GELOGE(GRAPH_FAILED, "Copyout ctrl edges failed");
    return GRAPH_FAILED;
  }
  graph->SetInputSize(graph->GetInputSize() + 1);
  graph->inputs_order_.emplace_back(node->GetName());
  return GRAPH_SUCCESS;
}
}  // namespace ge
