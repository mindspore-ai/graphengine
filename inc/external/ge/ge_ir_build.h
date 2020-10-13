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

#ifndef INC_EXTERNAL_GE_IR_BUILD_H_
#define INC_EXTERNAL_GE_IR_BUILD_H_

#include <string>
#include <map>
#include <memory>
#include "graph/graph.h"
#include "graph/ge_error_codes.h"

namespace {
#define IR_MAJOR_VERSION (int(1))
#define IR_MINOR_VERSION (int(0))
#define IR_PATCH_VERSION (int(0))
}

namespace ge{

struct ModelBufferData
{
  std::shared_ptr<uint8_t> data = nullptr;
  uint64_t length;
};

/**
 * @ingroup AscendCL
 * @brief build model.Notice the model is stored in buffer
 *
 * @param global_options[IN] global init params for build
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
graphStatus aclgrphBuildInitialize(std::map<std::string, std::string> global_options);

/**
 * @ingroup AscendCL
 * @brief build model.Notice the model is stored in buffer
 *
 */
void aclgrphBuildFinalize();

/**
 * @ingroup AscendCL
 * @brief build model.Notice the model is stored in buffer
 *
 * @param graph[IN]   the graph ready to build
 * @param options[IN] options used for build
 * @param model[OUT]  builded model
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
graphStatus aclgrphBuildModel(const ge::Graph &graph, const std::map<std::string, std::string> &build_options, ModelBufferData& model);

/**
 * @ingroup AscendCL
 * @brief save model buffer to file
 *
 * @param output_file[IN]   the file path to be saved
 * @param model[IN]         model buffer data
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
graphStatus aclgrphSaveModel(const string &output_file, const ModelBufferData& model);

/**
 * @ingroup AscendCL
 * @brief query IR interface version
 *
 * @param major_version[OUT] IR interface major version
 * @param minor_version[OUT] IR interface minor version
 * @param patch_version[OUT] IR interface patch version
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
graphStatus aclgrphGetIRVersion(int *major_version, int *minor_version, int *patch_version);

/**
 * @ingroup AscendCL
 * @brief infer shape and data type
 *
 * @param graph[IN] the graph ready to build
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
graphStatus aclgrphInferShapeAndType(ge::Graph &graph);

/**
 * @ingroup AscendCL
 * @brief dump graph
 *
 * @param graph[IN] the graph ready to build
 * @param file[IN] file path
 * @param file[IN] file path string len
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
graphStatus aclgrphDumpGraph(const ge::Graph &graph, const char *file, const size_t len);
}; // INC_EXTERNAL_GE_IR_BUILD_H_
#endif
