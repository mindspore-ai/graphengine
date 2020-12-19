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

#ifndef INC_GRAPH_MODEL_H_
#define INC_GRAPH_MODEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "detail/attributes_holder.h"
#include "graph/ge_attr_value.h"
#include "graph/graph.h"

namespace ge {
using std::map;
using std::string;
using std::vector;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Model : public AttrHolder {
 public:
  Model();

  ~Model() = default;

  Model(const string &name, const string &custom_version);

  string GetName() const;
  void SetName(const string &name);

  uint32_t GetVersion() const;

  void SetVersion(uint32_t version) { version_ = version; }

  std::string GetPlatformVersion() const;

  void SetPlatformVersion(string version) { platform_version_ = version; }

  Graph GetGraph() const;

  void SetGraph(const Graph &graph);

  void SetAttr(const ProtoAttrMapHelper &attrs);

  using AttrHolder::GetAllAttrNames;
  using AttrHolder::GetAllAttrs;
  using AttrHolder::GetAttr;
  using AttrHolder::HasAttr;
  using AttrHolder::SetAttr;

  graphStatus Save(Buffer &buffer, bool is_dump = false) const;

  graphStatus SaveToFile(const string& file_name) const;
  // Model will be rewrite
  static graphStatus Load(const uint8_t *data, size_t len, Model &model);
  graphStatus Load(ge::proto::ModelDef &model_def);
  graphStatus LoadFromFile(const string& file_name);

  bool IsValid() const;

 protected:
  ConstProtoAttrMapHelper GetAttrMap() const override;
  ProtoAttrMapHelper MutableAttrMap() override;

 private:
  void Init();
  ProtoAttrMapHelper attrs_;
  friend class ModelSerializeImp;
  friend class GraphDebugImp;
  friend class OnnxUtils;
  friend class ModelHelper;
  friend class ModelBuilder;
  string name_;
  uint32_t version_;
  std::string platform_version_{""};
  Graph graph_;
};
}  // namespace ge
using ModelPtr = std::shared_ptr<ge::Model>;

#endif  // INC_GRAPH_MODEL_H_
