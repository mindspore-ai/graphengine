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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_FUSION_PATTERN_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_FUSION_PATTERN_H_
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using std::initializer_list;
using std::map;
using std::string;
using std::vector;

using namespace std;

namespace fe {

/** Fusion pattern
 * @ingroup FUSION_PASS_GROUP
 * Describe Pattern of Ops waiting for fusion(Op type, etc)
 */
class FusionPattern {
 public:
  struct OpDesc;
  using OpDescPtr = std::shared_ptr<OpDesc>;
  /**
   * @ingroup fe
   * @brief description of Ops
   */
  struct OpDesc {
    string id;                       // Identifier
    std::vector<std::string> types;  // the Op types of Ops
    std::vector<OpDescPtr> inputs;   // all input Ops
    bool repeatable;                 // flag to show if match multiple Ops or not
    bool is_output;                  // flag to show if the op is output node
  };

 public:
  explicit FusionPattern(string name = "");
  ~FusionPattern();

  /** set pattern name
   *
   * @param name pattern name
   * @return FusionPattern
   */
  FusionPattern &SetName(const string &name);

  /** add Op description with unknown number of args
   *
   * @param id pattern id
   * @param types op type list
   * @return FusionPattern
   */
  FusionPattern &AddOpDesc(const string &id, const initializer_list<string> &types = {});

  /** add Op description with vector
   *
   * @param id pattern id
   * @param types op type list
   *
   * @return FusionPattern
   */
  FusionPattern &AddOpDesc(const string &id, const vector<string> &types);

  /** set input Ops with unknown number of args
   *
   * @param id pattern id
   *
   * @param input_ids inputs to id op
   *
   * @return FusionPattern
   */
  FusionPattern &SetInputs(const string &id, const initializer_list<string> &input_ids);

  /** set input Ops with unknown number of args
   *
   * @param id pattern id
   *
   * @param input_ids inputs to id op
   *
   * @return FusionPattern
   */
  FusionPattern &SetInputs(const string &id, const vector<string> &input_ids);

  /** set output Op
   *
   * @param id pattern id
   *
   * @return FusionPattern
   */
  FusionPattern &SetOutput(const string &id);

  /** build pattern and check if error exists
   *
   * @return True or False
   */
  bool Build();

  /** get pattern name
   *
   * @param id pattern id
   *
   * @return fusion pattern name
   */
  const string &GetName() const;

  /** get the OpDesc of input Ops (const)
   *
   * @param op_desc op_desc for getting inputs
   *
   * @return op_desc's iniput opdesc list
   */
  static const vector<std::shared_ptr<OpDesc>> *GetInputs(std::shared_ptr<OpDesc> op_desc);

  /** get the OpDesc of output Op
   *
   * @return pattern's output opdesc list
   */
  const std::shared_ptr<FusionPattern::OpDesc> GetOutput() const;

  /** print pattern
   *
   */
  void Dump() const;

  void GetOpDescList(vector<std::shared_ptr<OpDesc>> &op_desc_list);

  /** get OpDesc based on ID, return nullptr if failed
   *
   * @param id pattern id
   *
   * @return pattern's output opdesc list
   */
  std::shared_ptr<FusionPattern::OpDesc> GetOpDesc(const string &id) const;

 private:
  FusionPattern(const FusionPattern &) = default;
  FusionPattern &operator=(const FusionPattern &) = default;

  void SetError();

 private:
  string name_;

  vector<std::shared_ptr<OpDesc>> ops_;

  map<string, std::shared_ptr<OpDesc>> op_map_;

  std::shared_ptr<OpDesc> output_;

  bool has_error_;
};

}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_FUSION_PATTERN_H_
