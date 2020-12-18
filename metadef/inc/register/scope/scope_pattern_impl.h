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

#ifndef REGISTER_SCOPE_SCOPE_PATTERN_IMPL_H_
#define REGISTER_SCOPE_SCOPE_PATTERN_IMPL_H_

#include "external/register/scope/scope_fusion_pass_register.h"

namespace ge {
class ScopeAttrValue::ScopeAttrValueImpl {
 public:
  ScopeAttrValueImpl() : int_value_(0), float_value_(0.0), string_value_(""), bool_value_(false) {}
  ~ScopeAttrValueImpl() {}

  void SetIntValue(const int64_t &value) { int_value_ = value; }
  void SetFloatValue(const float &value) { float_value_ = value; }
  void SetStringValue(const std::string &value) { string_value_ = value; }
  void SetBoolValue(const bool &value) { bool_value_ = value; }
  const int64_t &GetIntValue() const { return int_value_; }
  const float &GetFloatValue() const { return float_value_; }
  const std::string &GetStrValue() const { return string_value_; }
  const bool &GetBoolValue() const { return bool_value_; }

 private:
  int64_t int_value_;
  float float_value_;
  std::string string_value_;
  bool bool_value_;
};

class NodeOpTypeFeature::NodeOpTypeFeatureImpl : ScopeBaseFeature {
 public:
  NodeOpTypeFeatureImpl(std::string nodeType, int num, int step = 0)
      : node_type_(nodeType), num_(num), step_(step) {}
  ~NodeOpTypeFeatureImpl() {}
  bool Match(const Scope *scope) override;

 public:
  std::string node_type_;  // Node type
  int num_;           // Node number
  int step_;          // step
};

class NodeAttrFeature::NodeAttrFeatureImpl : ScopeBaseFeature {
 public:
  NodeAttrFeatureImpl(std::string nodeType, std::string attr_name, ge::DataType datatype, ScopeAttrValue &attr_value)
      : node_type_(nodeType), attr_name_(attr_name), datatype_(datatype), attr_value_(attr_value) {}
  ~NodeAttrFeatureImpl() {}
  bool Match(const Scope *scope) override;

 public:
  std::string node_type_;                        // Node type
  std::string attr_name_;                        // attribute name
  ge::DataType datatype_;     // datatype
  ScopeAttrValue attr_value_;  // AttrValue
};

class ScopeFeature::ScopeFeatureImpl : ScopeBaseFeature {
 public:
  ScopeFeatureImpl(std::string sub_type, int32_t num, std::string suffix = "",
                   std::string sub_scope_mask = "", int step = 0)
      : sub_type_(sub_type), num_(num), suffix_(suffix), sub_scope_mask_(sub_scope_mask), step_(step) {}
  ~ScopeFeatureImpl() {}
  bool Match(const Scope *scope) override;
  bool SubScopesMatch(const std::vector<Scope *> &scopes);

 public:
  std::string sub_type_;
  int32_t num_;
  std::string suffix_;
  std::string sub_scope_mask_;
  int step_;
};

class ScopePattern::ScopePatternImpl {
 public:
  ScopePatternImpl() {}
  ~ScopePatternImpl() {}
  bool Match(const Scope *scope) const;
  void SetSubType(const std::string &sub_type);
  const std::string &SubType() const { return sub_type_; }
  void AddNodeOpTypeFeature(NodeOpTypeFeature &feature);
  void AddNodeAttrFeature(NodeAttrFeature &feature);
  void AddScopeFeature(ScopeFeature &feature);

 private:
  std::string sub_type_;  // get Scope sub type
  std::vector<NodeOpTypeFeature> node_optype_features_;
  std::vector<NodeAttrFeature> node_attr_features_;
  std::vector<ScopeFeature> scopes_features_;
};
}  // namespace ge
#endif  // REGISTER_SCOPE_SCOPE_PATTERN_IMPL_H_