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

#ifndef INC_GRAPH_UTILS_ATTR_UTILS_H_
#define INC_GRAPH_UTILS_ATTR_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include "graph/detail/attributes_holder.h"
#include "graph/ge_attr_value.h"
#include "graph/types.h"

namespace ge {
class OpDesc;
using OpDescPtr = std::shared_ptr<OpDesc>;
using ConstOpDescPtr = std::shared_ptr<const OpDesc>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AttrUtils {
 public:
  class ConstAttrHolderAdapter;
  class AttrHolderAdapter;
  // Set
  static bool HasAttr(ConstAttrHolderAdapter &&obj, const string &name);

  static bool SetInt(AttrHolderAdapter &&obj, const string &name, const int64_t &value);
  static bool SetListInt(AttrHolderAdapter &&obj, const string &name, const vector<int64_t> &value);
  static bool SetListInt(AttrHolderAdapter &&obj, const string &name, const vector<uint32_t> &value);
  static bool SetListInt(AttrHolderAdapter &&obj, const string &name, const vector<int32_t> &value);
  static bool SetListInt(AttrHolderAdapter &&obj, const string &name, std::initializer_list<int64_t> &&value);

  static bool SetFloat(AttrHolderAdapter &&obj, const string &name, const float &value);
  static bool SetListFloat(AttrHolderAdapter &&obj, const string &name, const vector<float> &value);
  static bool SetBool(AttrHolderAdapter &&obj, const string &name, const bool &value);
  static bool SetListBool(AttrHolderAdapter &&obj, const string &name, const vector<bool> &value);
  static bool SetStr(AttrHolderAdapter &&obj, const string &name, const string &value);
  static bool SetListStr(AttrHolderAdapter &&obj, const string &name, const vector<string> &value);
  static bool SetTensorDesc(AttrHolderAdapter &&obj, const string &name, const GeTensorDesc &value);
  static bool SetListTensorDesc(AttrHolderAdapter &&obj, const string &name, const vector<GeTensorDesc> &value);
  static bool SetTensor(AttrHolderAdapter &&obj, const string &name, const GeTensorPtr &value);
  static bool SetTensor(AttrHolderAdapter &&obj, const string &name, const ConstGeTensorPtr &value);
  static bool SetTensor(AttrHolderAdapter &&obj, const string &name, const GeTensor &value);
  static bool SetListTensor(AttrHolderAdapter &&obj, const string &name, const vector<GeTensorPtr> &value);
  static bool SetListTensor(AttrHolderAdapter &&obj, const string &name, const vector<ConstGeTensorPtr> &value);
  static bool SetListTensor(AttrHolderAdapter &&obj, const string &name,
                            std::initializer_list<ConstGeTensorPtr> &&value);
  static bool SetListTensor(AttrHolderAdapter &&obj, const string &name, const vector<GeTensor> &value);
  static bool SetGraph(AttrHolderAdapter &&obj, const string &name, const ComputeGraphPtr &value);
  static bool SetListGraph(AttrHolderAdapter &&obj, const string &name, const vector<ComputeGraphPtr> &value);
  static bool SetBytes(AttrHolderAdapter &&obj, const string &name, const GeAttrValue::BYTES &value);
  static bool SetListBytes(AttrHolderAdapter &&obj, const string &name, const vector<GeAttrValue::BYTES> &value);
  static bool SetNamedAttrs(AttrHolderAdapter &&obj, const string &name, const GeAttrValue::NAMED_ATTRS &value);
  static bool SetListNamedAttrs(AttrHolderAdapter &&obj, const string &name,
                                const vector<GeAttrValue::NAMED_ATTRS> &value);
  static bool SetListOpDesc(AttrHolderAdapter &&obj, const string &name, const vector<ConstOpDescPtr> &value);
  static bool SetListOpDesc(AttrHolderAdapter &&obj, const string &name, const vector<OpDescPtr> &value);

  // Get
  static bool GetInt(ConstAttrHolderAdapter &&obj, const string &name, int64_t &value);
  static bool GetInt(ConstAttrHolderAdapter &&obj, const string &name, int32_t &value);
  static bool GetInt(ConstAttrHolderAdapter &&obj, const string &name, uint32_t &value);
  static bool GetListInt(ConstAttrHolderAdapter &&obj, const string &name, vector<int64_t> &value);
  static bool GetListInt(ConstAttrHolderAdapter &&obj, const string &name, vector<int32_t> &value);
  static bool GetListInt(ConstAttrHolderAdapter &&obj, const string &name, vector<uint32_t> &value);
  static bool GetFloat(ConstAttrHolderAdapter &&obj, const string &name, float &value);
  static bool GetListFloat(ConstAttrHolderAdapter &&obj, const string &name, vector<float> &value);
  static bool GetBool(ConstAttrHolderAdapter &&obj, const string &name, bool &value);
  static bool GetListBool(ConstAttrHolderAdapter &&obj, const string &name, vector<bool> &value);
  static bool GetStr(ConstAttrHolderAdapter &&obj, const string &name, string &value);
  static bool GetListStr(ConstAttrHolderAdapter &&obj, const string &name, vector<string> &value);
  static bool GetTensorDesc(ConstAttrHolderAdapter &&obj, const string &name, GeTensorDesc &value);
  static bool GetListTensorDesc(ConstAttrHolderAdapter &&obj, const string &name, vector<GeTensorDesc> &value);
  static bool GetTensor(ConstAttrHolderAdapter &&obj, const string &name, ConstGeTensorPtr &value);
  static bool MutableTensor(AttrHolderAdapter &&obj, const string &name, GeTensorPtr &value);
  static bool GetListTensor(ConstAttrHolderAdapter &&obj, const string &name, vector<ConstGeTensorPtr> &value);
  static bool MutableListTensor(AttrHolderAdapter &&obj, const string &name, vector<GeTensorPtr> &value);
  static bool GetGraph(ConstAttrHolderAdapter &&obj, const string &name, ComputeGraphPtr &value);
  static bool GetListGraph(ConstAttrHolderAdapter &&obj, const string &name, vector<ComputeGraphPtr> &value);
  static bool GetBytes(ConstAttrHolderAdapter &&obj, const string &name, GeAttrValue::BYTES &value);
  static bool GetListBytes(ConstAttrHolderAdapter &&obj, const string &name, vector<GeAttrValue::BYTES> &value);
  static bool GetNamedAttrs(ConstAttrHolderAdapter &&obj, const string &name, GeAttrValue::NAMED_ATTRS &value);
  static bool GetListNamedAttrs(ConstAttrHolderAdapter &&obj, const string &name,
                                vector<GeAttrValue::NAMED_ATTRS> &value);
  static bool GetListOpDesc(ConstAttrHolderAdapter &&obj, const string &name, vector<OpDescPtr> &value);
  // Value will be moved
  static bool SetZeroCopyBytes(AttrHolderAdapter &&obj, const string &name, Buffer &&buffer);
  static bool GetZeroCopyBytes(ConstAttrHolderAdapter &&obj, const string &name, Buffer &buffer);
  // Value will be moved
  static bool SetZeroCopyListBytes(AttrHolderAdapter &&obj, const string &name,
                                   vector<Buffer> &listBuffer);
  static bool GetZeroCopyListBytes(ConstAttrHolderAdapter &&obj, const string &name, vector<Buffer> &listBuffer);

  static bool SetListListInt(AttrHolderAdapter &&obj, const string &name, const vector<vector<int64_t>> &value);
  static bool GetListListInt(ConstAttrHolderAdapter &&obj, const string &name, vector<vector<int64_t>> &value);

  static bool SetListDataType(AttrHolderAdapter &&obj, const string &name, const vector<ge::DataType> &value);
  static bool GetListDataType(ConstAttrHolderAdapter &&obj, const string &name, vector<ge::DataType> &value);

  static bool SetDataType(AttrHolderAdapter &&obj, const string &name, const ge::DataType &value);
  static bool GetDataType(ConstAttrHolderAdapter &&obj, const string &name, ge::DataType &value);

  static OpDescPtr CloneOpDesc(const ConstOpDescPtr &orgOpDesc);

  static OpDescPtr CopyOpDesc(const ConstOpDescPtr &orgOpDesc);

  static std::string GetAllAttrsStr(ConstAttrHolderAdapter &&obj);

  class AttrHolderAdapter {
   public:
    AttrHolderAdapter(AttrHolder *obj) : obj_(obj) {}
    ~AttrHolderAdapter() {}
    template <class T>
    AttrHolderAdapter(const std::shared_ptr<T> &obj) : obj_(obj.get()) {}
    AttrHolderAdapter(AttrHolder &obj) : obj_(&obj) {}
    operator bool() const { return obj_ != nullptr; }
    AttrHolder *operator->() { return obj_; }
    AttrHolder *get() { return obj_; }

    AttrHolder *obj_;
  };

  class ConstAttrHolderAdapter {
   public:
    ConstAttrHolderAdapter(const AttrHolder *obj) : obj_(obj) {}
    ~ConstAttrHolderAdapter() {}
    template <class T>
    ConstAttrHolderAdapter(const std::shared_ptr<T> obj) : obj_(obj.get()) {}
    ConstAttrHolderAdapter(const AttrHolder &obj) : obj_(&obj) {}
    operator bool() const { return obj_ != nullptr; }
    const AttrHolder *operator->() const { return obj_; }
    const AttrHolder *get() const { return obj_; }

   private:
    const AttrHolder *obj_;
  };
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_ATTR_UTILS_H_
