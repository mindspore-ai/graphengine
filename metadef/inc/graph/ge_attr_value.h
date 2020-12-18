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

#ifndef INC_GRAPH_GE_ATTR_VALUE_H_
#define INC_GRAPH_GE_ATTR_VALUE_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph/buffer.h"
#include "detail/attributes_holder.h"
#include "graph/ge_error_codes.h"
#include "graph/ge_tensor.h"

using std::map;
using std::string;
using std::vector;

namespace ge {
class GeTensor;

using GeTensorPtr = std::shared_ptr<GeTensor>;
using ConstGeTensorPtr = std::shared_ptr<const GeTensor>;

class ComputeGraph;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
using ConstComputeGraphPtr = std::shared_ptr<const ComputeGraph>;

class GeTensorDesc;
class GeAttrValue;
class GeAttrValueImp;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NamedAttrs : public AttrHolder {
 public:
  NamedAttrs();
  virtual ~NamedAttrs() = default;
  void SetName(const std::string &name);
  string GetName() const;
  GeAttrValue GetItem(const string &key) const;

 protected:
  ProtoAttrMapHelper MutableAttrMap() override;
  ConstProtoAttrMapHelper GetAttrMap() const override;

 private:
  // Create namedAttrs from protobuf obj
  NamedAttrs(const ProtoMsgOwner &owner, proto::NamedAttrs *protoMsg);
  GeIrProtoHelper<proto::NamedAttrs> named_attrs_;
  friend class GeAttrValueImp;
  friend class GeAttrValue;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeAttrValue {
 public:
  using INT = int64_t;
  using FLOAT = float;
  using BOOL = bool;
  using STR = std::string;
  using TENSOR = GeTensorPtr;
  using TENSOR_DESC = GeTensorDesc;
  using GRAPH = ComputeGraphPtr;
  using BYTES = Buffer;
  using NAMED_ATTRS = ge::NamedAttrs;
  using DATA_TYPE = ge::DataType;

  using LIST_INT = vector<INT>;
  using LIST_FLOAT = vector<FLOAT>;
  using LIST_BOOL = vector<BOOL>;
  using LIST_STR = vector<STR>;
  using LIST_TENSOR = vector<TENSOR>;
  using LIST_TENSOR_DESC = vector<TENSOR_DESC>;
  using LIST_GRAPH = vector<GRAPH>;
  using LIST_BYTES = vector<BYTES>;
  using LIST_NAMED_ATTRS = vector<NAMED_ATTRS>;
  using LIST_LIST_INT = vector<vector<int64_t>>;
  using LIST_DATA_TYPE = vector<ge::DataType>;

  using NamedAttrs = ge::NamedAttrs;    // for cce use (ge::GeAttrValue::NamedAttrs).

  enum ValueType {
    VT_NONE = 0,
    VT_STRING,
    VT_FLOAT,
    VT_BOOL,
    VT_INT,
    VT_TENSOR_DESC,
    VT_TENSOR,
    VT_BYTES,
    VT_GRAPH,
    VT_NAMED_ATTRS,
    VT_LIST_LIST_INT,
    VT_DATA_TYPE,

    VT_LIST_BASE = 1000,
    VT_LIST_STRING = VT_LIST_BASE + VT_STRING,
    VT_LIST_FLOAT = VT_LIST_BASE + VT_FLOAT,
    VT_LIST_BOOL = VT_LIST_BASE + VT_BOOL,
    VT_LIST_INT = VT_LIST_BASE + VT_INT,
    VT_LIST_TENSOR_DESC = VT_LIST_BASE + VT_TENSOR_DESC,
    VT_LIST_TENSOR = VT_LIST_BASE + VT_TENSOR,
    VT_LIST_BYTES = VT_LIST_BASE + VT_BYTES,
    VT_LIST_GRAPH = VT_LIST_BASE + VT_GRAPH,
    VT_LIST_NAMED_ATTRS = VT_LIST_BASE + VT_NAMED_ATTRS,
    VT_LIST_DATA_TYPE = VT_LIST_BASE + VT_DATA_TYPE,
  };

  template <class T>
  struct IsAttrTypeEnable {
    using DT = typename std::remove_cv<T>::type;

    static bool const VALUE = std::is_same<INT, DT>::value || std::is_same<FLOAT, DT>::value ||
                              std::is_same<BOOL, DT>::value || std::is_same<STR, DT>::value ||
                              std::is_same<GRAPH, DT>::value || std::is_same<TENSOR, DT>::value ||
                              std::is_same<TENSOR_DESC, DT>::value || std::is_same<BYTES, DT>::value ||
                              std::is_same<NAMED_ATTRS, DT>::value || std::is_same<DATA_TYPE, DT>::value;

    // Not has list type of NamedAttrs
    static bool const LIST_VALUE = std::is_same<LIST_INT, DT>::value || std::is_same<LIST_FLOAT, DT>::value ||
                                   std::is_same<LIST_BOOL, DT>::value || std::is_same<LIST_STR, DT>::value ||
                                   std::is_same<LIST_GRAPH, DT>::value || std::is_same<LIST_TENSOR, DT>::value ||
                                   std::is_same<LIST_TENSOR_DESC, DT>::value || std::is_same<LIST_BYTES, DT>::value ||
                                   std::is_same<LIST_NAMED_ATTRS, DT>::value ||
                                   std::is_same<LIST_LIST_INT, DT>::value || std::is_same<LIST_DATA_TYPE, DT>::value;
  };

  template <typename vector_type>
  // To cols
  using enable_if_vector_type_valid_t = typename std::enable_if<IsAttrTypeEnable<vector_type>::LIST_VALUE,
                                                                int>::type;

  template <typename one_type>
  using enable_if_one_type_valid_t = typename std::enable_if<IsAttrTypeEnable<one_type>::VALUE, int>::type;

  template <typename val_type>
  using enable_if_type_valid_t =
      typename std::enable_if<IsAttrTypeEnable<val_type>::VALUE || IsAttrTypeEnable<val_type>::LIST_VALUE, int>::type;

  template <typename seriliable_type>
  using enable_if_seriliable_type_valid_t = typename seriliable_type::__ge_serializable;

  GeAttrValue();
  ~GeAttrValue() = default;
  // SetValue, Set initializer_list
  template <typename T, typename DT, enable_if_vector_type_valid_t<T> = 0>
  graphStatus SetValue(std::initializer_list<DT> &&val) {
    T vectorVal;
    for (auto &item : val) {
      vectorVal.push_back(item);
    }
    return SetValue(vectorVal);
  }

  // SetValue, Set vector
  template <typename T, typename DT, enable_if_vector_type_valid_t<T> = 0>
  graphStatus SetValue(const std::vector<DT> &val) {
    T vectorVal;
    for (auto item : val) {
      vectorVal.push_back(item);
    }
    return SetValue(vectorVal);
  }

  // SetValue, not list type
  template <typename T, typename DT, enable_if_one_type_valid_t<T> = 0>
  graphStatus SetValue(DT &&val) {
    return SetValue(T(std::forward<DT>(val)));
  }

  // GE_SERIALIZABLE
  template <typename T, enable_if_seriliable_type_valid_t<T> = 0>
  graphStatus SetValue(const T &t) {
    return t.Save(*this);
  }

  template <typename T, enable_if_seriliable_type_valid_t<T> = 0>
  graphStatus SetValue(const vector<T> &t) {
    vector<NamedAttrs> attrs;
    for (auto &item : t) {
      GeAttrValue val;
      item.Save(val);
      NamedAttrs attrsItem;
      (void)val.GetValue<NamedAttrs>(attrsItem);
      attrs.push_back(attrsItem);
    }
    return SetValue(attrs);
  }

  // GetValue, list value
  template <typename T, typename DT, enable_if_vector_type_valid_t<T> = 0,
            typename std::enable_if<!std::is_same<DT, GeTensorPtr>::value, int>::type = 0>
  graphStatus GetValue(std::vector<DT> &val) const {
    T valGet;
    val.clear();
    auto status = GetValue(valGet);
    if (status != GRAPH_SUCCESS) {
      return status;
    }
    for (auto item : valGet) {
      val.push_back(item);
    }
    return GRAPH_SUCCESS;
  }

  // GetValue, not list type
  template <typename T, typename DT, enable_if_one_type_valid_t<T> = 0,
            typename std::enable_if<!std::is_same<DT, GeTensorPtr>::value, int>::type = 0>
  graphStatus GetValue(DT &val) const {
    T valGet;
    auto status = GetValue(valGet);
    if (status != GRAPH_SUCCESS) {
      return status;
    }
    val = DT(valGet);
    return GRAPH_SUCCESS;
  }

  // GE_SERIALIZABLE
  template <typename T, enable_if_seriliable_type_valid_t<T> = 0>
  graphStatus GetValue(T &t) {
    return t.Load(*this);
  }

  template <typename T, enable_if_seriliable_type_valid_t<T> = 0>
  graphStatus GetValue(vector<T> &t) {
    graphStatus status;
    t.clear();
    vector<NamedAttrs> attrs;
    status = this->GetValue(attrs);
    if (status != GRAPH_SUCCESS) {
      return status;
    }
    for (auto &attr : attrs) {
      T item;
      GeAttrValue val;
      (void)val.SetValue(attr);
      status = item.Load(val);
      if (status != GRAPH_SUCCESS) {
        return status;
      }
      t.push_back(item);
    }
    return GRAPH_SUCCESS;
  }

  template <typename T, typename DT, enable_if_type_valid_t<T> = 0>
  static GeAttrValue CreateFrom(DT &&val) {
    GeAttrValue valRet;
    (void)valRet.SetValue<T>(std::forward<DT>(val));
    return valRet;
  }

  template <typename T, typename DT, enable_if_vector_type_valid_t<T> = 0>
  static GeAttrValue CreateFrom(std::initializer_list<DT> &&val) {
    GeAttrValue valRet;
    (void)valRet.SetValue<T>(std::move(val));
    return valRet;
  }

  template <typename T, enable_if_seriliable_type_valid_t<T> = 0>
  static GeAttrValue CreateFrom(const T &val) {
    GeAttrValue valRet;
    (void)valRet.SetValue(val);
    return valRet;
  }

  template <typename T, enable_if_seriliable_type_valid_t<T> = 0>
  static GeAttrValue CreateFrom(const vector<T> &val) {
    GeAttrValue valRet;
    (void)valRet.SetValue(val);
    return valRet;
  }

  ValueType GetValueType() const;

  bool IsEmpty() const;

  GeAttrValue Copy() const;

  // For map key
  bool operator==(const GeAttrValue &other) const { return value_ == other.value_; }

  graphStatus MutableTensor(GeTensorPtr &tensor);
  graphStatus MutableListTensor(vector<GeTensorPtr> &list_tensor);

 private:
#define VALUE_SET_GET_DEC(DT)          \
  graphStatus SetValue(const DT &val); \
  graphStatus GetValue(DT &val) const;
  VALUE_SET_GET_DEC(GeAttrValue::STR)
  VALUE_SET_GET_DEC(GeAttrValue::INT)
  VALUE_SET_GET_DEC(GeAttrValue::FLOAT)
  VALUE_SET_GET_DEC(GeAttrValue::BOOL)
  VALUE_SET_GET_DEC(GeTensorDesc)
  VALUE_SET_GET_DEC(GeAttrValue::TENSOR)
  VALUE_SET_GET_DEC(GeAttrValue::GRAPH)
  VALUE_SET_GET_DEC(BYTES)
  VALUE_SET_GET_DEC(NamedAttrs)
  VALUE_SET_GET_DEC(ge::DataType)  // lint !e665
  VALUE_SET_GET_DEC(vector<GeAttrValue::STR>)
  VALUE_SET_GET_DEC(vector<GeAttrValue::INT>)
  VALUE_SET_GET_DEC(vector<GeAttrValue::FLOAT>)
  VALUE_SET_GET_DEC(vector<GeAttrValue::BOOL>)
  VALUE_SET_GET_DEC(vector<GeTensorDesc>)
  VALUE_SET_GET_DEC(vector<GeAttrValue::TENSOR>)
  VALUE_SET_GET_DEC(vector<GeAttrValue::GRAPH>)
  VALUE_SET_GET_DEC(vector<GeAttrValue::BYTES>)
  VALUE_SET_GET_DEC(vector<NamedAttrs>)
  VALUE_SET_GET_DEC(vector<vector<int64_t>>)  //lint !e665
  VALUE_SET_GET_DEC(vector<ge::DataType>)     //lint !e665
#undef VALUE_SET_GET_DEC

  GeIrProtoHelper<proto::AttrDef> value_;
  GeAttrValue(const ProtoMsgOwner &proto_owner, ge::proto::AttrDef *val);

  friend class AttrHolder;
  friend class ModelSerializeImp;
  friend class OnnxUtils;
};

class AttrValueImpl {
 public:
  AttrValueImpl() = default;
  ~AttrValueImpl() = default;

  GeAttrValue geAttrValue_;
};
}  // namespace ge
#endif  // INC_GRAPH_GE_ATTR_VALUE_H_
