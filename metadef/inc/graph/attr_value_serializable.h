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

#ifndef INC_GRAPH_ATTR_VALUE_SERIALIZABLE_H_
#define INC_GRAPH_ATTR_VALUE_SERIALIZABLE_H_

#include <string>
#include <vector>
#include "graph/ge_attr_value.h"
#include "graph/compiler_options.h"

namespace ge {

class GeAttrValue;
class _GeSerializable {
 public:
  template <typename T>
  struct ge_serializable_int64_t_support_type {
    using DT = typename std::remove_cv<T>::type;
    static const bool value = std::is_same<DT, uint64_t>::value  // by cast
                              || std::is_same<DT, int32_t>::value || std::is_same<DT, uint32_t>::value ||
                              std::is_same<DT, int16_t>::value || std::is_same<DT, uint16_t>::value ||
                              std::is_same<DT, int8_t>::value || std::is_same<DT, uint8_t>::value;
  };

  template <typename T, typename T::__ge_serializable = 0>
  static GeAttrValue SaveItemAsAttrValue(const T &t) {
    return GeAttrValue::CreateFrom(t);
  }

  template <typename T, typename T::__ge_serializable = 0>
  static GeAttrValue SaveItemAsAttrValue(const vector<T> &t) {
    return GeAttrValue::CreateFrom(t);
  }

  template <typename T, GeAttrValue::enable_if_type_valid_t<T> = 0, typename DT = typename std::remove_cv<T>::type>
  static GeAttrValue SaveItemAsAttrValue(const T &t) {
    return GeAttrValue::CreateFrom<DT>(t);
  }
  // int64_t support type
  template <typename T, typename std::enable_if<ge_serializable_int64_t_support_type<T>::value, int>::type = 0>
  static GeAttrValue SaveItemAsAttrValue(const T &t) {
    return GeAttrValue::CreateFrom<GeAttrValue::INT>(t);
  }
  // vector int64_t support type
  template <typename T, typename std::enable_if<ge_serializable_int64_t_support_type<T>::value, int>::type = 0>
  static GeAttrValue SaveItemAsAttrValue(const vector<T> &t) {
    return GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>(t);
  }

  template <typename T, typename T::__ge_serializable = 0>
  static graphStatus LoadItemFromAttrValue(T &t, GeAttrValue &attrVal) {
    return attrVal.GetValue(t);
  }

  template <typename T, typename T::__ge_serializable = 0>
  static graphStatus LoadItemFromAttrValue(vector<T> &t, GeAttrValue &attrVal) {
    return attrVal.GetValue(t);
  }

  template <typename T, GeAttrValue::enable_if_type_valid_t<T> = 0, typename DT = typename std::remove_cv<T>::type>
  static graphStatus LoadItemFromAttrValue(T &t, GeAttrValue &attrVal) {
    return attrVal.GetValue<DT>(t);
  }

  template <typename T, typename std::enable_if<ge_serializable_int64_t_support_type<T>::value, int>::type = 0>
  static graphStatus LoadItemFromAttrValue(T &t, GeAttrValue &attrVal) {
    return attrVal.GetValue<GeAttrValue::INT>(t);
  }

  template <typename T, typename std::enable_if<ge_serializable_int64_t_support_type<T>::value, int>::type = 0>
  static graphStatus LoadItemFromAttrValue(vector<T> &t, GeAttrValue &attrVal) {
    return attrVal.GetValue<GeAttrValue::LIST_INT>(t);
  }

  template <class T, class... Args>
  static void SaveItem(GeAttrValue::NAMED_ATTRS &namedAttrs, string itemName, T &item, Args &... args) {
    GeAttrValue itemVal = SaveItemAsAttrValue(item);
    (void)namedAttrs.SetAttr(itemName, itemVal);
    SaveItem(namedAttrs, args...);
  }

  static void SaveItem(GeAttrValue::NAMED_ATTRS &namedAttrs METADEF_ATTRIBUTE_UNUSED) {}

  template <class T, class... Args>
  static graphStatus LoadItem(GeAttrValue::NAMED_ATTRS &namedAttrs, string itemName, T &item, Args &... args) {
    auto itemVal = namedAttrs.GetItem(itemName);
    auto status = LoadItemFromAttrValue(item, itemVal);
    if (status != GRAPH_SUCCESS) {
      return status;
    }
    return LoadItem(namedAttrs, args...);
  }

  static graphStatus LoadItem(GeAttrValue::NAMED_ATTRS &namedAttrs METADEF_ATTRIBUTE_UNUSED) { return GRAPH_SUCCESS; }
};

#define _GE_FI(a) #a, a
#define _GE_MAP_FIELDS1(a1) _GE_FI(a1)
#define _GE_MAP_FIELDS2(a1, a2) _GE_FI(a1), _GE_FI(a2)
#define _GE_MAP_FIELDS3(a1, a2, a3) _GE_FI(a1), _GE_FI(a2), _GE_FI(a3)
#define _GE_MAP_FIELDS4(a1, a2, a3, a4) _GE_FI(a1), _GE_FI(a2), _GE_FI(a3), _GE_FI(a4)
#define _GE_MAP_FIELDS5(a1, a2, a3, a4, a5) _GE_FI(a1), _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5)
#define _GE_MAP_FIELDS6(a1, a2, a3, a4, a5, a6) _GE_FI(a1), _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5), _GE_FI(a6)
#define _GE_MAP_FIELDS7(a1, a2, a3, a4, a5, a6, a7) \
  _GE_FI(a1)                                        \
  , _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5), _GE_FI(a6), _GE_FI(a7)
#define _GE_MAP_FIELDS8(a1, a2, a3, a4, a5, a6, a7, a8) \
  _GE_FI(a1)                                            \
  , _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5), _GE_FI(a6), _GE_FI(a7), _GE_FI(a8)
#define _GE_MAP_FIELDS9(a1, a2, a3, a4, a5, a6, a7, a8, a9) \
  _GE_FI(a1)                                                \
  , _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5), _GE_FI(a6), _GE_FI(a7), _GE_FI(a8), _GE_FI(a9)
#define _GE_MAP_FIELDS10(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10) \
  _GE_FI(a1)                                                      \
  , _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5), _GE_FI(a6), _GE_FI(a7), _GE_FI(a8), _GE_FI(a9), _GE_FI(a10)
#define _GE_MAP_FIELDS11(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)                                           \
  _GE_FI(a1)                                                                                                     \
  , _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5), _GE_FI(a6), _GE_FI(a7), _GE_FI(a8), _GE_FI(a9), _GE_FI(a10), \
      _GE_FI(a11)
#define _GE_MAP_FIELDS12(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)                                      \
  _GE_FI(a1)                                                                                                     \
  , _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5), _GE_FI(a6), _GE_FI(a7), _GE_FI(a8), _GE_FI(a9), _GE_FI(a10), \
      _GE_FI(a11), _GE_FI(a12)
#define _GE_MAP_FIELDS13(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)                                 \
  _GE_FI(a1)                                                                                                     \
  , _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5), _GE_FI(a6), _GE_FI(a7), _GE_FI(a8), _GE_FI(a9), _GE_FI(a10), \
      _GE_FI(a11), _GE_FI(a12), _GE_FI(a13)
#define _GE_MAP_FIELDS14(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)                            \
  _GE_FI(a1)                                                                                                     \
  , _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5), _GE_FI(a6), _GE_FI(a7), _GE_FI(a8), _GE_FI(a9), _GE_FI(a10), \
      _GE_FI(a11), _GE_FI(a12), _GE_FI(a13), _GE_FI(a14)
#define _GE_MAP_FIELDS15(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)                       \
  _GE_FI(a1)                                                                                                     \
  , _GE_FI(a2), _GE_FI(a3), _GE_FI(a4), _GE_FI(a5), _GE_FI(a6), _GE_FI(a7), _GE_FI(a8), _GE_FI(a9), _GE_FI(a10), \
      _GE_FI(a11), _GE_FI(a12), _GE_FI(a13), _GE_FI(a14), _GE_FI(a15)

#define _GE_PRIVATE_ARGS_GLUE(x, y) x y

#define _GE_PRIVATE_MACRO_VAR_ARGS_IMPL_COUNT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, \
                                              ...)                                                                 \
  N
#define _GE_PRIVATE_MACRO_VAR_ARGS_IMPL(args) _GE_PRIVATE_MACRO_VAR_ARGS_IMPL_COUNT args
#define _GE_COUNT_MACRO_VAR_ARGS(...) \
  _GE_PRIVATE_MACRO_VAR_ARGS_IMPL((__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

#define _GE_PRIVATE_MACRO_CHOOSE_HELPER2(M, count) M##count
#define _GE_PRIVATE_MACRO_CHOOSE_HELPER1(M, count) _GE_PRIVATE_MACRO_CHOOSE_HELPER2(M, count)
#define _GE_PRIVATE_MACRO_CHOOSE_HELPER(M, count) _GE_PRIVATE_MACRO_CHOOSE_HELPER1(M, count)

#define _GE_INVOKE_VAR_MACRO(...)                                                                               \
  _GE_PRIVATE_ARGS_GLUE(_GE_PRIVATE_MACRO_CHOOSE_HELPER(_GE_MAP_FIELDS, _GE_COUNT_MACRO_VAR_ARGS(__VA_ARGS__)), \
                        (__VA_ARGS__))

#define GE_SERIALIZABLE(...)                                                          \
 public:                                                                              \
  friend class ge::GeAttrValue;                                                       \
  using __ge_serializable = int;                                                      \
                                                                                      \
 private:                                                                             \
  ge::graphStatus Save(GeAttrValue &ar) const {                                       \
    GeAttrValue::NAMED_ATTRS named_attrs;                                             \
    _GeSerializable::SaveItem(named_attrs, _GE_INVOKE_VAR_MACRO(__VA_ARGS__));        \
    return ar.SetValue<GeAttrValue::NAMED_ATTRS>(named_attrs);                        \
  }                                                                                   \
  ge::graphStatus Load(const GeAttrValue &ar) {                                       \
    GeAttrValue::NAMED_ATTRS named_attrs;                                             \
    ge::graphStatus status = ar.GetValue<GeAttrValue::NAMED_ATTRS>(named_attrs);      \
    if (status != GRAPH_SUCCESS) {                                                    \
      return status;                                                                  \
    }                                                                                 \
    return _GeSerializable::LoadItem(named_attrs, _GE_INVOKE_VAR_MACRO(__VA_ARGS__)); \
  }

// end NamedAttrs Helper: GE_SERIALIZABLE
}  // namespace ge
#endif  // INC_GRAPH_ATTR_VALUE_SERIALIZABLE_H_
