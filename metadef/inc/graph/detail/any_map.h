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

#ifndef INC_GRAPH_DETAIL_ANY_MAP_H_
#define INC_GRAPH_DETAIL_ANY_MAP_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "graph/compiler_options.h"

namespace ge {
using std::shared_ptr;
using std::string;

class TypeID {
 public:
  template <class T>
  static TypeID Of() {
    return TypeID(METADEF_FUNCTION_IDENTIFIER);
  }

  ~TypeID() = default;

  bool operator==(const TypeID &__arg) const { return type_ == __arg.type_; }

 private:
  explicit TypeID(string type) : type_(std::move(type)) {}  // lint !e30 !e32

  string type_;
};

class AnyMap {
 public:
  template <class DT>
  bool Set(const string &name, const DT &val);

  template <class T>
  bool Get(const string &name, T &retValue) const;

  bool Has(const string &name) const { return anyValues_.find(name) != anyValues_.end(); }

  void Swap(AnyMap &other) {
    anyValues_.swap(other.anyValues_);
  }

 private:
  class Placeholder {
   public:
    virtual ~Placeholder() = default;

    virtual const TypeID &GetTypeInfo() const = 0;
  };

  template <typename VT>
  class Holder : public Placeholder {
   public:
    explicit Holder(const VT &value) : value_(value) {}

    ~Holder() override = default;

    const TypeID &GetTypeInfo() const override {
      static const TypeID typeId = TypeID::Of<VT>();
      return typeId;
    }

    const VT value_;
  };

  std::map<string, shared_ptr<Placeholder>> anyValues_;
};

template <class DT>
bool AnyMap::Set(const string &name, const DT &val) {
  auto it = anyValues_.find(name);

  std::shared_ptr<Holder<DT>> tmp;
  try {
    tmp = std::make_shared<Holder<DT>>(val);
  } catch (std::bad_alloc &e) {
    tmp = nullptr;
  } catch (...) {
    tmp = nullptr;
  }

  if (it == anyValues_.end()) {
    (void)anyValues_.emplace(name, tmp);
  } else {
    if (it->second && it->second->GetTypeInfo() == TypeID::Of<DT>()) {
      it->second = tmp;
    } else {
      return false;
    }
  }
  return true;
}

template <class T>
bool AnyMap::Get(const string &name, T &retValue) const {
  auto it = anyValues_.find(name);
  if (it != anyValues_.end() && it->second && it->second->GetTypeInfo() == TypeID::Of<T>()) {
    auto retPtr = std::static_pointer_cast<Holder<T>>(it->second);
    retValue = retPtr->value_;
    return true;
  }
  return false;
}
}  // namespace ge
#endif  // INC_GRAPH_DETAIL_ANY_MAP_H_
