/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_FRAMEWORK_COMMON_HELPER_MODEL_SAVE_HELPER_FACTORY_H_
#define INC_FRAMEWORK_COMMON_HELPER_MODEL_SAVE_HELPER_FACTORY_H_

#include <functional>
#include <memory>

#include "framework/common/ge_types.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
class ModelSaveHelper;
using ModelSaveHelperPtr = std::shared_ptr<ModelSaveHelper>;

class ModelSaveHelperFactory {
 public:
  using ModelSaveHelperCreatorFun = std::function<ModelSaveHelperPtr(void)>;

  static ModelSaveHelperFactory &Instance() {
    static ModelSaveHelperFactory instance;
    return instance;
  }

  ModelSaveHelperPtr Create(const OfflineModelFormat type) {
    const auto iter = creator_map_.find(type);
    if (iter == creator_map_.end()) {
      return nullptr;
    }
    return iter->second();
  }

  // ModelSaverHelper registerar
  class Registerar {
   public:
    Registerar(const OfflineModelFormat type, const ModelSaveHelperCreatorFun &func) {
      ModelSaveHelperFactory::Instance().RegisterCreator(type, func);
    }

    ~Registerar() = default;
  };

 private:
  ModelSaveHelperFactory() = default;
  ~ModelSaveHelperFactory() = default;

  // register creator, this function will can in constructor
  void RegisterCreator(const OfflineModelFormat type, const ModelSaveHelperCreatorFun &func) {
    const auto iter = creator_map_.find(type);
    if (iter != creator_map_.end()) {
      return;
    }

    creator_map_[type] = func;
  }

  std::map<OfflineModelFormat, ModelSaveHelperCreatorFun> creator_map_;
};
}  // namespace ge

#define REGISTER_MODEL_SAVE_HELPER(type, clazz)                                                                      \
  namespace {                                                                                                        \
  ModelSaveHelperPtr Creator_##type##_Model_Save_Helper() {                                                          \
    try {                                                                                                            \
      return std::make_shared<clazz>();                                                                              \
    } catch (...) {                                                                                                  \
      return nullptr;                                                                                                \
    }                                                                                                                \
  }                                                                                                                  \
  ModelSaveHelperFactory::Registerar g_##type##_Model_Save_Helper_Creator(type, Creator_##type##_Model_Save_Helper); \
  }  // namespace

#endif  // INC_FRAMEWORK_COMMON_HELPER_MODEL_SAVE_HELPER_H_
