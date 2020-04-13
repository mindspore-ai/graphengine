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

#include "graph/load/graph_loader.h"

#include <string>
#include <vector>

#include "common/helper/model_helper.h"
#include "common/util.h"
#include "graph/ge_context.h"
#include "graph/load/new_model_manager/davinci_model_parser.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "omm/csa_interact.h"
#include "runtime/dev.h"

namespace ge {
GraphLoader::GraphLoader() = default;

GraphLoader::~GraphLoader() = default;

Status GraphLoader::LoadGraph(const std::shared_ptr<ge::GeModel> &ge_model_ptr,
                              const std::shared_ptr<GraphModelListener> &model_listener, ModelIdInfo &model_id_info) {
  if (ge_model_ptr == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[LoadGraph] GE load graph model_ptr is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  if (model_listener == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[LoadGraph] GE load graph model_listener is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  std::shared_ptr<ge::Model> model_ptr;
  if (ModelHelper::TransGeModelToModel(ge_model_ptr, model_ptr) != SUCCESS) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[LoadGraph] GE load graph ge_model_ptr trans to ModelPtr failed.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  GELOGI("[LoadGraph] GE load graph via new ome begin.");
  Status ret = LoadModelOnline(model_id_info.model_id, model_ptr, model_listener);
  if (ret != SUCCESS) {
    GELOGE(ret, "[LoadGraph] GE load graph  LoadGraph() return fail. err: %u", ret);
    return ret;
  }
  GELOGI("[LoadGraph] GE load graph success. modelId: %u", model_id_info.model_id);
  return ret;
}

Status GraphLoader::LoadGraphAsync(const std::shared_ptr<ge::GeModel> &ge_model_ptr,
                                   const std::shared_ptr<RunAsyncListener> &model_async_listener,
                                   ModelIdInfo &model_id_info) {
  if (ge_model_ptr == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[LoadGraphAsync] GE load graph model_ptr is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  if (model_async_listener == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[LoadGraphAsync] GE load graph model_listener is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  std::shared_ptr<ge::Model> model_ptr;
  if (ModelHelper::TransGeModelToModel(ge_model_ptr, model_ptr) != SUCCESS) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[LoadGraph] GE load graph ge_model_ptr trans to ModelPtr failed.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  GELOGI("[LoadGraphAsync] GE load graph begin.");
  Status ret = LoadModelOnline(model_id_info.model_id, model_ptr, model_async_listener);
  if (ret != SUCCESS) {
    GELOGE(ret, "[LoadGraphAsync] GE load graph  LoadGraphAsync() return fail. err: %u", ret);
    return ret;
  }

  GELOGI("[LoadGraphAsync] GE load graph success. modelId: %u", model_id_info.model_id);
  return ret;
}

Status GraphLoader::UnloadModel(uint32_t model_id) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  GELOGI("UnLoad model begin, model_id:%u.", model_id);

  Status ret = model_manager->Stop(model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "UnloadModel: Stop failed.");
  }

  ret = model_manager->Unload(model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "UnloadModel: Unload failed.");
    CsaInteract::GetInstance().WriteErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_UNLOAD);
    return ret;
  }
  GELOGI("UnLoad model success, model_id:%u.", model_id);
  return SUCCESS;
}

Status GraphLoader::LoadModelOnline(uint32_t &model_id, std::shared_ptr<ge::Model> &model,
                                    const std::shared_ptr<ModelListener> &listener) {
  rtError_t rt_ret = rtSetDevice(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    CsaInteract::GetInstance().WriteErrorCode(rt_ret, ERROR_MODULE_RUNTIME, JOBSUBSTATE_GRAPH_LOAD);
    return RT_FAILED;
  }

  try {
    GELOGI("Load begin, model_id:%u.", model_id);
    auto model_manager = ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret = model_manager->LoadModelOnline(model_id, model, listener);
    if (ret != SUCCESS) {
      GELOGE(ret, "LoadModel: Load failed. ret = %u", ret);
      CsaInteract::GetInstance().WriteErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_LOAD);

      rt_ret = rtDeviceReset(GetContext().DeviceId());
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      }
      return ret;
    }

    ret = model_manager->Start(model_id);
    if (ret != SUCCESS) {
      if (model_manager->Unload(model_id) != SUCCESS) {
        GELOGE(ret, "LoadModel: Unload failed while trying to unload after a failed start.");
      }

      rt_ret = rtDeviceReset(GetContext().DeviceId());
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      }

      GELOGE(ret, "LoadModel: Start failed.");
      CsaInteract::GetInstance().WriteErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
      return ret;
    }

    GELOGI("Load model success, model_id:%u.", model_id);
  } catch (std::bad_alloc &) {
    rt_ret = rtDeviceReset(GetContext().DeviceId());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    }

    GELOGE(MEMALLOC_FAILED, "Load model failed, bad memory allocation occur !");
    CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_LOAD);
    return MEMALLOC_FAILED;
  } catch (...) {
    rt_ret = rtDeviceReset(GetContext().DeviceId());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    }

    GELOGE(FAILED, "Load failed, some exceptions occur !");
    CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_LOAD);
    return FAILED;
  }

  rt_ret = rtDeviceReset(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  return SUCCESS;
}

Status GraphLoader::GetMaxUsedMemory(uint32_t model_id, uint64_t &max_size) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetMaxUsedMemory(model_id, max_size);
  if (ret != SUCCESS) {
    GELOGE(ret, "GetMaxUsedMemory: GetMaxUsedMemory failed.");
    return ret;
  }
  return SUCCESS;
}

Status GraphLoader::LoadDataFromFile(const std::string &path, const std::string &key_path, int32_t priority,
                                     ModelData &model_data) {
  Status ret;
  try {
    if (!CheckInputPathValid(path)) {
      GELOGE(PARAM_INVALID, "model path is invalid: %s", path.c_str());
      return PARAM_INVALID;
    }

    GELOGI("Load model begin, model path is: %s", path.c_str());
    if (!key_path.empty() && !CheckInputPathValid(key_path)) {
      GELOGE(PARAM_INVALID, "decrypt_key path is invalid: %s", key_path.c_str());
      return PARAM_INVALID;
    }

    ret = DavinciModelParser::LoadFromFile(path.c_str(), key_path.c_str(), priority, model_data);
    if (ret != SUCCESS) {
      GELOGE(ret, "LoadModelFromFile: Load failed. ret = %u", ret);
      return ret;
    }

    return SUCCESS;
  } catch (std::bad_alloc &) {
    GELOGE(MEMALLOC_FAILED, "Load model from file failed, bad memory allocation");
    ret = MEMALLOC_FAILED;
  } catch (...) {
    GELOGE(FAILED, "Load model from file failed with exception");
    ret = FAILED;
  }

  if (model_data.model_data != nullptr) {
    delete[] static_cast<char *>(model_data.model_data);
    model_data.model_data = nullptr;
  }
  return ret;
}

Status GraphLoader::LoadModelFromFile(const std::string &path, const std::string &key_path, int32_t priority,
                                      const std::shared_ptr<ModelListener> &listener, uint32_t &model_id) {
  Status ret;
  ModelData model_data;

  try {
    ret = LoadDataFromFile(path, key_path, priority, model_data);
    if (ret != SUCCESS) {
      GELOGE(ret, "LoadModelFromFile: Load failed. ret = %u", ret);
      if (model_data.model_data != nullptr) {
        delete[] static_cast<char *>(model_data.model_data);
        model_data.model_data = nullptr;
      }
      return ret;
    }

    ret = LoadModel(model_data, listener, model_id);
    if (ret != SUCCESS) {
      GELOGE(ret, "LoadModel: Load failed. ret = %u", ret);
      if (model_data.model_data != nullptr) {
        delete[] static_cast<char *>(model_data.model_data);
        model_data.model_data = nullptr;
      }
    }
  } catch (std::bad_alloc &) {
    GELOGE(MEMALLOC_FAILED, "Load model from file failed, bad memory allocation");
    ret = MEMALLOC_FAILED;
  } catch (...) {
    GELOGE(FAILED, "Load model from file failed with exception");
    ret = FAILED;
  }

  if (model_data.model_data != nullptr) {
    delete[] static_cast<char *>(model_data.model_data);
    model_data.model_data = nullptr;
  }

  return ret;
}

Status GraphLoader::LoadModel(const ModelData &model_data, const std::shared_ptr<ModelListener> &listener,
                              uint32_t &model_id) {
  try {
    GELOGI("Load model begin, model_id:%u.", model_id);

    // For GeOp, Open Device 0 here.
    GE_CHK_RT_RET(rtSetDevice(0));
    auto model_manager = ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret = model_manager->LoadModelOffline(model_id, model_data, listener);
    if (ret != SUCCESS) {
      GE_CHK_RT(rtDeviceReset(0));
      GELOGE(ret, "LoadModel: Load failed.");
      return ret;
    }
    ret = model_manager->Start(model_id);
    if (ret != SUCCESS) {
      if (model_manager->Unload(model_id) != SUCCESS) {
        GELOGE(FAILED, "LoadModel: Unload failed while trying to unload after a failed start.");
      }
      GELOGE(ret, "LoadModel: Start failed.");
      return ret;
    }
    GELOGI("LoadModel: Start model success, model_id:%u.", model_id);
  } catch (std::bad_alloc &) {
    GELOGE(MEMALLOC_FAILED, "Load model failed, bad memory allocation occur !");
    return MEMALLOC_FAILED;
  } catch (...) {
    GELOGE(FAILED, "Load model failed, some exceptions occur !");
    return FAILED;
  }

  return SUCCESS;
}

Status GraphLoader::CommandHandle(const Command &command) {
  try {
    auto model_manager = ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret = model_manager->HandleCommand(command);
    if (ret != SUCCESS) {
      GELOGE(ret, "CommandHandle: Command Handle failed.");

      return ret;
    }
  } catch (std::bad_alloc &) {
    GELOGE(MEMALLOC_FAILED, "Command handle failed, bad memory allocation occur !");

    return MEMALLOC_FAILED;
  } catch (...) {
    GELOGE(FAILED, "Command handle failed, some exceptions occur !");

    return FAILED;
  }

  return SUCCESS;
}

Status GraphLoader::LoadModelFromData(uint32_t &model_id, const ModelData &model_data, void *dev_ptr, size_t memsize,
                                      void *weight_ptr, size_t weightsize) {
  try {
    GELOGI("Load model begin, model_id:%u.", model_id);

    // For ACL, Open Device from App.
    auto model_manager = ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret =
      model_manager->LoadModelOffline(model_id, model_data, nullptr, dev_ptr, memsize, weight_ptr, weightsize);
    if (ret != SUCCESS) {
      GELOGE(ret, "Load model failed, model_id:%u.", model_id);
      return ret;
    }

    GELOGI("Load model success, model_id:%u.", model_id);
  } catch (std::bad_alloc &) {
    GELOGE(MEMALLOC_FAILED, "Load model failed, bad memory allocation occur !");
    return MEMALLOC_FAILED;
  } catch (...) {
    GELOGE(FAILED, "Load model failed, some exceptions occur !");
    return FAILED;
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Load task list from ModelData with queue.
/// @param [out] model_id: model id allocate from manager.
/// @param [in] model_data: Model data load from offline model.
/// @param [in] input_queue_ids: input queue ids create from user.
/// @param [in] output_queue_ids: input queue ids create from user.
/// @return: 0 for success / others for fail
///
Status GraphLoader::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                                   const std::vector<uint32_t> &input_queue_ids,
                                   const std::vector<uint32_t> &output_queue_ids) {
  GELOGI("Load model with queue begin, model_id:%u.", model_id);

  // For ACL, Open Device from App.
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids);
  if (ret != SUCCESS) {
    GELOGE(ret, "Load model with queue failed, model_id:%u.", model_id);
    return ret;
  }

  GELOGI("Load model with queue success, model_id:%u.", model_id);
  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief  execute model
/// @param [in] model_id  model id
/// @param [in] stream   stream to execute model on
/// @param [in] async_mode  is asynchronize mode.
/// @param [in] input_data  model input data
/// @param [out] output_data  model output data
///
Status GraphLoader::ExecuteModel(uint32_t model_id, rtStream_t stream, bool async_mode, const InputData &input_data,
                                 OutputData &output_data) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->ExecuteModel(model_id, stream, async_mode, input_data, output_data);
  if (ret != SUCCESS) {
    GELOGE(ret, "Execute model failed, model_id:%u.", model_id);
    return ret;
  }

  GELOGI("Execute model success, model_id:%u.", model_id);
  return SUCCESS;
}

Status GraphLoader::GetMemoryInfo(int64_t &free) {
  rtError_t rt_ret = rtSetDevice(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    CsaInteract::GetInstance().WriteErrorCode(rt_ret, ERROR_MODULE_RUNTIME, JOBSUBSTATE_GRAPH_LOAD);
    return RT_FAILED;
  }
  size_t total_mem = 0;
  size_t free_mem = 0;
  rt_ret = rtMemGetInfo(&free_mem, &total_mem);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  rt_ret = rtDeviceReset(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  // Add small page memory size
  free = static_cast<int64_t>(free_mem + VarManager::Instance(0)->GetUseMaxMemorySize() - total_mem);
  GELOGI("GetMemoryInfo free[%zu], total[%zu], return free[%ld]", free_mem, total_mem, free);
  return SUCCESS;
}

Status GraphLoader::DestroyAicpuKernel(uint64_t session_id, uint32_t model_id) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->DestroyAicpuKernel(session_id, model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "Destroy aicpu kernel failed.");
    return ret;
  }
  return SUCCESS;
}
}  // namespace ge
