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

#include "graph/load/new_model_manager/davinci_model_parser.h"

#include <fstream>
#include <memory>
#include <vector>
#include "securec.h"

#include "common/debug/log.h"
#include "graph/load/new_model_manager/davinci_model.h"


namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelInfoParser(const ModelData &model, ModelInfo &model_info) {
  GE_CHK_RT_RET(rtSetDevice(0));
  try {
    uint32_t model_len = 0;
    uint8_t *model_data = nullptr;

    Status ret = DavinciModelParser::ParseModelContent(model, model_data, model_len);

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, GE_CHK_RT(rtDeviceReset(0)); return ret, "Parse model failed");

    auto *file_header = reinterpret_cast<ModelFileHeader *>(model.model_data);

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(file_header == nullptr, GE_CHK_RT(rtDeviceReset(0));
                                     return PARAM_INVALID, "file_header is null.");

    model_info.version = file_header->version;
    model_info.is_encrypt = false;
    GE_IF_BOOL_EXEC(ENCRYPTED == file_header->is_encrypt, model_info.is_encrypt = true);

    std::shared_ptr<DavinciModel> davinci_model =
        std::shared_ptr<DavinciModel>(new (std::nothrow) DavinciModel(model.priority, nullptr));

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(davinci_model == nullptr, GE_CHK_RT(rtDeviceReset(0));
                                     return PARAM_INVALID, "davinci_model is null.");

    GE_MAKE_GUARD(davinci_model, [&] {
      davinci_model = nullptr;
    });

    ModelHelper model_helper;
    ret = model_helper.LoadModel(model);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((ret != SUCCESS), GE_CHK_RT(rtDeviceReset(0)); return FAILED, "load model failed");

    ret = davinci_model->Assign(model_helper.GetGeModel());
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, GE_CHK_RT(rtDeviceReset(0));
                                     return ret, "Parse davinci model data failed");

    ret = davinci_model->Init();

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, GE_CHK_RT(rtDeviceReset(0));
                                     return ret, "Davinci model init failed");

    vector<InputOutputDescInfo> input_list;
    vector<InputOutputDescInfo> output_list;

    ret = davinci_model->GetInputOutputDescInfo(input_list, output_list);

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, GE_CHK_RT(rtDeviceReset(0));
                                     return ret, "Davinci model GetInputOutputDescInfo failed");

    for (const auto &desc : input_list) {
      model_info.input_desc.push_back(desc.shape_info);
    }
    for (const auto &desc : output_list) {
      model_info.output_desc.push_back(desc.shape_info);
    }

    model_info.name = davinci_model->Name();
  } catch (...) {
    DOMI_LOGE("OM model parser failed, some exceptions occur !");
    GE_CHK_RT(rtDeviceReset(0));
    return FAILED;
  }

  GE_CHK_RT(rtDeviceReset(0));

  return SUCCESS;
}

DavinciModelParser::DavinciModelParser() {}

DavinciModelParser::~DavinciModelParser() {}
}  // namespace ge
