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

#include "omm/csa_interact.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/util.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_utils.h"
#include "mmpa/mmpa_api.h"
#include "nlohmann/json.hpp"

namespace ge {
namespace {
const char FMK_STATUS_FILE_DIR_ENV[] = "FMK_STATUS_FILE_DIR";
const char JOBSTATE_FILE_NAME[] = "jobstateupdate_framework";
const char HCOM_DETECT_FILE_NAME[] = "hcom_detection_result";
const char FILE_SEPARATE[] = "/";
}  // namespace

///
/// @brief Obtain  CsaInteract instance
/// @return CsaInteract instance
///
CsaInteract &CsaInteract::GetInstance() {
  static CsaInteract instance;
  return instance;
}

///
/// @brief CsaInteract instance initialization
/// @param [in] dev_index  device index
/// @param [in] job_id  job id
/// @return void
///
void CsaInteract::Init(int32_t dev_index, int64_t job_id) {
  if (!is_init_) {
    dev_index_ = dev_index;
    job_id_ = job_id;

    char file_dir_env[MMPA_MAX_PATH] = { 0x00 };
    INT32 res = mmGetEnv(FMK_STATUS_FILE_DIR_ENV, file_dir_env, MMPA_MAX_PATH);
    string csa_path_prefix;
    if (res == EN_OK) {
      csa_path_prefix = file_dir_env;
    }
    if (!csa_path_prefix.empty()) {
      job_state_file_ = csa_path_prefix + std::to_string(dev_index_) + FILE_SEPARATE + JOBSTATE_FILE_NAME;
      hcom_detect_file_ = csa_path_prefix + std::to_string(dev_index_) + FILE_SEPARATE + HCOM_DETECT_FILE_NAME;
    }
    is_init_ = true;
  }
}

///
/// @brief Update job state file
/// @param [in] job_state  job state
/// @param [in] job_sub_state  detailed job state
/// @param [in] module_ret_errcode  sub module training failure error code
/// @param [in] error_module  error module identified by FMK
/// @return Status
///
Status CsaInteract::WriteJobState(JobState job_state, JobSubState job_sub_state, uint32_t module_ret_errcode,
                                  ErrorModule error_module) {
  if (!is_init_) {
    GELOGE(INTERNAL_ERROR, "[Init][CsaInteract] obj has not init, can't WriteJobState");
    REPORT_INNER_ERROR("E19999", "WriteJobState failed before init. ");
    return INTERNAL_ERROR;
  }
  if ((curr_state_ == JOBSTATE_FAILED) || (curr_state_ == JOBSTATE_KILLED)) {
    return SUCCESS;
  }

  if (job_state_file_.empty()) {
    return SUCCESS;
  }

  std::string content;
  try {
    nlohmann::json content_json;
    content_json["job_id"] = job_id_;
    content_json["jobstate"] = job_state;
    // Only the running or running failure state has a job sub state
    if ((job_state == JOBSTATE_RUNNING) || (job_state == JOBSTATE_FAILED)) {
      content_json["job_sub_state"] = job_sub_state;
    }
    content_json["time"] = CurrentTimeInStr();
    // Write error code only if run failed
    if (job_state == JOBSTATE_FAILED) {
      content_json["errorcode"] = module_ret_errcode;
      content_json["errmodule"] = error_module;
    }

    content = content_json.dump();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(INTERNAL_ERROR, "[Create][JsonObject] exception:%s job_state:%u job_sub_state:%u.",
           e.what(), job_state, job_sub_state);
    REPORT_INNER_ERROR("E19999", "Create json object failed. exception:%s job_state:%u job_sub_state:%u.",
                       e.what(), job_state, job_sub_state);
    return INTERNAL_ERROR;
  }

  if (WriteFile(job_state_file_, content) != SUCCESS) {
    // The error log subfunction has been printed and will not print again
    return INTERNAL_ERROR;
  }

  curr_state_ = job_state;
  return SUCCESS;
}

///
/// @brief Update error code in the job state file
/// @param [in] module_ret_errcode  sub module training failure error code
/// @param [in] error_module  error module identified by FMK
/// @param [in] job_sub_state  detailed job state
/// @return void
///
void CsaInteract::WriteErrorCode(uint32_t module_ret_errcode, ErrorModule error_module, JobSubState job_sub_state) {
  // The error log subfunction has been printed and will not print again
  Status ret = WriteJobState(JOBSTATE_FAILED, job_sub_state, module_ret_errcode, error_module);
  if (ret != SUCCESS) {
    GELOGW("write error code fail. ret_code: %u, status: %u", module_ret_errcode, job_sub_state);
  }
}

///
/// @brief Record errors that occurred durning the training
/// @param [in] module_ret_errcode  sub module training failure error code
/// @param [in] error_module  error module identified by FMK
/// @param [in] job_sub_state  detailed job state
/// @return void
///
void CsaInteract::StoreInternalErrorCode(uint32_t module_ret_errcode, ErrorModule error_module,
                                         JobSubState job_sub_state) {
  is_have_internal_error_ = true;

  csa_error_code_.module_ret_errcode = module_ret_errcode;
  csa_error_code_.error_module = error_module;
  csa_error_code_.job_sub_state = job_sub_state;
}

///
/// @brief Update training error code in the job state file
/// @return void
///
void CsaInteract::WriteInternalErrorCode() {
  if (is_have_internal_error_) {
    WriteErrorCode(csa_error_code_.module_ret_errcode, csa_error_code_.error_module, csa_error_code_.job_sub_state);
  }
}

///
/// @brief Update network connectivity detect file
/// @param [in] content network connectivity content
/// @return Status
///
Status CsaInteract::WriteHcomDetection(const std::string &content) {
  if (!is_init_) {
    GELOGE(INTERNAL_ERROR, "[Init][CsaInteract] obj has not init, can't WriteJobState");
    REPORT_INNER_ERROR("E19999", "WriteHcomDetection failed before init.");
    return INTERNAL_ERROR;
  }

  if (hcom_detect_file_.empty()) {
    return SUCCESS;
  }

  return WriteFile(hcom_detect_file_, content);
}

///
/// @ingroup WriteFile
/// @brief Write the content into the file. If the file does not exist, create the file
/// @param [in] file_name: File name to be written
/// @param [in] content: Contents to be written
/// @return Status
///
Status CsaInteract::WriteFile(const std::string &file_name, const std::string &content) {
  // if file path is not exist, then make path
  INT32 flags = M_WRONLY | O_TRUNC | M_CREAT;
  int32_t fd = mmOpen2(file_name.c_str(), flags, M_IRUSR | M_IWUSR | M_UMASK_GRPREAD);
  if (fd == EN_ERROR) {
    if (MakePath(file_name) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Create][File Path] errno is %d", errno);
      REPORT_CALL_ERROR("E19999", "MakePath failed. errno is %d", errno);
      return INTERNAL_ERROR;
    }
    fd = mmOpen2(file_name.c_str(), flags, M_IRUSR | M_IWUSR | M_UMASK_GRPREAD);
    if (fd == EN_ERROR) {
      GELOGE(INTERNAL_ERROR, "[Open][File] errno is %d file_name: %s", errno, file_name.c_str());
      REPORT_CALL_ERROR("E19999", "mmOpen2 failed. errno is %d file_name: %s", errno, file_name.c_str());
      return INTERNAL_ERROR;
    }
  }

  mmSsize_t ret = mmWrite(fd, reinterpret_cast<void *>(const_cast<char *>(content.c_str())), content.length());
  if (ret == EN_ERROR) {
    GELOGE(INTERNAL_ERROR, "[Write][File] errno is %d", errno);
    REPORT_CALL_ERROR("E19999", "mmWrite failed. errno is %d", errno);
    ret = mmClose(fd);
    if (ret == EN_ERROR) {
      GELOGE(INTERNAL_ERROR, "[Close][File] error is %d", errno);
      REPORT_CALL_ERROR("E19999", "mmClose failed. error is %d", errno);
    }
    return INTERNAL_ERROR;
  }
  ret = mmClose(fd);
  if (ret == EN_ERROR) {
    GELOGE(INTERNAL_ERROR, "[Close][File] error is %d", errno);
    REPORT_CALL_ERROR("E19999", "mmClose failed. error is %d", errno);
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

///
/// @ingroup MakePath
/// @brief Verify whether the file path exists, if not, recursively create the folder
/// @param [in] file_name: File name to be verified
/// @return Status
///
Status CsaInteract::MakePath(const std::string &file_name) {
  std::size_t found = file_name.find_last_of("/");
  if (found == std::string::npos) {
    return PARAM_INVALID;
  }

  std::string file_path = file_name.substr(0, found + 1);
  if (mmAccess(file_path.c_str()) == EN_OK) {
    return SUCCESS;
  }

  found = file_path.find_first_of("/");
  while (found != std::string::npos) {
    std::string pre_path = file_path.substr(0, found + 1);
    if (mmAccess(pre_path.c_str()) != EN_OK) {
      if (mmMkdir(pre_path.c_str(), M_IRWXU) != EN_OK) {
        GELOGE(INTERNAL_ERROR, "[Create][FileDir] fail, errno is %d, pre_path:%s", errno, pre_path.c_str());
        REPORT_CALL_ERROR("E19999", "mmMkdir failed. errno is %d pre_path:%s", errno, pre_path.c_str());
        return INTERNAL_ERROR;
      }
    }
    found = file_path.find_first_of("/", found + 1);
  }

  return SUCCESS;
}
}  // namespace ge
