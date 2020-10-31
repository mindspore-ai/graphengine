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

#ifndef GE_OMM_CSA_INTERACT_H_
#define GE_OMM_CSA_INTERACT_H_

#include <string>

#include "framework/common/ge_inner_error_codes.h"

namespace ge {
enum JobState {
  JOBSTATE_WAITING = 1,
  JOBSTATE_RUNNING,
  JOBSTATE_KILLING,
  JOBSTATE_SUCCEED,
  JOBSTATE_FAILED,
  JOBSTATE_KILLED,
  JOBSTATE_UNKOWN
};

enum JobSubState {
  JOBSUBSTATE_ENV_INIT = 201,
  JOBSUBSTATE_ENV_FIN,
  JOBSUBSTATE_RESOUCE_ALLOC,
  JOBSUBSTATE_MODEL_COMPILE,
  JOBSUBSTATE_GRAPH_PREPARE,
  JOBSUBSTATE_GRAPH_SPLIT,
  JOBSUBSTATE_GRAPH_OPTIMIZE,
  JOBSUBSTATE_GRAPH_BUILD,
  JOBSUBSTATE_GRAPH_LOAD,
  JOBSUBSTATE_GRAPH_EXEC,
  JOBSUBSTATE_GRAPH_UNLOAD,
  JOBSUBSTATE_OTHER
};

enum ErrorModule {
  ERROR_MODULE_DRIVER = 0x01,
  ERROR_MODULE_RUNTIME = 0x04,
  ERROR_MODULE_CCE = 0x06,
  ERROR_MODULE_FMK = 0x08,
  ERROR_MODULE_HCCL = 0x12
};

struct CsaErrorCode {
  CsaErrorCode()
      : module_ret_errcode(0),
        error_module(ERROR_MODULE_FMK),
        job_sub_state(JOBSUBSTATE_OTHER) {}
  ~CsaErrorCode() {}
  uint32_t module_ret_errcode;
  ErrorModule error_module;
  JobSubState job_sub_state;
};
class CsaInteract {
 public:
  ///
  /// @brief Obtain  CsaInteract instance
  /// @return CsaInteract instance
  ///
  static CsaInteract& GetInstance();

  ///
  /// @brief CsaInteract instance initialization
  /// @param [in] dev_index  device index
  /// @param [in] job_id  job id
  /// @return void
  ///
  void Init(int32_t dev_index, int64_t job_id);

  ///
  /// @brief Update job state file
  /// @param [in] job_state  job state
  /// @param [in] job_sub_state  detailed job state
  /// @param [in] module_ret_errcode  sub module training failure error code
  /// @param [in] error_module  error module identified by FMK
  /// @return Status
  ///
  Status WriteJobState(JobState job_state,
                       JobSubState job_sub_state = JOBSUBSTATE_OTHER,
                       uint32_t module_ret_errcode = SUCCESS,
                       ErrorModule error_module = ERROR_MODULE_FMK);

  ///
  /// @brief Update error code in the job state file
  /// @param [in] module_ret_errcode  sub module training failure error code
  /// @param [in] error_module  error module identified by FMK
  /// @param [in] job_sub_state  detailed job state
  /// @return void
  ///
  void WriteErrorCode(uint32_t module_ret_errcode, ErrorModule error_module,
                      JobSubState job_sub_state);

  ///
  /// @brief Record errors that occurred durning the training
  /// @param [in] module_ret_errcode  sub module training failure error code
  /// @param [in] error_module  error module identified by FMK
  /// @param [in] job_sub_state  detailed job state
  /// @return void
  ///
  void StoreInternalErrorCode(uint32_t module_ret_errcode,
                              ErrorModule error_module,
                              JobSubState job_sub_state);

  ///
  /// @brief Update training error code in the job state file
  /// @return void
  ///
  void WriteInternalErrorCode();

  ///
  /// @brief Update network connectivity detect file
  /// @param [in] content network connectivity content
  /// @return Status
  ///
  Status WriteHcomDetection(const std::string& content);

 private:
  CsaInteract()
      : dev_index_(0),
        job_id_(0),
        is_init_(false),
        curr_state_(JOBSTATE_UNKOWN),
        is_have_internal_error_(false) {}

  ~CsaInteract() {}

  CsaInteract(const CsaInteract&) = delete;
  CsaInteract(CsaInteract&&) = delete;
  CsaInteract& operator=(const CsaInteract&) = delete;
  CsaInteract& operator=(CsaInteract&&) = delete;

  ///
  /// @ingroup WriteFile
  /// @brief Write the content into the file. If the file does not exist, create the file
  /// @param [in] file_name: File name to be written
  /// @param [in] content: Contents to be written
  /// @return Status
  ///
  Status WriteFile(const std::string& file_name, const std::string& content);

  ///
  /// @ingroup MakePath
  /// @brief Verify whether the file path exists, if not, recursively create the folder
  /// @param [in] file_name: File name to be verified
  /// @return Status
  ///
  Status MakePath(const std::string& file_name);

  // device index
  int32_t dev_index_;
  // job id
  int64_t job_id_;
  // is initialization complete
  bool is_init_;
  // current job state
  JobState curr_state_;
  // job state file
  std::string job_state_file_;
  // network connectivity detect file
  std::string hcom_detect_file_;
  // identification of internal errors that occurred during the training
  bool is_have_internal_error_;
  // error code information
  CsaErrorCode csa_error_code_;
};
}  // namespace ge

#endif  // GE_OMM_CSA_INTERACT_H_

