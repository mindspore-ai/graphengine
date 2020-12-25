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

#ifndef PLATFORM_INFOS_DEF_H
#define PLATFORM_INFOS_DEF_H

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "platform_info_def.h"

namespace fe {
class StrInfoImpl;
using StrInfoImplPtr = std::shared_ptr<StrInfoImpl>;
class StrInfos {
 public:
  bool Init();
  std::string GetAIcVersion();
  std::string GetCcecAIcVersion();
  std::string GetCcecAIvVersion();
  std::string IsSupportAICpuCompiler();

  void SetAIcVersion(std::string &aic_version);
  void SetCcecAIcVersion(std::string &ccec_aic_version);
  void SetCcecAIvVersion(std::string &ccec_aiv_version);
  void SetIsSupportAICpuCompiler(std::string &is_support_ai_cpu_compiler);
 private:
  StrInfoImplPtr str_info_impl_{nullptr};
};

class SoCInfoImpl;
using SoCInfoImplPtr = std::shared_ptr<SoCInfoImpl>;
class SoCInfos {
 public:
  bool Init();
  uint32_t GetAICoreCnt();
  uint32_t GetVectorCoreCnt();
  uint32_t GetAICpuCnt();
  MemoryType GetMemType();
  uint64_t GetMemSize();
  L2Type GetL2Type();
  uint64_t GetL2Size();
  uint32_t GetL2PageNum();

  void SetAICoreCnt(uint32_t ai_core_cnt);
  void SetVectorCoreCnt(uint32_t vector_core_cnt);
  void SetAICpuCnt(uint32_t ai_cpu_cnt);
  void SetMemType(MemoryType memory_type);
  void SetMemSize(uint64_t memory_size);
  void SetL2Type(L2Type l2_type);
  void SetL2Size(uint64_t l2_size);
  void SetL2PageNum(uint32_t l2_page_num);
 private:
  SoCInfoImplPtr soc_info_impl_{nullptr};
};

class AICoreSpecImpl;
using AICoreSpecImplPtr = std::shared_ptr<AICoreSpecImpl>;
class AICoreSpecs {
 public:
  bool Init();
  double GetCubeFreq();
  uint64_t GetCubeMSize();
  uint64_t GetCubeNSize();
  uint64_t GetCubeKSize();
  uint64_t GetVecCalcSize();
  uint64_t GetL0aSize();
  uint64_t GetL0bSize();
  uint64_t GetL0cSize();
  uint64_t GetL1Size();
  uint64_t GetSmaskBuffer();
  uint64_t GetUBSize();
  uint64_t GetUBBlockSize();
  uint64_t GetUBBankSize();
  uint64_t GetUBBankNum();
  uint64_t GetUBBurstInOneBlock();
  uint64_t GetUBBankGroupNum();
  uint32_t GetUnzipEngines();
  uint32_t GetUnzipMaxRatios();
  uint32_t GetUnzipChannels();
  uint8_t GetUnzipIsTight();
  uint8_t GetCubeVectorSplit();

  void SetCubeFreq(double cube_freq);
  void SetCubeMSize(uint64_t cube_m_size);
  void SetCubeNSize(uint64_t cube_n_size);
  void SetCubeKSize(uint64_t cube_k_size);
  void SetVecCalcSize(uint64_t vec_calc_size);
  void SetL0aSize(uint64_t l0_a_size);
  void SetL0bSize(uint64_t l0_b_size);
  void SetL0cSize(uint64_t l0_c_size);
  void SetL1Size(uint64_t l1_size);
  void SetSmaskBuffer(uint64_t smask_buffer);
  void SetUBSize(uint64_t ub_size);
  void SetUBBlockSize(uint64_t ubblock_size);
  void SetUBBankSize(uint64_t ubbank_size);
  void SetUBBankNum(uint64_t ubbank_num);
  void SetUBBurstInOneBlock(uint64_t ubburst_in_one_block);
  void SetUBBankGroupNum(uint64_t ubbank_group_num);
  void SetUnzipEngines(uint32_t unzip_engines);
  void SetUnzipMaxRatios(uint32_t unzip_max_ratios);
  void SetUnzipChannels(uint32_t unzip_channels);
  void SetUnzipIsTight(uint8_t unzip_is_tight);
  void SetCubeVectorSplit(uint8_t cube_vector_split);
 private:
  AICoreSpecImplPtr aicore_spec_impl_{nullptr};
};

class AICoreMemRateImpl;
using AICoreMemRateImplPtr = std::shared_ptr<AICoreMemRateImpl>;
class AICoreMemRates {
 public:
  bool Init();
  double GetDdrRate();
  double GetDdrReadRate();
  double GetDdrWriteRate();
  double GetL2Rate();
  double GetL2ReadRate();
  double GetL2WriteRate();
  double GetL1ToL0aRate();
  double GetL1ToL0bRate();
  double GetL1ToUBRate();
  double GetL0cToUBRate();
  double GetUBToL2Rate();
  double GetUBToDdrRate();
  double GetUBToL1Rate();

  void SetDdrRate(double ddr_rate);
  void SetDdrReadRate(double ddr_read_rate);
  void SetDdrWriteRate(double ddr_write_rate);
  void SetL2Rate(double l2_rate);
  void SetL2ReadRate(double l2_read_rate);
  void SetL2WriteRate(double l2_write_rate);
  void SetL1ToL0aRate(double l1_to_l0_a_rate);
  void SetL1ToL0bRate(double l1_to_l0_b_rate);
  void SetL1ToUBRate(double l1_to_ub_rate);
  void SetL0cToUBRate(double l0_c_to_ub_rate);
  void SetUBToL2Rate(double ub_to_l2_rate);
  void SetUBToDdrRate(double ub_to_ddr_rate);
  void SetUBToL1Rate(double ub_to_l1_rate);
 private:
  AICoreMemRateImplPtr aicore_mem_rate_impl_{nullptr};
};

class VectorCoreSpecImpl;
using VectorCoreSpecImplPtr = std::shared_ptr<VectorCoreSpecImpl>;
class VectorCoreSpecs {
 public:
  bool Init();
  double GetVecFreq();
  uint64_t GetVecCalcSize();
  uint64_t GetSmaskBuffer();
  uint64_t GetUBSize();
  uint64_t GetUBBlockSize();
  uint64_t GetUBBankSize();
  uint64_t GetUBBankNum();
  uint64_t GetUBBurstInOneBlock();
  uint64_t GetUBBankGroupNum();
  uint64_t GetVectorRegSize();
  uint64_t GetPredicateRegSize();
  uint64_t GetAddressRegSize();
  uint64_t GetAlignmentRegSize();

  void SetVecFreq(double vec_freq);
  void SetVecCalcSize(uint64_t vec_calc_size);
  void SetSmaskBuffer(uint64_t smask_buffer);
  void SetUBSize(uint64_t ub_size);
  void SetUBBlockSize(uint64_t ubblock_size);
  void SetUBBankSize(uint64_t ubbank_size);
  void SetUBBankNum(uint64_t ubbank_num);
  void SetUBBurstInOneBlock(uint64_t ubburst_in_one_block);
  void SetUBBankGroupNum(uint64_t ubbank_group_num);
  void SetVectorRegSize(uint64_t vector_reg_size);
  void SetPredicateRegSize(uint64_t predicate_reg_size);
  void SetAddressRegSize(uint64_t address_reg_size);
  void SetAlignmentRegSize(uint64_t alignment_reg_size);
 private:
  VectorCoreSpecImplPtr vector_core_spec_impl_{nullptr};
};

class VectorCoreMemRateImpl;
using VectorCoreMemRateImplPtr = std::shared_ptr<VectorCoreMemRateImpl>;
class VectorCoreMemRates {
 public:
  bool Init();
  double GetDdrRate();
  double GetDdrReadRate();
  double GetDdrWriteRate();
  double GetL2Rate();
  double GetL2ReadRate();
  double GetL2WriteRate();
  double GetUBToL2Rate();
  double GetUBToDdrRate();

  void SetDdrRate(double ddr_rate);
  void SetDdrReadRate(double ddr_read_rate);
  void SetDdrWriteRate(double ddr_write_rate);
  void SetL2Rate(double l2_rate);
  void SetL2ReadRate(double l2_read_rate);
  void SetL2WriteRate(double l2_write_rate);
  void SetUBToL2Rate(double ub_to_l2_rate);
  void SetUBToDdrRate(double ub_to_ddr_rate);
 private:
  VectorCoreMemRateImplPtr vector_core_mem_rate_impl_{nullptr};
};

class CPUCacheImpl;
using CPUCacheImplPtr = std::shared_ptr<CPUCacheImpl>;
class CPUCaches {
 public:
  bool Init();
  uint32_t GetAICPUSyncBySW();
  uint32_t GetTSCPUSyncBySW();

  void SetAICPUSyncBySW(uint32_t AICPUSyncBySW);
  void SetTSCPUSyncBySW(uint32_t TSCPUSyncBySW);
 private:
  CPUCacheImplPtr cpu_cache_impl_{nullptr};
};

class PlatFormInfosImpl;
using PlatFormInfosImplPtr = std::shared_ptr<PlatFormInfosImpl>;
class PlatFormInfos {
 public:
  bool Init();
  StrInfos GetStrInfo();
  SoCInfos GetSocInfo();
  AICoreSpecs GetAICoreSpec();
  AICoreMemRates GetAICoreMemRates();
  std::map<std::string, std::vector<std::string>> GetAICoreIntrinsicDtype();
  VectorCoreSpecs GetVectorCoreSpec();
  VectorCoreMemRates GetVectorCoreMemRates();
  CPUCaches GetCPUCache();
  std::map<std::string, std::vector<std::string>> GetVectorCoreIntrinsicDtype();

  void SetStrInfo(StrInfos &str_infos);
  void SetSocInfo(SoCInfos &SoC_infos);
  void SetAICoreSpec(AICoreSpecs &AICore_specs);
  void SetAICoreMemRates(AICoreMemRates &AICore_mem_rates);
  void SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsic_dtypes);
  void SetVectorCoreSpec(VectorCoreSpecs &vector_core_specs);
  void SetVectorCoreMemRates(VectorCoreMemRates &vectorcore_mem_rates);
  void SetCPUCache(CPUCaches &CPU_caches);
  void SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsic_dtypes);

 private:
  PlatFormInfosImplPtr platform_infos_impl_{nullptr};
};

class OptionalInfosImpl;
using OptionalInfosImplPtr = std::shared_ptr<OptionalInfosImpl>;
class OptionalInfos {
 public:
  bool Init();
  std::string GetSocVersion();
  std::string GetCoreType();
  uint32_t GetAICoreNum();
  std::string GetL1FusionFlag();

  void SetSocVersion(std::string soc_version);
  void SetCoreType(std::string core_type);
  void SetAICoreNum(uint32_t ai_core_num);
  void SetL1FusionFlag(std::string l1_fusion_flag);
 private:
  OptionalInfosImplPtr optional_infos_impl_{nullptr};
};

}
#endif
