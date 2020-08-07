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

#include "host_aicpu_engine/ops_kernel_store/op/random_uniform_op.h"
#include <random>
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"
#include "host_aicpu_engine/ops_kernel_store/op/op_factory.h"

namespace ge {
namespace host_aicpu {
Status RandomUniformOp::Compute(const ge::OpDescPtr &op_desc_ptr, const std::vector<ge::GeTensorPtr> &inputs,
                                std::vector<ge::GeTensorPtr> &outputs) {
  GELOGI("RandomUniformOp [%s, %s] compute begin.", node_.GetName().c_str(), node_.GetType().c_str());
  int64_t seed = 0;
  int64_t seed2 = 0;
  (void)AttrUtils::GetInt(op_desc_ptr, "seed", seed);
  (void)AttrUtils::GetInt(op_desc_ptr, "seed2", seed2);
  DataType data_type = DT_UNDEFINED;
  if (AttrUtils::GetDataType(op_desc_ptr, VAR_ATTR_DTYPE, data_type) != GRAPH_SUCCESS) {
    GELOGE(PARAM_INVALID, "get attr VAR_ATTR_DTYPE failed");
    return PARAM_INVALID;
  }

  switch (data_type) {
    case DT_FLOAT16:
      break;
    case DT_FLOAT:
      if (Generate<float>(op_desc_ptr, seed, seed2, outputs) != SUCCESS) {
        GELOGE(FAILED, "Generate random_distribution for RandomUniformOp failed, data_type=DT_FLOAT");
        return FAILED;
      }
      break;
    case DT_DOUBLE:
      if (Generate<double>(op_desc_ptr, seed, seed2, outputs) != SUCCESS) {
        GELOGE(FAILED, "Generate random_distribution for RandomUniformOp failed, data_type=DT_DOUBLE");
        return FAILED;
      }
      break;
    default:
      GELOGE(UNSUPPORTED, "Supported DataType for RandomUniformOp is DT_FLOAT16 / DT_FLOAT / DT_DOUBLE, but dtype=%s",
             TypeUtils::DataTypeToSerialString(data_type).c_str());
      return UNSUPPORTED;
  }

  GELOGI("RandomUniformOp [%s, %s] compute success.", node_.GetName().c_str(), node_.GetType().c_str());
  return SUCCESS;
}

template <typename T>
Status RandomUniformOp::Generate(const ge::OpDescPtr &op_desc_ptr, int64_t seed, int64_t seed2,
                                 std::vector<ge::GeTensorPtr> &outputs) {
  GE_CHECK_NOTNULL(op_desc_ptr);
  // RandomUniformOp has and only has one output
  int64_t data_num = op_desc_ptr->GetOutputDesc(0).GetShape().GetShapeSize();
  std::unique_ptr<T[]> buf(new (std::nothrow) T[data_num]());
  if (buf == nullptr) {
    GELOGE(MEMALLOC_FAILED, "New sizeof(T) * data_num(%zu) memory failed", static_cast<size_t>(sizeof(T) * data_num));
    return MEMALLOC_FAILED;
  }

  int64_t final_seed;
  if (seed == 0) {
    if (seed2 == 0) {
      std::random_device rd;
      final_seed = rd();
    } else {
      final_seed = seed2;
    }
  } else {
    final_seed = seed;
  }
  std::mt19937_64 gen(final_seed);
  std::uniform_real_distribution<T> distribution(0, 1);
  for (int64_t i = 0; i < data_num; i++) {
    *(buf.get() + i) = distribution(gen);
  }

  GeTensorPtr output =
    MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0), reinterpret_cast<uint8_t *>(buf.get()), data_num * sizeof(T));
  GE_CHECK_NOTNULL(output);
  outputs.emplace_back(output);

  return SUCCESS;
}

REGISTER_OP_CREATOR(RandomUniform, RandomUniformOp);
}  // namespace host_aicpu
}  // namespace ge
