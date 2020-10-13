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

#include "hybrid/node_executor/host_cpu/kernel/random_uniform_kernel.h"
#include <random>
#include "common/fp16_t.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/utils/type_utils.h"
#include "hybrid/node_executor/host_cpu/kernel_factory.h"

namespace {
const char *const kAttrDtype = "dtype";
}

namespace ge {
namespace hybrid {
namespace host_cpu {
Status RandomUniformKernel::Compute(TaskContext& context) {
  GELOGI("[%s] compute begin.", node_->GetName().c_str());

  int64_t seed = 0;
  int64_t seed2 = 0;
  (void)AttrUtils::GetInt(node_->GetOpDesc(), "seed", seed);
  (void)AttrUtils::GetInt(node_->GetOpDesc(), "seed2", seed2);
  DataType data_type = DT_FLOAT;
  if (!AttrUtils::GetDataType(node_->GetOpDesc(), kAttrDtype, data_type)) {
    GELOGE(PARAM_INVALID, "[%s] get attr dtype failed.", node_->GetName().c_str());
    return PARAM_INVALID;
  }
  switch (data_type) {
    case DT_FLOAT16:
      if (GenerateFP16(node_->GetOpDesc(), seed, seed2, context) != SUCCESS) {
        GELOGE(FAILED, "Generate random_distribution failed, data_type=DT_FLOAT");
        return FAILED;
      }
      break;
    case DT_FLOAT:
      if (Generate<float>(node_->GetOpDesc(), seed, seed2, context) != SUCCESS) {
        GELOGE(FAILED, "Generate random_distribution failed, data_type=DT_FLOAT");
        return FAILED;
      }
      break;
    case DT_DOUBLE:
      if (Generate<double>(node_->GetOpDesc(), seed, seed2, context) != SUCCESS) {
        GELOGE(FAILED, "Generate random_distribution failed, data_type=DT_DOUBLE");
        return FAILED;
      }
      break;
    default:
      GELOGE(UNSUPPORTED, "Supported DataType is DT_FLOAT16 / DT_FLOAT / DT_DOUBLE, but data_type=%s",
             TypeUtils::DataTypeToSerialString(data_type).c_str());
      return UNSUPPORTED;
  }

  GELOGI("[%s] compute success.", node_->GetName().c_str());
  return SUCCESS;
}

template <typename T>
Status RandomUniformKernel::Generate(const ge::OpDescPtr &op_desc_ptr, int64_t seed, int64_t seed2,
                                     TaskContext& context) {
  GE_CHECK_NOTNULL(op_desc_ptr);
  // RandomUniformOp has and only has one output
  int64_t data_num = op_desc_ptr->GetOutputDesc(0).GetShape().GetShapeSize();
  AllocationAttr attr;
  attr.SetMemType(HOST_DDR);
  auto tensor_size = data_num * sizeof(T);
  TensorValue tensor;
  GE_CHK_STATUS_RET(context.AllocateTensor(tensor_size, tensor, &attr),
                    "[%s] Failed to allocate output of size %zu",
                    context.GetNodeName(),
                    tensor_size);

  auto *buf = reinterpret_cast<T *>(tensor.MutableData());
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
    *(buf + i) = distribution(gen);
  }

  GE_CHK_STATUS_RET(context.SetOutput(0, tensor), "[%s] Failed to set output.", context.GetNodeName());
  return SUCCESS;
}

Status RandomUniformKernel::GenerateFP16(const ge::OpDescPtr &op_desc_ptr, int64_t seed, int64_t seed2,
                                         TaskContext& context) {
  GE_CHECK_NOTNULL(op_desc_ptr);
  // RandomUniformOp has and only has one output
  int64_t data_num = op_desc_ptr->GetOutputDesc(0).GetShape().GetShapeSize();
  AllocationAttr attr;
  attr.SetMemType(HOST_DDR);
  auto tensor_size = data_num * sizeof(fp16_t);
  TensorValue tensor;
  GE_CHK_STATUS_RET(context.AllocateTensor(tensor_size, tensor, &attr),
                    "[%s] Failed to allocate output of size %zu",
                    context.GetNodeName(),
                    tensor_size);

  auto *buf = reinterpret_cast<fp16_t *>(tensor.MutableData());
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
  std::uniform_real_distribution<float> distribution(0, 1);
  for (int64_t i = 0; i < data_num; i++) {
    *(buf + i) = static_cast<fp16_t>(distribution(gen));
  }

  GE_CHK_STATUS_RET(context.SetOutput(0, tensor), "[%s] Failed to set output.", context.GetNodeName());
  return SUCCESS;
}

REGISTER_KERNEL_CREATOR(RandomUniform, RandomUniformKernel);
}  // namespace host_cpu
}  // namespace hybrid
}  // namespace ge
