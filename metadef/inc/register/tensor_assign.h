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

#ifndef TENSOR_ASSIGN_H_
#define TENSOR_ASSIGN_H_

#include "graph/ge_tensor.h"
#include "proto/tensorflow/tensor.pb.h"

namespace domi {
using GeTensorPtr = std::shared_ptr<ge::GeTensor>;
using Status = uint32_t;
using domi::tensorflow::TensorProto;
using google::protobuf::int32;
using google::protobuf::int64;

class TensorAssign {
 public:
  static Status SetGeTensor(const TensorProto &tensor, GeTensorPtr &weight);

  static Status SetGeTensorDataType(int64_t dataType, GeTensorPtr &weight);

  static ge::DataType ConvertTensorflowDataType(uint32_t tf_data_type);

 private:
  static bool CheckBoolVal(tensorflow::DataType data_type);

  static bool CheckHalfVal(tensorflow::DataType data_type);

  static bool CheckFloatVal(tensorflow::DataType data_type);

  static bool CheckDoubleVal(tensorflow::DataType data_type);

  static bool CheckComplex64Val(tensorflow::DataType data_type);

  static bool CheckComplex128Val(tensorflow::DataType data_type);

  static bool CheckStringVal(tensorflow::DataType data_type);

  static bool CheckByte(tensorflow::DataType data_type);

  static bool CheckDoubleByte(tensorflow::DataType data_type);

  static bool CheckSignedFourByte(tensorflow::DataType data_type);

  static bool CheckUnsignedFourByte(tensorflow::DataType data_type);

  static bool CheckSignedEightByte(tensorflow::DataType data_type);

  static bool CheckUnsignedEightByte(tensorflow::DataType data_type);

  static Status GetDoubleByteVal(int32_t val_size, const google::protobuf::RepeatedField<int32> &val_vector, int count,
                                 GeTensorPtr &weight);
  static Status GetByteVal(int32_t val_size, const google::protobuf::RepeatedField<int32> &val_vector, int count,
                           GeTensorPtr &weight);

  static Status GetStringVal(int32_t val_size, const google::protobuf::RepeatedPtrField<std::string> &val_vector,
                             int count, GeTensorPtr &weight);

  static void SetGeTensorWeightData(const TensorProto &tensor, int32_t val_size, int count, GeTensorPtr &weight);

  static void SetWeightData(tensorflow::DataType data_type, int count, const std::string &tensor_content,
                            GeTensorPtr &weight);

  template <typename T>
  static Status GetVal(int32_t val_size, const google::protobuf::RepeatedField<T> &val_vector, int count,
                       GeTensorPtr &weight) {
    bool zerosLike = (count != val_size && val_size == 1);
    T *addr = new (std::nothrow) T[count]();
    GE_CHECK_NOTNULL(addr);
    int minCount = (count > val_size) ? val_size : count;
    if (!zerosLike) {
      for (int32_t i = 0; i < minCount; i++) {
        *(addr + i) = val_vector.Get(i);
      }
      for (int32_t i = minCount; i < count; i++) {
        *(addr + i) = val_vector.Get(minCount - 1);
      }
    } else {
      for (int32_t i = 0; i < count; i++) {
        *(addr + i) = val_vector.Get(0);
      }
    }
    (void)weight->SetData(reinterpret_cast<uint8_t *>(addr), count * sizeof(T));
    GE_DELETE_NEW_ARRAY(addr);
    return SUCCESS;
  }
};
}  // namespace domi
#endif  // TENSOR_ASSIGN_H_
