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

#ifndef INC_FRAMEWORK_COMMON_OP_GE_OP_UTILS_H_
#define INC_FRAMEWORK_COMMON_OP_GE_OP_UTILS_H_

#include <cce/dnn.h>
#include <memory>
#include <vector>

#include "common/op/attr_value_util.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "proto/insert_op.pb.h"

namespace ge {
using namespace cce;
using domi::Status;

// Add Sub Mul
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t ADD_INPUT_NUM;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t SUB_INPUT_NUM;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t MUL_INPUT_NUM;

// Permute
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const int32_t PERMUTE_ORDER_NUM;

// Ssd PriroBox
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const double SSD_PRIORBOX_ASPECT_RATIO_VALUE;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t STRIDEDSLICE_INPUT_NUM;

// Switch
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t SWITCH_INPUT_NUM;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t SWITCH_OUTPUT_NUM;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t SWITCH_FALSE_OUTPUT;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t SWITCH_TRUE_OUTPUT;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t SWITCH_DATA_INPUT;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t SWITCH_PRED_INPUT;

// FunctionOp
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t IF_COND_INPUT;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t FOR_START_INPUT;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t FOR_LIMIT_INPUT;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t FOR_DELTA_INPUT;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const uint32_t FOR_DATA_INPUT;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const int NORMAL_TENSOR_SIZE;

class OpUtils {
 public:
  ///
  /// @ingroup domi_ome
  /// @brief Check whether check_value is in [min_enum_value, max_enum_value]
  /// @return true Within
  /// @return false out of range
  //
  static inline bool CheckEnumValid(int32_t check_value, int32_t min_enum_value, int32_t max_enum_value) {
    return check_value < min_enum_value ? false : (check_value >= max_enum_value ? false : true);
  }
  ///
  /// @ingroup domi_omg
  /// @brief Convert the dimension of array according to different format
  /// @param [in] src_format src_shape format
  /// @param [in] src Dimension array to be converted
  /// @param [in] dst_format Target format after conversion
  /// @param [out] dst Dimension array after conversion
  /// @return SUCCESS success
  /// @return FAILED fail
  ///
  static bool ConvertDim(ccTensorFormat_t src_format, const std::vector<int64_t> &src, ccTensorFormat_t dst_format,
                         std::vector<int64_t> &dst);
  ///
  /// @ingroup domi_omg
  /// @brief Determine whether to manually calculate the tensor size based on the values of format and dim
  /// @param [in] format, Format information of the tensor
  /// @param [in] real_dim_cnt, Tensor dim
  /// @return true Manually calculate the size based on dim and datatype
  /// @return false skip
  ///
  static bool IsComputDimsSize(const int32_t format, const uint32_t real_dim_cnt);
  ///
  /// @ingroup domi_ome
  /// @brief Initialize the tensor description, which is used for input and output.
  /// @param [in] model_tensor Tensor information defined by the offline model
  /// @param [out] cc_tensor Tensor definition used by CC
  /// @return SUCCESS success
  /// @return FAILED fail
  ///
  static Status InitTensorDescriptor(const ge::GeTensorDesc &model_tensor, ccTensorDescriptor_t &cc_tensor);
  ///
  /// @ingroup domi_ome
  /// @brief Initialize the tensor description, which is used for input and output.
  /// @param [in] model_tensor Tensor information defined by the offline model
  /// @param [in] dst_data_type data_type of the target cc_tensor
  /// @param [out] cc_tensor Tensor definition used by CC
  /// @return SUCCESS success
  /// @return FAILED fail
  ///
  static Status InitTensorDescriptor(const ge::GeTensorDesc &model_tensor, int32_t dst_data_type,
                                     ccTensorDescriptor_t &cc_tensor);
  ///
  /// @ingroup domi_ome
  /// @brief Initialize the tensor description for bias.
  /// @param [in] model_tensor Tensor information defined by the offline model
  /// @param [out]  cc_tensor Tensor definition used by CC
  /// @return SUCCESS success
  /// @return FAILED fail
  ///
  ///
  static Status InitTensorDescriptor(const ge::GeTensor &model_tensor, ccTensorDescriptor_t &cc_tensor);
  ///
  /// @ingroup domi_ome
  /// @brief Initialize the tensor description for bias.
  /// @param [in] model_tensor Tensor information defined by the offline model
  /// @param [in] dst_data_type data_type of the target cc_tensor
  /// @param [out] cc_tensor Tensor definition used by CC
  /// @return SUCCESS success
  /// @return FAILED fail
  ///
  static Status InitTensorDescriptor(const ge::GeTensor &model_tensor, int32_t dst_data_type,
                                     ccTensorDescriptor_t &cc_tensor);

  static Status InitTensorDescriptor(int32_t format, int32_t data_type, const std::vector<int64_t> &dim,
                                     ccTensorDescriptor_t &cc_tensor, uint32_t real_dim_cnt = 4);
  ///
  /// @ingroup domi_ome
  /// @brief Destroys a tensor
  /// @param [inout] cc_tensor Tensor definition used by CC
  ///
  static void DestroyTensorDescriptor(ccTensorDescriptor_t &cc_tensor) noexcept;

  ///
  /// @ingroup domi_ome
  /// @brief Destroys a tensor
  /// @param [inout] cc_filter cc_filter Definition of the filter used by CC
  ///
  static void DestroyFilterDescriptor(ccFilterDescriptor_t &cc_filter);

  ///
  /// @ingroup domi_ome
  /// @brief Initializing Filter Description
  /// @param [in] model_filter Filter information defined in the offline model
  /// @param [out] cc_filter Definition of the filter used by CC
  /// @return SUCCESS success
  /// @return FAILED fail
  ///
  static Status InitFilterDescriptor(const ge::GeTensor &model_filter, ccFilterDescriptor_t &cc_filter);

  ///
  /// @brief Extract AIPP parameters from AttrDefMap and splice them
  /// @param [in] aipp_attr attr of operator
  /// @param [out] aipp_params aipp parameters
  /// @return enum of tagCCAippInputFormat
  ///
  static Status ConvertAippParams(const GeAttrValue::NamedAttrs &aipp_attr, domi::AippOpParams *aipp_params);
  static Status TransferDim(const std::vector<int64_t> &dim, std::vector<int64_t> &dim_vector);
  template <typename T>
  static void SliceData(const std::vector<char *> &input, int64_t chunk_size, std::vector<char *> &output,
                        int64_t begin, int64_t out_dim, int64_t stride);
  template <typename T>
  static Status SetDataByDataType(size_t out_size, const std::vector<char *> &chunk_input,
                                  const std::vector<char *> &chunk_output, GeTensor *output);
  template <typename T>
  static Status SetOutputSliceDataByDataType(void *data, int64_t data_size, const std::vector<int64_t> &input_dims,
                                             const std::vector<int64_t> &begin, const std::vector<int64_t> &output_dims,
                                             ge::GeTensor *output, const std::vector<int64_t> &stride);
  static Status SetOutputSliceData(void *data, int64_t data_size, int32_t data_type, std::vector<int64_t> &input_dims,
                                   std::vector<int64_t> &begin, std::vector<int64_t> &output_dims, ge::GeTensor *output,
                                   std::vector<int64_t> &stride);

  ///
  /// @ingroup domi_omg
  /// @brief Convert the convolutional weight data from [h, w, c, k] to [k, c, h, w]
  /// @param [in] input Weight data in HWCK format
  /// @param [in] H value of H dimension
  /// @param [in] W value of W dimension
  /// @param [in] C value of C dimension
  /// @param [in] K value of K dimension
  /// @param [out] output Data pointer after conversion. The format is KCHW.
  ///
  static void TransDataHWCK2KCHW(const void *input, int64_t H, int64_t W, int64_t C, int64_t K, void **output);
  ///
  /// @ingroup domi_omg
  /// @brief Converts the convolutional weight data from [k, c, h, w] to [h, w, c, k].
  /// @param [in] input Weight data in HWCK format
  /// @param [in] K value of K dimension
  /// @param [in] C value of C dimension
  /// @param [in] H value of H dimension
  /// @param [in] W value of W dimension
  /// @param [out] output Data pointer after conversion. The format is HWCK
  ///
  static void TransDataKCHW2HWCK(const void *input, int64_t K, int64_t C, int64_t H, int64_t W, void *output);
  ///
  /// @ingroup domi_omg
  /// @brief Initialize the input and output description of the data node which is applied to filter weight in the
  /// training network
  /// @param [in] model_tensor input and output tensor information
  /// @param [out] cc_tensor Tensor in CCE format after conversion
  ///
  static Status InitFilterTensorDescriptor(const ge::GeTensorDesc &model_tensor, ccFilterDescriptor_t &cc_tensor);

  static void SetTensorDescriptorAllOffsetQuantizeInfo(const GeTensorDesc &tensor, ccTensorDescriptor_t cc_tensor);
  static vector<ConstGeTensorPtr> GetWeights(const ge::Node &node);
  static vector<ConstGeTensorPtr> GetWeights(ge::ConstNodePtr node);
  static vector<GeTensorPtr> MutableWeights(const ge::Node &node);
  static vector<GeTensorPtr> MutableWeights(const ge::NodePtr node);
  static Status SetWeights(ge::Node &node, const vector<ge::GeTensorPtr> &weights);
  static Status SetWeights(ge::NodePtr node, const vector<ge::GeTensorPtr> &weights);
  static Status GetShapeDataFromConstTensor(const ConstGeTensorPtr &tensor, DataType type, std::vector<int64_t> &dims);

 private:
  friend class CceTensorDescriptor;
  static uint32_t GetRealDimCnt(const GeTensorDesc &tensor_desc);
};

class CceTensorDescriptor;

using CceTensorDescriptorPtr = std::shared_ptr<CceTensorDescriptor>;

class CceTensorDescriptor {
 public:
  explicit CceTensorDescriptor(ccTensorDescriptor_t cc_tensor);
  CceTensorDescriptor(const CceTensorDescriptor &) = delete;
  CceTensorDescriptor &operator=(const CceTensorDescriptor &) = delete;

  ~CceTensorDescriptor();

  ccTensorDescriptor_t GetPtr() { return cc_tensor_; }

  ///
  /// @brief      Initializes the tensor based on shape information.
  /// @param[in]  format  data permutation format
  /// @param[in]  data_type Data Type
  /// @param[in]  dim dim information
  /// @return     return code
  ///
  Status InitTensor(int32_t format, int32_t data_type, const std::vector<int64_t> &dims);

  Status InitTensor(int32_t format, int32_t data_type, const ge::GeShape &shape);

  ///
  /// @brief      get format of tensor
  /// @param[out] format format of the tensor
  /// @return     return code
  ///
  Status GetFormat(ccTensorFormat_t *format);

  ///
  /// @brief      Obtains the size of the tensor.
  /// @param[out] size size of Tensor
  /// @return     return code
  ///
  Status GetTensorSizeInBytes(uint32_t *size);

  ///
  /// @brief transform tensor between 4d(NCHW) and 5d(NC1HWC0)
  /// @param [in] xDesc   descriptor of input tensor
  /// @param [in] x   point to input data in host memory
  /// @param [in] dataTypeTransmode   mode of data type transform
  /// @param [in] yDesc   descriptor of output tensor
  /// @param [in|out] y   point to output data in host memory
  /// @param [in] ySizeInBytes   size of outputData
  /// @return return code
  ///
  static Status TransTensor(const ccTensorDescriptor_t xDesc, const void *x, const CceTensorDescriptorPtr &yDesc,
                            void *y, uint32_t ySizeInBytes);

  ///
  /// @brief      CceTensorDescriptor Static Constructor
  /// @return     CceTensorDescriptor smart pointer
  ///
  static CceTensorDescriptorPtr Create();

  ccTensorDescriptor_t cc_tensor_ = nullptr;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_OP_GE_OP_UTILS_H_
