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

#ifndef INC_FRAMEWORK_GENERATOR_GENERATOR_API_H_
#define INC_FRAMEWORK_GENERATOR_GENERATOR_API_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t Status_t;

typedef void *OpAttr_t;
typedef void *OpTensor_t;

///
/// @ingroup ge
/// @brief Generate offline model for the op.
/// @param [in] op_type: type name of the op.
/// @param [in] in_tensor: input description array (created by OpTensorCreate).
/// @param [in] in_num: number of in_tensor.
/// @param [in] out_tensor: output description array (created by OpTensorCreate).
/// @param [in] out_num: number of out_tensor.
/// @param [in] attr: the attributes of the op (created by OpAttrCreate).
/// @param [in] om_file: file name for the om to save.
/// @return 0 for success / others for fail
///
GE_FUNC_VISIBILITY extern Status_t OpTaskGernerator(const char *op_type, const OpTensor_t *in_tensor, int in_num,
                                                    const OpTensor_t *out_tensor, int out_num, const OpAttr_t attr,
                                                    const char *om_file);

///
/// @ingroup ge
/// @brief Create Tensor Description.
/// @param [in] format: tensor format of the data.
/// @param [in] datatype: tensor type of the data.
/// @param [in] shape: tensor shape array.
/// @param [in] num: number of shape.
/// @return OpTensor_t for success / nullptr for failure
///
GE_FUNC_VISIBILITY extern OpTensor_t OpTensorCreate(int format, int datatype, const int64_t *shape, int num);

///
/// @ingroup ge
/// @brief Destroy Tensor Description.
/// @param [in] OpTensor_t tensor: created by OpTensorCreate.
/// @param [out] none
/// @return 0 for success / others for failure.
///
GE_FUNC_VISIBILITY extern Status_t OpTensorDestroy(OpTensor_t tensor);

///
/// @ingroup ge
/// @brief Create an attribute holder.
/// @param [in] none
/// @param [out] none
/// @return OpAttr_t for success / nullptr for failure.
///
GE_FUNC_VISIBILITY extern OpAttr_t OpAttrCreate();

///
/// @ingroup ge
/// @brief Destroy Attribute holder.
/// @param [in] OpAttr_t attr: created by OpAttrCreate.
/// @param [out] none
/// @return 0 for success / others for failure.
///
GE_FUNC_VISIBILITY extern Status_t OpAttrDestroy(OpAttr_t attr);

///
/// @ingroup ge
/// @brief Set a boolean attribute to the attribute holder.
/// @param [in] attr: attribute holder (created by OpAttrCreate).
/// @param [in] name: attribute name (can`t be nullptr, end with '\0').
/// @param [in] value: attributed value.
/// @return 0 for success / others for failure.
///
GE_FUNC_VISIBILITY extern Status_t SetAttrBool(OpAttr_t attr, const char *name, bool value);

///
/// @ingroup ge
/// @brief Set an integer attribute to the attribute holder.
/// @param [in] attr: attribute holder (created by OpAttrCreate).
/// @param [in] name: attribute name (can`t be nullptr, end with '\0').
/// @param [in] value: attribute value.
/// @return 0 for success / others for failure.
///
GE_FUNC_VISIBILITY extern Status_t SetAttrInt(OpAttr_t attr, const char *name, int64_t value);

///
/// @ingroup ge
/// @brief Set a float attribute to the attribute holder.
/// @param [in] attr: attribute holder (created by OpAttrCreate).
/// @param [in] name: attribute name (can`t be nullptr, end with '\0').
/// @param [in] value: attribute value.
/// @return 0 for success / others for failure.
///
GE_FUNC_VISIBILITY extern Status_t SetAttrFloat(OpAttr_t attr, const char *name, float value);

///
/// @ingroup ge
/// @brief Set a string attribute to the attribute holder.
/// @param [in] attr: attribute holder (created by OpAttrCreate).
/// @param [in] name: attribute name (can`t be nullptr, end with '\0').
/// @param [in] value: attribute value (can`t be nullptr, end with '\0').
/// @return 0 for success / others for failure.
///
GE_FUNC_VISIBILITY extern Status_t SetAttrString(OpAttr_t attr, const char *name, const char *value);

///
/// @ingroup ge
/// @brief Set a boolean array attribute to the attribute holder.
/// @param [in] attr: attribute holder (created by OpAttrCreate).
/// @param [in] name: attribute name (can`t be nullptr, end with '\0').
/// @param [in] value: attribute value array.
/// @param [in] num: number of value array.
/// @return 0 for success / others for failure.
///
GE_FUNC_VISIBILITY extern Status_t SetAttrBoolList(OpAttr_t attr, const char *name, const bool *value, int num);

///
/// @ingroup ge
/// @brief Set an integer array attribute to the attribute holder.
/// @param [in] attr: attribute holder (created by OpAttrCreate).
/// @param [in] name: attribute name (can`t be nullptr, end with '\0').
/// @param [in] value: attribute value array.
/// @param [in] num: number of value array.
/// @return 0 for success / others for failure.
///
GE_FUNC_VISIBILITY extern Status_t SetAttrIntList(OpAttr_t attr, const char *name, const int64_t *value, int num);

///
/// @ingroup ge
/// @brief Set a float array attribute to the attribute holder.
/// @param [in] attr: attribute holder (created by OpAttrCreate).
/// @param [in] name: attribute name (can`t be nullptr, end with '\0').
/// @param [in] value: attribute value array.
/// @param [in] num: number of value array.
/// @return 0 for success / others for failure.
///
GE_FUNC_VISIBILITY extern Status_t SetAttrFloatList(OpAttr_t attr, const char *name, const float *value, int num);

///
/// @ingroup ge
/// @brief Set a string array attribute to the attribute holder.
/// @param [in] attr: attribute holder (created by OpAttrCreate).
/// @param [in] name: attribute name (can`t be nullptr, end with '\0').
/// @param [in] value: attribute value array (each value can`t be nullptr, end with '\0').
/// @param [in] num: number of value array.
/// @return 0 for success / others for failure.
///
GE_FUNC_VISIBILITY extern Status_t SetAttrStringList(OpAttr_t attr, const char *name, const char **value, int num);

#ifdef __cplusplus
}
#endif

#endif  // INC_FRAMEWORK_GENERATOR_GENERATOR_API_H_
