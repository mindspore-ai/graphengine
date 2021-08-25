/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

/*!
 * \file vector_search.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_VECTOR_SEARCH_H_
#define OPS_BUILT_IN_OP_PROTO_INC_VECTOR_SEARCH_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Generate ADC(asymmetric distance computation) table. \n
*
* @par Inputs:
* Four inputs, including:
* @li query: A Tensor. Must be one of the following types: float16, float32.
* @li code_book: A Tensor. Must be one of the following types: float16, float32.
* @li centroids: A Tensor. Must be one of the following types: float16, float32.
* @li bucket_list: A Tensor. Must be one of the following types: int32, int64.
*
* @par Outputs:
* @li adc_tables: A Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(GenADC)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(code_book, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(centroids, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bucket_list, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(adc_tables, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(GenADC)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_VECTOR_SEARCH_H_
