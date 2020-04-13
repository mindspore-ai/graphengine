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

#ifndef GE_OP_MATH_OPS_H_
#define GE_OP_MATH_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

REG_OP(Igamma)
    .INPUT(a, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Igamma)

REG_OP(Igammac)
    .INPUT(a, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Igammac)

REG_OP(CompareAndBitpack)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, \
        DT_INT16, DT_INT32, DT_INT64, DT_BOOL }))
    .INPUT(threshold, TensorType({ DT_FLOAT, DT_FLOAT16, DT_DOUBLE, \
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_BOOL }))
    .OUTPUT(y, TensorType(DT_UINT8))
    .OP_END_FACTORY_REG(CompareAndBitpack)

REG_OP(Bincount)
    .INPUT(array, TensorType(DT_INT32))
    .INPUT(size, TensorType(DT_INT32))
    .INPUT(weights, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE }))
    .OUTPUT(bins, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE }))
    .OP_END_FACTORY_REG(Bincount)

REG_OP(Betainc)
    .INPUT(a, TensorType({DT_DOUBLE, DT_FLOAT}))
    .INPUT(b, TensorType({DT_DOUBLE, DT_FLOAT}))
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OP_END_FACTORY_REG(Betainc)

REG_OP(Zeta)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
    .INPUT(q, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OP_END_FACTORY_REG(Zeta)

REG_OP(Bucketize)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(boundaries, ListFloat)
    .OP_END_FACTORY_REG(Bucketize)

REG_OP(SparseSegmentSum)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
        DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(segment_ids, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
        DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(SparseSegmentSum)

REG_OP(SparseSegmentMean)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(segment_ids, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseSegmentMean)

REG_OP(SparseSegmentMeanGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(segment_ids, TensorType({DT_INT32}))
    .INPUT(output_dim0, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseSegmentMeanGrad)

REG_OP(IgammaGradA)
    .INPUT(a, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(IgammaGradA)

REG_OP(InitData)
    .ATTR(channel_name, String, "")
    .OP_END_FACTORY_REG(InitData)

REG_OP(GetNext)
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64,
                                        DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes, ListListInt, {})
    .ATTR(output_num, Int, 1)
    .ATTR(channel_name, String, "")
    .OP_END_FACTORY_REG(GetNext)
/**
*@brief: Computes the Gauss error function of `x` element-wise.

*@par Inputs:\n
*x: A Tensor of type float16 or float32.

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(Erf)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Erf)

/**
*@brief: Computes the Gauss complementary error function of "x" element-wise.

*@par Inputs:\n
*x: A Tensor of type float16 or float32.

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(Erfc)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Erfc)

/**
*@brief This operation returns a rank 1 histogram counting the number of entries in `values` \n
*  that fell into every bin.The bins are equal width and determined by the arguments \n
*  'value_range' and 'nbins'. \n

*@par Inputs:
*Three inputs, including: \n
*@li x: A Tensor of type float32,float16,int32.
*@li range: A Tensor of type float32,float16,int32.
*@li nbins: A Tensor of type int32.

*@par Attributes:
* dtype: An optional attribute. Defaults to "int32".

*@par Outputs:
*y: A Tensor. A Tensor of type int32.
*/
REG_OP(HistogramFixedWidth)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(range, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(nbins, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(dtype, String, "int32")
    .OP_END_FACTORY_REG(HistogramFixedWidth)

/**
*@brief This operation returns a rank 1 histogram counting the number of entries in `values` \n
*  that fell into every bin.The bins are equal width and determined by the arguments \n
*  'value_range' and 'nbins'. \n

*@par Inputs:
*Two inputs, including: \n
*@li x: A Tensor of type float32,float16,int32.
*@li range: A Tensor of type float32,float16,int32.

*@par Attributes:
*@li dtype: An optional attribute. Defaults to "int32".
*@li nbins: A required attribute,the type is int32.

*@par Outputs:
*y: A Tensor. A Tensor of type int32.
*/
REG_OP(HistogramFixedWidthD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(range, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(nbins, Int)
    .ATTR(dtype, String, "int32")
    .OP_END_FACTORY_REG(HistogramFixedWidthD)
}  // namespace ge

#endif  // GE_OP_MATH_OPS_H_
