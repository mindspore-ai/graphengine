/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file list_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_LIST_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_LIST_OPS_H_

#include <algorithm>
#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Creates and returns an empty tensor list. \n

*@par Inputs:
*@li element_shape: A shape compatible with that of elements in the list.
*@li max_num_elements: The maximum number of elements. \n

*@par Attributes:
*@li element_dtype: The type of elements in the list. \n

*@par Outputs:
*@li handle: An empty tensor list . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow EmptyTensorList operator.
*/
REG_OP(EmptyTensorList)
    .INPUT(element_shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(max_num_elements, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(EmptyTensorList)

/**
*@brief Returns a list which has the passed-in `Tensor` as last element
and the other elements of the given list in `input_handle`. \n

*@par Inputs:
*@li input_handle: The old list.
*@li tensor: The tensor to put on the list. \n

*@par Attributes:
*@li element_dtype: The type of elements in the list. \n

*@par Outputs:
*@li output_handle:A list with the elements of old list followed by tensor. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListPushBack operator.
*/
REG_OP(TensorListPushBack)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,DT_RESOURCE,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListPushBack)

/**
*@brief The last element of the input list as well as a
list with all but that element. \n

*@par Inputs:
*@li input_handle: The input list.
*@li element_shape: A shape compatible with that of elements in the list. \n

*@par Attributes:
*@li element_dtype: The type of elements in the list. \n

*@par Outputs:
*@li output_handle:A list with the elements of the old list followed by tensor.
*@li tensor:The withdrawn last element of the list. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListPopBack operator.
*/
REG_OP(TensorListPopBack)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(element_shape, TensorType({DT_INT32}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .OUTPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,DT_RESOURCE,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListPopBack)

/**
*@brief The number of tensors in the input tensor list. \n

*@par Inputs:
*@li input_handle: The input list. \n

*@par Outputs:
*@li length:The number of tensors in the list. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListLength operator.
*/
REG_OP(TensorListLength)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .OUTPUT(length, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(TensorListLength)

/**
*@brief The shape of elements in the input tensor list. \n

*@par Inputs:
*@li input_handle: The input list. \n

*@par Attributes:
*@li shape_type: The type of shape in the list. \n

*@par Outputs:
*@li element_shape:A shape compatible with that of elements in the list. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListElementShape operator.
*/
REG_OP(TensorListElementShape)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .OUTPUT(element_shape, TensorType({DT_INT32,DT_INT64}))
    .ATTR(shape_type, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListElementShape)

/**
*@brief List of the given size with empty elements. \n

*@par Inputs:
*@li element_shape: A shape compatible with that of elements in the list.
*@li num_elements: The number of elements to reserve. \n

*@par Attributes:
*@li element_dtype: The type of elements in the list.
*@li shape_type: The type of shape in the list. \n

*@par Outputs:
*@li handle: An output tensor list . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListReserve operator.
*/
REG_OP(TensorListReserve)
    .INPUT(element_shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(num_elements, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .ATTR(shape_type, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListReserve)

/**
*@brief Get input tensor list elements of index position. \n

*@par Inputs:
*@li input_handle: The input list.
*@li index: A tensor of position.
*@li element_shape: A shape compatible with that of elements in the list. \n

*@par Attributes:
*@li element_dtype: The type of elements in the list. \n

*@par Outputs:
*@li item: An output tensor value of index position . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListGetItem operator.
*/
REG_OP(TensorListGetItem)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(element_shape, TensorType({DT_INT32}))
    .OUTPUT(item, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListGetItem)

/**
*@brief Sets the index-th position of the list to contain the given tensor. \n

*@par Inputs:
*@li input_handle: The input list.
*@li index: The position in the list to which the tensor will be assigned.
*@li item: The element to be assigned to that position. \n

*@par Attributes:
*@li element_dtype: The type of elements in the list. \n

*@par Outputs:
*@li output_handle: An output tensor list . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListSetItem operator.
*/
REG_OP(TensorListSetItem)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(item, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,DT_RESOURCE,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListSetItem)

}   // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_LIST_OPS_H_
