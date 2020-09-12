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

/*!
 * \file ctc_ops.h
 * \brief
 */
#ifndef GE_OP_CTC_OPS_H
#define GE_OP_CTC_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Calculates the CTC Loss (log probability) for each batch entry. \n
Also calculates the gradient. 

*@par Inputs:
*@li inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
*@li labels_indices: The indices of a `SparseTensor<int32, 2>`. \n
`labels_indices(i, :) == [b, t]` means `labels_values(i)` stores the id for \n
`(batch b, time t)`.
*@li labels_values: The values (labels) associated with the given batch and time.
*@li sequence_length: A vector containing sequence lengths (batch).

*@par Outputs:
*@li loss: A vector (batch) containing log-probabilities.
*@li gradient: The gradient of `loss`.  3-D, shape: `(max_time x \n
batch_size x num_classes)`.

*@par Attributes:
*@li preprocess_collapse_repeated: Scalar, if true then repeated labels are collapsed prior to \n
the CTC calculation.If not specified, defaults to false
*@li ctc_merge_repeated: Scalar. If set to false, *during* CTC calculation \n
repeated non-blank labels will not be merged and are interpreted as \n
individual labels.  This is a simplified version of CTC. \n
If not specified, defaults to true

*@par Third-party framework compatibility
* Compatible with TensorFlow CTCLoss operator.
*/
REG_OP(CTCLoss)
    .INPUT(inputs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(labels_indices, TensorType({DT_INT64}))
    .INPUT(labels_values, TensorType({DT_INT32}))
    .INPUT(sequence_length, TensorType({DT_INT32}))
    .OUTPUT(loss, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(gradient, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(preprocess_collapse_repeated, Bool, false)
    .ATTR(ctc_merge_repeated, Bool, true)
    .ATTR(ignore_longer_outputs_than_inputs, Bool, false)
    .OP_END_FACTORY_REG(CTCLoss)

/**
*@brief Performs greedy decoding on the logits given in inputs.

*@par Inputs:
*@li inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
*@li sequence_length: A vector containing sequence lengths, size `(batch_size)`.

*@par Attributes:
*@li merge_repeated: If True, merge repeated classes in output.

*@par Outputs:
*@li decoded_indices: Indices matrix, size `(total_decoded_outputs x 2)`,\n
of a `SparseTensor<int64, 2>`.  The rows store: [batch, time].
*@li decoded_values: Values vector, size: `(total_decoded_outputs)`,\n
of a `SparseTensor<int64, 2>`.  The vector stores the decoded classes.
*@li decoded_shape: Shape vector, size `(2)`, of the decoded SparseTensor.\n
Values are: `[batch_size, max_decoded_length]`.
*@li log_probability: Matrix, size `(batch_size x 1)`, containing sequence\n
log-probabilities.

*@par Third-party framework compatibility
* Compatible with TensorFlow CTCGreedyDecoder operator.
*/
REG_OP(CTCGreedyDecoder)
    .INPUT(inputs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(sequence_length, TensorType({DT_INT32}))
    .ATTR(merge_repeated, Bool, false)
    .OUTPUT(decoded_indices, TensorType({DT_INT64}))
    .OUTPUT(decoded_values, TensorType({DT_INT64}))
    .OUTPUT(decoded_shape, TensorType({DT_INT64}))
    .OUTPUT(log_probability, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(CTCGreedyDecoder)

/**
*@brief Performs beam search decoding on the logits given in input.

*@par Inputs:
*@li inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
*@li sequence_length: A vector containing sequence lengths, size `(batch_size)`.

*@par Attributes:
*@li merge_repeated: If True, merge repeated classes in output.

*@par Outputs:
*@li decoded_indices: A list (length: top_paths) of indices matrices.  Matrix j,\n
size `(total_decoded_outputs[j] x 2)`, has indices of a\n
`SparseTensor<int64, 2>`.  The rows store: [batch, time].
*@li decoded_values: A list (length: top_paths) of values vectors.  Vector j,\n
size `(length total_decoded_outputs[j])`, has the values of a\n
`SparseTensor<int64, 2>`.  The vector stores the decoded classes for beam j.
*@li decoded_shape: A list (length: top_paths) of shape vector.  Vector j,\n
size `(2)`, stores the shape of the decoded `SparseTensor[j]`.\n
Its values are: `[batch_size, max_decoded_length[j]]`.
*@li log_probability: A matrix, shaped: `(batch_size x top_paths)`.  The\n
sequence log-probabilities.

*@par Third-party framework compatibility
* Compatible with TensorFlow CTCBeamSearchDecoder operator.
*/
REG_OP(CTCBeamSearchDecoder)
    .INPUT(inputs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(sequence_length, TensorType({DT_INT32}))
    .REQUIRED_ATTR(beam_width, Int)
    .REQUIRED_ATTR(top_paths, Int)
    .ATTR(merge_repeated, Bool, true)
    .DYNAMIC_OUTPUT(decoded_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(decoded_values, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(decoded_shape, TensorType({DT_INT64}))
    .OUTPUT(log_probability, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(CTCBeamSearchDecoder)

}  // namespace ge

#endif //GE_OP_CTC_OPS_H