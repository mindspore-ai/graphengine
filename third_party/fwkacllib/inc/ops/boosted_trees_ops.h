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

#ifndef GE_OP_BOOSTED_TREES_OPS_H_
#define GE_OP_BOOSTED_TREES_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Bucketize each feature based on bucket boundaries.

*@par Inputs:
*The input float_values can be 1-D tensor, bucket_boundaries can be 1-D. Inputs include: \n
* @li float_values: List of Rank 1 Tensor each containing float values for a single feature. \n
* @li bucket_boundaries:List of Rank 1 Tensors each containing the bucket boundaries for a single. \n

*@par Attributes:
*@li num_features:number of features \n

*@par Outputs:
*@li y:List of Rank 1 Tensors each containing the bucketized values for a single feature. \n

*@attention Constraints: \n
*-The implementation for BoostedTreesBucketize on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(BoostedTreesBucketize)
    .DYNAMIC_INPUT(float_values, TensorType({DT_FLOAT}))
    .DYNAMIC_INPUT(bucket_boundaries, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(num_features, Int)
    .OP_END_FACTORY_REG(BoostedTreesBucketize)

}  // namespace ge

#endif  // GE_OP_BOOSTED_TREES_OPS_H_
