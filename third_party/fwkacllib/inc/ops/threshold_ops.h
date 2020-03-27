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

 #ifndef GE_OP_THRESHOLD_H
 #define GE_OP_THRESHOLD_H

 #include "graph/operator_reg.h"

 namespace ge {

 REG_OP(Threshold)
     .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32}))
     .OUTPUT(output_y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32}))
     .ATTR(threshold, Float, 0.0)
    //  .INFER_SHAPE_AND_TYPE(ThresholdInferShape)
     .OP_END_FACTORY_REG(Threshold);

 } // namespace ge

 #endif // GE_OP_THRESHOLD_OPS_H
