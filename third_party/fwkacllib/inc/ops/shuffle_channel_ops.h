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

 #ifndef GE_OP_SHUFFLE_CHANNEL_OPS_H
 #define GE_OP_SHUFFLE_CHANNEL_OPS_H

 #include "graph/operator_reg.h"

 namespace ge {

 REG_OP(ShuffleChannel)
       .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
       .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
  	   .ATTR(group, Int, 1)
       .OP_END_FACTORY_REG(ShuffleChannel)
 } // namespace ge

 #endif // GE_OP_SHUFFLE_CHANNEL_OPS_H
