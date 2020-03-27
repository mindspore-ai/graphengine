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

#ifndef GE_OP_STRING_OPS_H_
#define GE_OP_STRING_OPS_H_

#include <sstream>
#include "graph/operator_reg.h"

namespace ge {
REG_OP(StringSplit)
    .INPUT(input, TensorType({DT_STRING}))
    .INPUT(delimiter, TensorType({DT_STRING}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_STRING}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .ATTR(skip_empty, Bool, true)
    .OP_END_FACTORY_REG(StringSplit)

REG_OP(StringSplitV2)
    .INPUT(input, TensorType({DT_STRING}))
    .INPUT(sep, TensorType({DT_STRING}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_STRING}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .ATTR(maxsplit, Int, -1)
    .OP_END_FACTORY_REG(StringSplitV2)

REG_OP(UnicodeScript)
    .INPUT(x, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(UnicodeScript)

REG_OP(Substr)
    .INPUT(input, TensorType({DT_STRING}))
    .INPUT(pos, TensorType({DT_INT32, DT_INT64}))
    .INPUT(len, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(output, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(Substr)

REG_OP(StringToHashBucketFast)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .ATTR(num_buckets, Int, 1)
    .OP_END_FACTORY_REG(StringToHashBucketFast)

REG_OP(StringToHashBucketStrong)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .ATTR(num_buckets, Int, 1)
    .REQUIRED_ATTR(key, ListInt)
    .OP_END_FACTORY_REG(StringToHashBucketStrong)

REG_OP(StringToHashBucket)
    .INPUT(string_tensor, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .ATTR(num_buckets, Int, 1)
    .OP_END_FACTORY_REG(StringToHashBucket)

REG_OP(StringStrip)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(StringStrip)

REG_OP(StringLength)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(unit, String, "BYTE")
    .OP_END_FACTORY_REG(StringLength)

REG_OP(StringJoin)
    .DYNAMIC_INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .REQUIRED_ATTR(N, Int)
    .ATTR(separator, String, "")
    .OP_END_FACTORY_REG(StringJoin)

REG_OP(StringFormat)
    .DYNAMIC_INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_STRING, DT_FLOAT16, \
        DT_FLOAT, DT_DOUBLE, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .ATTR(template, String, "%s")
    .ATTR(placeholder, String, "%s")
    .ATTR(summarize, Int, 3)
    .OP_END_FACTORY_REG(StringFormat)

REG_OP(RegexFullMatch)
    .INPUT(x, TensorType({DT_STRING}))
    .INPUT(pattern, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(RegexFullMatch)

REG_OP(RegexReplace)
    .INPUT(x, TensorType({DT_STRING}))
    .INPUT(pattern, TensorType({DT_STRING}))
    .INPUT(rewrite, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .ATTR(replace_global, Bool, true)
    .OP_END_FACTORY_REG(RegexReplace)

REG_OP(AsString)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT, \
        DT_DOUBLE, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .ATTR(precision, Int, -1)
    .ATTR(scientific, Bool, false)
    .ATTR(shortest, Bool, false)
    .ATTR(width, Int, -1)
    .ATTR(fill, String, "")
    .OP_END_FACTORY_REG(AsString)

REG_OP(EncodeBase64)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .ATTR(pad, Bool, false)
    .OP_END_FACTORY_REG(EncodeBase64)

REG_OP(DecodeBase64)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(DecodeBase64)
}  // namespace ge

#endif  // GE_OP_STRING_OPS_H_
