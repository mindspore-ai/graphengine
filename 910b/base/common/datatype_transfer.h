/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef GE_COMMON_FORMATS_FORMAT_TRANSFERS_DATATYPE_TRANSFER_H_
#define GE_COMMON_FORMATS_FORMAT_TRANSFERS_DATATYPE_TRANSFER_H_

#include "formats/register_format_transfer.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
namespace formats {
class DataTypeTransfer {
 public:
  static bool DataTypeTransferExists(const CastArgs &args);
  static Status TransDataType(const CastArgs &args, TransResult &result);
};

bool IsTransDataTypeSupport(const CastArgs &args);

Status TransTensorDataType(const CastArgs &args, TransResult &result);
}  // namespace formats
}  // namespace ge

#endif  // GE_COMMON_FORMATS_FORMAT_TRANSFERS_DATATYPE_TRANSFER_H_
