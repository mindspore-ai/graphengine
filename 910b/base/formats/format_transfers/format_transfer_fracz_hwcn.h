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

#ifndef GE_COMMON_FORMATS_FORMAT_TRANSFERS_FORMAT_TRANSFER_FRACZ_HWCN_H_
#define GE_COMMON_FORMATS_FORMAT_TRANSFERS_FORMAT_TRANSFER_FRACZ_HWCN_H_

#include <vector>
#include "formats/register_format_transfer.h"

namespace ge {
namespace formats {
class FormatTransferFracZHwcn : public FormatTransfer {
 public:
  Status TransFormat(const TransArgs &args, TransResult &result) override;
  Status TransShape(const Format src_format, const std::vector<int64_t> &src_shape,
                    const DataType data_type, const Format dst_format,
                    std::vector<int64_t> &dst_shape) override;
};
}  // namespace formats
}  // namespace ge
#endif  // GE_COMMON_FORMATS_FORMAT_TRANSFERS_FORMAT_TRANSFER_FRACZ_HWCN_H_
