/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "formats/register_format_transfer.h"

#include <map>

namespace ge {
namespace formats {
namespace {
struct FormatTransferRegistry {
  Status RegisterBuilder(const Format src, const Format dst, FormatTransferBuilder builder) {
    src_dst_builder[src][dst] = std::move(builder);
    return SUCCESS;
  }

  std::shared_ptr<FormatTransfer> GenerateFormatTransfer(const Format src, const Format dst) {
    const std::map<Format,
                   std::map<Format, FormatTransferBuilder>>::const_iterator dst_builder = src_dst_builder.find(src);
    if (dst_builder == src_dst_builder.cend()) {
      return nullptr;
    }
    const auto builder_iter = dst_builder->second.find(dst);
    if (builder_iter == dst_builder->second.end()) {
      return nullptr;
    }
    return builder_iter->second();
  }

  bool IsFormatTransferExists(const Format src, const Format dst) {
    const std::map<Format,
                   std::map<Format, FormatTransferBuilder>>::const_iterator dst_builder = src_dst_builder.find(src);
    if (dst_builder == src_dst_builder.cend()) {
      return false;
    }
    return dst_builder->second.count(dst) > 0UL;
  }

private:
  std::map<Format, std::map<Format, FormatTransferBuilder>> src_dst_builder;
};

FormatTransferRegistry &GetFormatTransferRegistry() {
  static FormatTransferRegistry registry;
  return registry;
}
}  // namespace

FormatTransferRegister::FormatTransferRegister(FormatTransferBuilder builder, const Format src,
                                               const Format dst) noexcept {
  (void)GetFormatTransferRegistry().RegisterBuilder(src, dst, std::move(builder));
  // RegisterBuilder() always return success, no need to check value
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::shared_ptr<FormatTransfer> BuildFormatTransfer(
    const TransArgs &args) {
  return GetFormatTransferRegistry().GenerateFormatTransfer(args.src_primary_format, args.dst_primary_format);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool FormatTransferExists(const TransArgs &args) {
  return GetFormatTransferRegistry().IsFormatTransferExists(args.src_primary_format, args.dst_primary_format);
}
}  // namespace formats
}  // namespace ge
