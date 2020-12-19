/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "register/register_format_transfer.h"

namespace ge {
namespace formats {
namespace {
struct FormatTransferRegistry {
  Status RegisterBuilder(Format src, Format dst, FormatTransferBuilder builder) {
    src_dst_builder[src][dst] = std::move(builder);
    return SUCCESS;
  }
  std::map<Format, std::map<Format, FormatTransferBuilder>> src_dst_builder;
};

FormatTransferRegistry &GetFormatTransferRegistry() {
  static FormatTransferRegistry registry;
  return registry;
}
}  // namespace

FormatTransferRegister::FormatTransferRegister(FormatTransferBuilder builder, Format src, Format dst) {
  (void)GetFormatTransferRegistry().RegisterBuilder(src, dst, std::move(builder));
  // RegisterBuilder() always return success, no need to check value
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::shared_ptr<FormatTransfer> BuildFormatTransfer(
    const TransArgs &args) {
  auto registry = GetFormatTransferRegistry();
  auto dst_builder = registry.src_dst_builder.find(args.src_format);
  if (dst_builder == registry.src_dst_builder.end()) {
    return nullptr;
  }
  auto builder_iter = dst_builder->second.find(args.dst_format);
  if (builder_iter == dst_builder->second.end()) {
    return nullptr;
  }
  return builder_iter->second();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool FormatTransferExists(const TransArgs &args) {
  auto registry = GetFormatTransferRegistry();
  auto dst_builder = registry.src_dst_builder.find(args.src_format);
  if (dst_builder == registry.src_dst_builder.end()) {
    return false;
  }
  return dst_builder->second.count(args.dst_format) > 0;
}
}  // namespace formats
}  // namespace ge
