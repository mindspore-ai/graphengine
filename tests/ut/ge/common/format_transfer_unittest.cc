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

#include <gtest/gtest.h>

#include "common/formats/format_transfers/format_transfer_nchw_nc1hwc0.h"

#include "common/formats/format_transfers/format_transfer.h"
#include "common/formats/utils/formats_trans_utils.h"

namespace ge {
namespace formats {

class UtestFormatTransfer : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFormatTransfer, build_transfer_success) {
  uint8_t data[1 * 3 * 224 * 224 * 2];
  TransArgs args{data, FORMAT_NCHW, FORMAT_NC1HWC0, {1, 3, 224, 224}, {1, 1, 224, 224, 16}, DT_FLOAT16};
  auto transfer = BuildFormatTransfer(args);
  EXPECT_NE(transfer, nullptr);
}

TEST_F(UtestFormatTransfer, build_unsupported_transfer) {
  uint8_t data[1 * 3 * 224 * 224 * 2];
  TransArgs args1{data, FORMAT_RESERVED, FORMAT_NCHW, {1, 1, 224, 224, 16}, {1, 3, 224, 224}, DT_FLOAT16};
  auto transfer1 = BuildFormatTransfer(args1);
  EXPECT_EQ(transfer1, nullptr);

  TransArgs args2{data, FORMAT_NCHW, FORMAT_RESERVED, {1, 3, 224, 224}, {1, 1, 224, 224, 16}, DT_FLOAT16};
  auto transfer2 = BuildFormatTransfer(args2);
  EXPECT_EQ(transfer2, nullptr);
}

TEST_F(UtestFormatTransfer, get_size_by_data_type) {
  EXPECT_EQ(GetSizeByDataType(DT_FLOAT), 4);
  EXPECT_EQ(GetSizeByDataType(DT_FLOAT16), 2);
  EXPECT_EQ(GetSizeByDataType(DT_INT8), 1);
  EXPECT_EQ(GetSizeByDataType(DT_INT16), 2);
  EXPECT_EQ(GetSizeByDataType(DT_UINT16), 2);
  EXPECT_EQ(GetSizeByDataType(DT_UINT8), 1);
  EXPECT_EQ(GetSizeByDataType(DT_INT32), 4);
  EXPECT_EQ(GetSizeByDataType(DT_INT64), 8);
  EXPECT_EQ(GetSizeByDataType(DT_UINT32), 4);
  EXPECT_EQ(GetSizeByDataType(DT_UINT64), 8);
  EXPECT_EQ(GetSizeByDataType(DT_BOOL), 1);
  EXPECT_EQ(GetSizeByDataType(DT_DOUBLE), 8);
  EXPECT_EQ(GetSizeByDataType(DT_STRING), -1);
  EXPECT_EQ(GetSizeByDataType(DT_DUAL_SUB_INT8), 1);
  EXPECT_EQ(GetSizeByDataType(DT_DUAL_SUB_UINT8), 1);
  EXPECT_EQ(GetSizeByDataType(DT_COMPLEX64), 8);
  EXPECT_EQ(GetSizeByDataType(DT_COMPLEX128), 16);
  EXPECT_EQ(GetSizeByDataType(DT_QINT8), 1);
  EXPECT_EQ(GetSizeByDataType(DT_QINT16), 2);
  EXPECT_EQ(GetSizeByDataType(DT_QINT32), 4);
  EXPECT_EQ(GetSizeByDataType(DT_QUINT8), 1);
  EXPECT_EQ(GetSizeByDataType(DT_QUINT16), 2);
  EXPECT_EQ(GetSizeByDataType(DT_RESOURCE), -1);
  EXPECT_EQ(GetSizeByDataType(DT_STRING_REF), -1);
  EXPECT_EQ(GetSizeByDataType(DT_DUAL), 5);
  EXPECT_EQ(GetSizeByDataType(DT_UNDEFINED), -1);
  EXPECT_EQ(DT_UNDEFINED, 26);
}
}  // namespace formats
}  // namespace ge