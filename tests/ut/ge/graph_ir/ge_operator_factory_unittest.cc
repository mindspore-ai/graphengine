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

#include "../graph/ops_stub.h"
#include "operator_factory.h"

#define protected public
#define private public
#include "operator_factory_impl.h"
#undef private
#undef protected

using namespace ge;
class UTEST_GE_OPERATOR_FACTORY : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST(UTEST_GE_OPERATOR_FACTORY, CreateOperator) {
  Operator acosh = OperatorFactory::CreateOperator("acosh", "Acosh");
  EXPECT_EQ("Acosh", acosh.GetOpType());
  EXPECT_EQ("acosh", acosh.GetName());
  EXPECT_EQ(false, acosh.IsEmpty());
}

TEST(UTEST_GE_OPERATOR_FACTORY, CreateOperatorNullPtr) {
  Operator abc = OperatorFactory::CreateOperator("abc", "ABC");
  EXPECT_EQ(true, abc.IsEmpty());
}

TEST(UTEST_GE_OPERATOR_FACTORY, GetInferShapeFunc) {
  OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  InferShapeFunc inferShapeFunc = OperatorFactoryImpl::GetInferShapeFunc("ABC");
  EXPECT_EQ(nullptr, inferShapeFunc);
}

TEST(UTEST_GE_OPERATOR_FACTORY, GetVerifyFunc) {
  OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  VerifyFunc verifyFunc = OperatorFactoryImpl::GetVerifyFunc("ABC");
  EXPECT_EQ(nullptr, verifyFunc);
}

TEST(UTEST_GE_OPERATOR_FACTORY, GetOpsTypeList) {
  std::vector<std::string> allOps;
  graphStatus status = OperatorFactory::GetOpsTypeList(allOps);
  EXPECT_NE(0, allOps.size());
  EXPECT_EQ(GRAPH_SUCCESS, status);
}

TEST(UTEST_GE_OPERATOR_FACTORY, IsExistOp) {
  graphStatus status = OperatorFactory::IsExistOp("Acosh");
  EXPECT_EQ(true, status);
  status = OperatorFactory::IsExistOp("ABC");
  EXPECT_EQ(false, status);
}

TEST(UTEST_GE_OPERATOR_FACTORY, RegisterFunc) {
  OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  graphStatus status = OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  EXPECT_EQ(GRAPH_FAILED, status);
  status = OperatorFactoryImpl::RegisterInferShapeFunc("ABC", nullptr);
  EXPECT_EQ(GRAPH_SUCCESS, status);

  OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  status = OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  EXPECT_EQ(GRAPH_FAILED, status);
  status = OperatorFactoryImpl::RegisterVerifyFunc("ABC", nullptr);
  EXPECT_EQ(GRAPH_SUCCESS, status);
}

TEST(UTEST_GE_OPERATOR_FACTORY, GetOpsTypeListFail) {
  auto OperatorCreatorsTemp = OperatorFactoryImpl::operator_creators_;
  OperatorFactoryImpl::operator_creators_ = nullptr;
  std::vector<std::string> allOps;
  graphStatus status = OperatorFactoryImpl::GetOpsTypeList(allOps);
  EXPECT_EQ(GRAPH_FAILED, status);
  OperatorFactoryImpl::operator_creators_ = OperatorCreatorsTemp;
}