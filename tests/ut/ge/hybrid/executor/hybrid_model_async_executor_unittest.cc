/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <gmock/gmock.h>
#include <vector>

#define private public
#define protected public
#include "hybrid/executor/hybrid_model_async_executor.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"


using namespace std;
using namespace testing;


namespace ge {
using namespace hybrid;

class UtestHybridModelAsyncExecutor : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() { }
};

TEST_F(UtestHybridModelAsyncExecutor, CopyOutputs_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelAsyncExecutor executor(&hybrid_model);
  
  TensorValue input_tensor;
  HybridModelExecutor::ExecuteArgs args;
  args.inputs.emplace_back(input_tensor);
  auto desc = MakeShared<GeTensorDesc>();
  GeShape geshape({2,2,2,2});
  desc->SetShape(geshape);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto output_tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  args.outputs.emplace_back(output_tensor);
  args.output_desc.emplace_back(desc);
  
  OutputData output_data;
  std::vector<ge::Tensor> outputs;
  auto ret = executor.CopyOutputs(args, &output_data, outputs);
  ASSERT_EQ(ret,SUCCESS);
}

TEST_F(UtestHybridModelAsyncExecutor, BuildDeviceTensor) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelAsyncExecutor executor(&hybrid_model);
  
  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  GeTensorDesc ge_tensor_desc;
  int64_t output_size = 100;
  std::vector<ge::Tensor> outputs;
  executor.BuildDeviceTensor(tensor, ge_tensor_desc, output_size, outputs);
  auto size = tensor.GetSize();
  ASSERT_EQ(size, 100);
}
} // namespace ge