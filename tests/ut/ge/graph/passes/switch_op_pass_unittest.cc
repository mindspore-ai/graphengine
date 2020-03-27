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

#include "omg/omg_inner_types.h"
#define protected public
#define private public
#include "graph/passes/switch_op_pass.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/attr_value_util.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/graph.h"
#include "graph/passes/control_op_attr_pass.h"
#include "inc/pass_manager.h"
#undef protected
#undef private

using namespace domi;
using namespace testing;
using namespace ge;

class UTEST_graph_passes_switch_op_pass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
  void make_graph(ComputeGraphPtr graph, bool match = true) {
    GeTensorDesc boolTensorDesc(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
    GeTensorDesc intTensorDesc(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc scalarTensorDesc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

    auto xOpDef = std::make_shared<OpDesc>("x", VARIABLEV2);
    xOpDef->AddOutputDesc(scalarTensorDesc);
    auto xNode = graph->AddNode(xOpDef);

    auto yOpDef = std::make_shared<OpDesc>("y", VARIABLEV2);
    yOpDef->AddOutputDesc(scalarTensorDesc);
    auto yNode = graph->AddNode(yOpDef);

    auto zOpDef = std::make_shared<OpDesc>("z", VARIABLEV2);
    zOpDef->AddOutputDesc(scalarTensorDesc);
    auto zNode = graph->AddNode(zOpDef);

    auto condOpDef = std::make_shared<OpDesc>("Less", "Less");
    condOpDef->AddInputDesc(scalarTensorDesc);
    condOpDef->AddInputDesc(scalarTensorDesc);
    condOpDef->AddOutputDesc(boolTensorDesc);
    auto condNode = graph->AddNode(condOpDef);

    auto switchOpDef1 = std::make_shared<OpDesc>("Add/Switch", SWITCH);
    switchOpDef1->AddInputDesc(scalarTensorDesc);
    switchOpDef1->AddInputDesc(boolTensorDesc);
    switchOpDef1->AddOutputDesc(scalarTensorDesc);
    switchOpDef1->AddOutputDesc(scalarTensorDesc);
    auto switchNode1 = graph->AddNode(switchOpDef1);

    auto switchOpDef2 = std::make_shared<OpDesc>("Add/Switch_1", SWITCH);
    switchOpDef2->AddInputDesc(scalarTensorDesc);
    switchOpDef2->AddInputDesc(boolTensorDesc);
    switchOpDef2->AddOutputDesc(scalarTensorDesc);
    switchOpDef2->AddOutputDesc(scalarTensorDesc);
    auto switchNode2 = graph->AddNode(switchOpDef2);

    auto switchOpDef3 = std::make_shared<OpDesc>("Square/Switch", SWITCH);
    switchOpDef3->AddInputDesc(scalarTensorDesc);
    switchOpDef3->AddInputDesc(boolTensorDesc);
    switchOpDef3->AddOutputDesc(scalarTensorDesc);
    switchOpDef3->AddOutputDesc(scalarTensorDesc);
    auto switchNode3 = graph->AddNode(switchOpDef3);

    auto addOpDef = std::make_shared<OpDesc>("Add", "ADD");
    addOpDef->AddInputDesc(scalarTensorDesc);
    addOpDef->AddInputDesc(scalarTensorDesc);
    addOpDef->AddOutputDesc(scalarTensorDesc);
    auto addNode = graph->AddNode(addOpDef);

    auto mergeOpDef = std::make_shared<OpDesc>("Merge", "Merge");
    mergeOpDef->AddInputDesc(scalarTensorDesc);
    mergeOpDef->AddInputDesc(scalarTensorDesc);
    mergeOpDef->AddOutputDesc(scalarTensorDesc);
    mergeOpDef->AddOutputDesc(intTensorDesc);
    auto mergeNode = graph->AddNode(mergeOpDef);

    auto outputOpDef = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    outputOpDef->AddInputDesc(scalarTensorDesc);
    outputOpDef->AddOutputDesc(scalarTensorDesc);
    auto outputNode = graph->AddNode(outputOpDef);

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), switchNode1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switchNode1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), switchNode2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switchNode2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), switchNode3->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switchNode3->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switchNode1->GetOutDataAnchor(1), addNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switchNode2->GetOutDataAnchor(1), addNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(addNode->GetOutDataAnchor(0), mergeNode->GetInDataAnchor(1));
    (void)GraphUtils::AddEdge(switchNode3->GetOutDataAnchor(0), mergeNode->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(mergeNode->GetOutDataAnchor(0), outputNode->GetInDataAnchor(0));
  }

  void make_graph_const(ComputeGraphPtr graph, bool match = true) {
    // resnet50 PolynomialDecay
    GeTensorDesc scalarTensorDesc(GeShape({1, 1, 1, 1}));
    GeTensorDesc boolTensorDesc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_BOOL);
    GeTensorDesc intTensorDesc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_INT32);

    auto xOpDef = std::make_shared<OpDesc>("x", VARIABLEV2);
    xOpDef->AddOutputDesc(scalarTensorDesc);
    auto xNode = graph->AddNode(xOpDef);

    auto yOpDef = std::make_shared<OpDesc>("y", "Const");
    yOpDef->AddOutputDesc(scalarTensorDesc);
    auto yNode = graph->AddNode(yOpDef);

    auto zOpDef = std::make_shared<OpDesc>("z", VARIABLEV2);
    zOpDef->AddOutputDesc(scalarTensorDesc);
    auto zNode = graph->AddNode(zOpDef);

    auto constOpDef = std::make_shared<OpDesc>("Const", "Const");
    constOpDef->AddOutputDesc(scalarTensorDesc);
    auto constNode = graph->AddNode(constOpDef);

    auto condOpDef = std::make_shared<OpDesc>("Equal", "Equal");
    condOpDef->AddInputDesc(scalarTensorDesc);
    condOpDef->AddInputDesc(scalarTensorDesc);
    condOpDef->AddOutputDesc(boolTensorDesc);
    auto condNode = graph->AddNode(condOpDef);

    auto identityOpDef = std::make_shared<OpDesc>("identity", "Identity");
    identityOpDef->AddInputDesc(boolTensorDesc);
    identityOpDef->AddOutputDesc(boolTensorDesc);
    auto identityNode = graph->AddNode(identityOpDef);

    auto switchOpDef1 = std::make_shared<OpDesc>("Switch", SWITCH);
    switchOpDef1->AddInputDesc(boolTensorDesc);
    switchOpDef1->AddInputDesc(boolTensorDesc);
    switchOpDef1->AddOutputDesc(boolTensorDesc);
    switchOpDef1->AddOutputDesc(boolTensorDesc);
    auto switchNode1 = graph->AddNode(switchOpDef1);

    auto tIdentityOpDef = std::make_shared<OpDesc>("switch_t", "Identity");
    tIdentityOpDef->AddInputDesc(scalarTensorDesc);
    tIdentityOpDef->AddOutputDesc(scalarTensorDesc);
    auto tIdentityNode = graph->AddNode(tIdentityOpDef);

    auto fIdentityOpDef = std::make_shared<OpDesc>("switch_f", "Identity");
    fIdentityOpDef->AddInputDesc(scalarTensorDesc);
    fIdentityOpDef->AddOutputDesc(scalarTensorDesc);
    auto fIdentityNode = graph->AddNode(fIdentityOpDef);

    auto switchOpDef2 = std::make_shared<OpDesc>("Switch_1", SWITCH);
    switchOpDef2->AddInputDesc(scalarTensorDesc);
    switchOpDef2->AddInputDesc(boolTensorDesc);
    switchOpDef2->AddOutputDesc(scalarTensorDesc);
    switchOpDef2->AddOutputDesc(scalarTensorDesc);
    auto switchNode2 = graph->AddNode(switchOpDef2);

    auto mulOpDef = std::make_shared<OpDesc>("truediv", "Mul");
    mulOpDef->AddInputDesc(scalarTensorDesc);
    mulOpDef->AddInputDesc(scalarTensorDesc);
    mulOpDef->AddOutputDesc(scalarTensorDesc);
    auto mulNode = graph->AddNode(mulOpDef);

    auto ceilOpDef = std::make_shared<OpDesc>("Ceil", "Ceil");
    ceilOpDef->AddInputDesc(scalarTensorDesc);
    ceilOpDef->AddOutputDesc(scalarTensorDesc);
    auto ceilNode = graph->AddNode(ceilOpDef);

    auto mergeOpDef = std::make_shared<OpDesc>("Merge", "Merge");
    mergeOpDef->AddInputDesc(scalarTensorDesc);
    mergeOpDef->AddInputDesc(scalarTensorDesc);
    mergeOpDef->AddOutputDesc(scalarTensorDesc);
    mergeOpDef->AddOutputDesc(intTensorDesc);
    auto mergeNode = graph->AddNode(mergeOpDef);

    auto outputOpDef = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    outputOpDef->AddInputDesc(scalarTensorDesc);
    outputOpDef->AddOutputDesc(scalarTensorDesc);
    auto outputNode = graph->AddNode(outputOpDef);

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), identityNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(identityNode->GetOutDataAnchor(0), switchNode1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(identityNode->GetOutDataAnchor(0), switchNode1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switchNode1->GetOutDataAnchor(0), fIdentityNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switchNode1->GetOutDataAnchor(1), tIdentityNode->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(fIdentityNode->GetOutControlAnchor(), zNode->GetInControlAnchor());
    (void)GraphUtils::AddEdge(tIdentityNode->GetOutControlAnchor(), constNode->GetInControlAnchor());

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), switchNode2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(identityNode->GetOutDataAnchor(0), switchNode2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), mulNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switchNode2->GetOutDataAnchor(0), mulNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(mulNode->GetOutDataAnchor(0), ceilNode->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(constNode->GetOutDataAnchor(0), mergeNode->GetInDataAnchor(1));
    (void)GraphUtils::AddEdge(ceilNode->GetOutDataAnchor(0), mergeNode->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(mergeNode->GetOutDataAnchor(0), outputNode->GetInDataAnchor(0));
  }

  void make_graph_cyclic_dependence(ComputeGraphPtr graph, bool match = true) {
    GeTensorDesc scalarTensorDesc(GeShape({1, 1, 1, 1}));
    GeTensorDesc boolTensorDesc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_BOOL);
    GeTensorDesc intTensorDesc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_INT32);

    auto xOpDef = std::make_shared<OpDesc>("x", VARIABLEV2);
    xOpDef->AddOutputDesc(scalarTensorDesc);
    auto xNode = graph->AddNode(xOpDef);

    auto yOpDef = std::make_shared<OpDesc>("y", VARIABLEV2);
    yOpDef->AddOutputDesc(scalarTensorDesc);
    auto yNode = graph->AddNode(yOpDef);

    auto zOpDef = std::make_shared<OpDesc>("z", VARIABLEV2);
    zOpDef->AddOutputDesc(scalarTensorDesc);
    auto zNode = graph->AddNode(zOpDef);

    auto condOpDef = std::make_shared<OpDesc>("Less", "Less");
    condOpDef->AddInputDesc(scalarTensorDesc);
    condOpDef->AddInputDesc(scalarTensorDesc);
    condOpDef->AddOutputDesc(boolTensorDesc);
    auto condNode = graph->AddNode(condOpDef);

    auto switchOpDef1 = std::make_shared<OpDesc>("Switch_f_1", SWITCH);
    switchOpDef1->AddInputDesc(scalarTensorDesc);
    switchOpDef1->AddInputDesc(boolTensorDesc);
    switchOpDef1->AddOutputDesc(scalarTensorDesc);
    switchOpDef1->AddOutputDesc(scalarTensorDesc);
    auto switchNode1 = graph->AddNode(switchOpDef1);

    auto switchOpDef2 = std::make_shared<OpDesc>("Switch_t_1", SWITCH);
    switchOpDef2->AddInputDesc(scalarTensorDesc);
    switchOpDef2->AddInputDesc(boolTensorDesc);
    switchOpDef2->AddOutputDesc(scalarTensorDesc);
    switchOpDef2->AddOutputDesc(scalarTensorDesc);
    auto switchNode2 = graph->AddNode(switchOpDef2);

    auto switchOpDef3 = std::make_shared<OpDesc>("Switch_f_2", SWITCH);
    switchOpDef3->AddInputDesc(scalarTensorDesc);
    switchOpDef3->AddInputDesc(boolTensorDesc);
    switchOpDef3->AddOutputDesc(scalarTensorDesc);
    switchOpDef3->AddOutputDesc(scalarTensorDesc);
    auto switchNode3 = graph->AddNode(switchOpDef3);

    auto switchOpDef4 = std::make_shared<OpDesc>("Switch_t_2", SWITCH);
    switchOpDef4->AddInputDesc(scalarTensorDesc);
    switchOpDef4->AddInputDesc(boolTensorDesc);
    switchOpDef4->AddOutputDesc(scalarTensorDesc);
    switchOpDef4->AddOutputDesc(scalarTensorDesc);
    auto switchNode4 = graph->AddNode(switchOpDef4);

    auto squareOpDef1 = std::make_shared<OpDesc>("Square1", "Square");
    squareOpDef1->AddInputDesc(scalarTensorDesc);
    squareOpDef1->AddOutputDesc(scalarTensorDesc);
    auto squareNode1 = graph->AddNode(squareOpDef1);

    auto squareOpDef2 = std::make_shared<OpDesc>("Square2", "Square");
    squareOpDef2->AddInputDesc(scalarTensorDesc);
    squareOpDef2->AddOutputDesc(scalarTensorDesc);
    auto squareNode2 = graph->AddNode(squareOpDef2);

    auto squareOpDef3 = std::make_shared<OpDesc>("Square3", "Square");
    squareOpDef3->AddInputDesc(scalarTensorDesc);
    squareOpDef3->AddOutputDesc(scalarTensorDesc);
    auto squareNode3 = graph->AddNode(squareOpDef3);

    auto squareOpDef4 = std::make_shared<OpDesc>("Square4", "Square");
    squareOpDef4->AddInputDesc(scalarTensorDesc);
    squareOpDef4->AddOutputDesc(scalarTensorDesc);
    auto squareNode4 = graph->AddNode(squareOpDef4);

    auto mergeOpDef1 = std::make_shared<OpDesc>("Merge1", "Merge");
    mergeOpDef1->AddInputDesc(scalarTensorDesc);
    mergeOpDef1->AddInputDesc(scalarTensorDesc);
    mergeOpDef1->AddOutputDesc(scalarTensorDesc);
    mergeOpDef1->AddOutputDesc(intTensorDesc);
    auto mergeNode1 = graph->AddNode(mergeOpDef1);

    auto mergeOpDef2 = std::make_shared<OpDesc>("Merge2", "Merge");
    mergeOpDef2->AddInputDesc(scalarTensorDesc);
    mergeOpDef2->AddInputDesc(scalarTensorDesc);
    mergeOpDef2->AddOutputDesc(scalarTensorDesc);
    mergeOpDef2->AddOutputDesc(intTensorDesc);
    auto mergeNode2 = graph->AddNode(mergeOpDef2);

    auto outputOpDef = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    outputOpDef->AddInputDesc(scalarTensorDesc);
    outputOpDef->AddOutputDesc(scalarTensorDesc);
    auto outputNode = graph->AddNode(outputOpDef);

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), switchNode1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switchNode1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), switchNode2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switchNode2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switchNode1->GetOutDataAnchor(0), squareNode1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switchNode2->GetOutDataAnchor(1), squareNode2->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(squareNode1->GetOutDataAnchor(0), mergeNode1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(squareNode2->GetOutDataAnchor(0), mergeNode1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(mergeNode1->GetOutDataAnchor(0), switchNode3->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switchNode3->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), switchNode4->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switchNode4->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switchNode3->GetOutDataAnchor(0), squareNode3->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switchNode4->GetOutDataAnchor(1), squareNode4->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(squareNode3->GetOutDataAnchor(0), mergeNode2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(squareNode4->GetOutDataAnchor(0), mergeNode2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(mergeNode2->GetOutDataAnchor(0), outputNode->GetInDataAnchor(0));
  }

  void make_graph_case(ComputeGraphPtr graph, bool match = true) {
    GeTensorDesc scalarTensorDesc(GeShape({1, 1, 1, 1}));
    GeTensorDesc boolTensorDesc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_BOOL);
    GeTensorDesc intTensorDesc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_INT32);

    auto xOpDef = std::make_shared<OpDesc>("x", VARIABLEV2);
    xOpDef->AddOutputDesc(scalarTensorDesc);
    auto xNode = graph->AddNode(xOpDef);

    auto yOpDef = std::make_shared<OpDesc>("y", VARIABLEV2);
    yOpDef->AddOutputDesc(scalarTensorDesc);
    auto yNode = graph->AddNode(yOpDef);

    auto zOpDef = std::make_shared<OpDesc>("z", VARIABLEV2);
    zOpDef->AddOutputDesc(scalarTensorDesc);
    auto zNode = graph->AddNode(zOpDef);

    auto greaterOpDef = std::make_shared<OpDesc>("Greater", "Greater");
    greaterOpDef->AddInputDesc(scalarTensorDesc);
    greaterOpDef->AddInputDesc(scalarTensorDesc);
    greaterOpDef->AddOutputDesc(boolTensorDesc);
    auto greaterNode = graph->AddNode(greaterOpDef);

    auto lessOpDef = std::make_shared<OpDesc>("Less", "Less");
    lessOpDef->AddInputDesc(scalarTensorDesc);
    lessOpDef->AddInputDesc(scalarTensorDesc);
    lessOpDef->AddOutputDesc(boolTensorDesc);
    auto lessNode = graph->AddNode(lessOpDef);

    auto switchOpDef1 = std::make_shared<OpDesc>("greater/Switch_t", SWITCH);
    switchOpDef1->AddInputDesc(boolTensorDesc);
    switchOpDef1->AddInputDesc(boolTensorDesc);
    switchOpDef1->AddOutputDesc(boolTensorDesc);
    switchOpDef1->AddOutputDesc(boolTensorDesc);
    auto switchNode1 = graph->AddNode(switchOpDef1);

    auto switchOpDef2 = std::make_shared<OpDesc>("greater/Switch_f", SWITCH);
    switchOpDef2->AddInputDesc(scalarTensorDesc);
    switchOpDef2->AddInputDesc(boolTensorDesc);
    switchOpDef2->AddOutputDesc(scalarTensorDesc);
    switchOpDef2->AddOutputDesc(scalarTensorDesc);
    auto switchNode2 = graph->AddNode(switchOpDef2);

    auto switchOpDef3 = std::make_shared<OpDesc>("less/Switch_t", SWITCH);
    switchOpDef3->AddInputDesc(scalarTensorDesc);
    switchOpDef3->AddInputDesc(boolTensorDesc);
    switchOpDef3->AddOutputDesc(scalarTensorDesc);
    switchOpDef3->AddOutputDesc(scalarTensorDesc);
    auto switchNode3 = graph->AddNode(switchOpDef3);

    auto switchOpDef4 = std::make_shared<OpDesc>("less/Switch_f", SWITCH);
    switchOpDef4->AddInputDesc(scalarTensorDesc);
    switchOpDef4->AddInputDesc(boolTensorDesc);
    switchOpDef4->AddOutputDesc(scalarTensorDesc);
    switchOpDef4->AddOutputDesc(scalarTensorDesc);
    auto switchNode4 = graph->AddNode(switchOpDef4);

    auto mergeOpDef1 = std::make_shared<OpDesc>("Merge1", "Merge");
    mergeOpDef1->AddInputDesc(scalarTensorDesc);
    mergeOpDef1->AddInputDesc(scalarTensorDesc);
    mergeOpDef1->AddOutputDesc(scalarTensorDesc);
    mergeOpDef1->AddOutputDesc(intTensorDesc);
    auto mergeNode1 = graph->AddNode(mergeOpDef1);

    auto mergeOpDef2 = std::make_shared<OpDesc>("Merge2", "Merge");
    mergeOpDef2->AddInputDesc(scalarTensorDesc);
    mergeOpDef2->AddInputDesc(scalarTensorDesc);
    mergeOpDef2->AddOutputDesc(scalarTensorDesc);
    mergeOpDef2->AddOutputDesc(intTensorDesc);
    auto mergeNode2 = graph->AddNode(mergeOpDef2);

    auto outputOpDef = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    outputOpDef->AddInputDesc(scalarTensorDesc);
    outputOpDef->AddOutputDesc(scalarTensorDesc);
    auto outputNode = graph->AddNode(outputOpDef);

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), greaterNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), greaterNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), lessNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), lessNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), switchNode1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(greaterNode->GetOutDataAnchor(0), switchNode1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(lessNode->GetOutDataAnchor(0), switchNode2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(greaterNode->GetOutDataAnchor(0), switchNode2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), switchNode3->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switchNode2->GetOutDataAnchor(0), switchNode3->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), switchNode4->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switchNode2->GetOutDataAnchor(0), switchNode4->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switchNode3->GetOutDataAnchor(1), mergeNode1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switchNode4->GetOutDataAnchor(0), mergeNode1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switchNode1->GetOutDataAnchor(1), mergeNode2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(mergeNode1->GetOutDataAnchor(0), mergeNode2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(mergeNode2->GetOutDataAnchor(0), outputNode->GetInDataAnchor(0));
  }
};
