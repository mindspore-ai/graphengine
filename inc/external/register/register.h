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

#ifndef INC_EXTERNAL_REGISTER_REGISTER_H_
#define INC_EXTERNAL_REGISTER_REGISTER_H_

#include <google/protobuf/message.h>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "graph/operator.h"
#include "register/register_error_codes.h"
#include "register/register_fmk_types.h"
#include "register/register_types.h"

using std::unique_ptr;
using std::map;
using std::make_shared;
using std::to_string;
using std::string;
using std::pair;
using std::vector;

namespace ge {
class Operator;
class TensorDesc;
class Tensor;
class TBEPluginManager;
}

namespace domi {
struct OpOutput {
  ge::Operator op;
  // The output name of op
  std::string outputName;
};

struct InferShapeContext {
  ge::Operator op;
  // Input name, input
  std::map<std::string, OpOutput> inputs;
};

struct InferShapeOutput {
  std::vector<ge::TensorDesc> outputDescs;
  std::vector<uint32_t> realDimCnt;
};

enum OmgMoveTypeToAttr {
  OMG_MOVE_TYPE_DTYPE = 0,
  OMG_MOVE_TYPE_VALUE,
  OMG_MOVE_TYPE_SHAPE,
  OMG_MOVE_TYPE_FORMAT,
  OMG_MOVE_TYPE_AXIS,
  OMG_MOVE_TYPE_SCALAR_VALUE,
  OMG_REMOVE_TYPE_WITH_COND = 1000,
};

struct MoveInputToAttrStu {
  int inputIdx;
  std::string attrName;
  OmgMoveTypeToAttr moveType;
  bool attrValue;
};

Status AutoMappingFn(const google::protobuf::Message *op_src, ge::Operator &op);
Status AutoMappingFnDynamic(const google::protobuf::Message *op_src, ge::Operator &op,
                            std::map<std::string, std::pair<std::string, std::string>> dynamic_name_attr_value,
                            int in_pos = -1, int out_pos = -1);
using google::protobuf::Message;

using ParseParamFunc = std::function<domi::Status(const google::protobuf::Message *, ge::Operator &)>;
using InferShapeFunc = std::function<domi::Status(const ge::Operator &, std::vector<ge::TensorDesc> &)>;
using InferShapeFuncV2 = std::function<domi::Status(const InferShapeContext &, InferShapeOutput &)>;
using GetWorkspaceSizeFunc = std::function<domi::Status(const ge::Operator &, std::vector<int64_t> &)>;
using UpdateOpDescFunc = std::function<domi::Status(ge::Operator &)>;
using BuildTeBinFunc = std::function<domi::Status(const ge::Operator &, TEBinInfo &)>;

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpRegistrationData {
 public:
  OpRegistrationData(const std::string &om_optype);

  ~OpRegistrationData();

  OpRegistrationData &FrameworkType(const domi::FrameworkType &fmk_type);

  OpRegistrationData &OriginOpType(const std::initializer_list<std::string> &ori_optype_list);

  OpRegistrationData &OriginOpType(const std::string &ori_optype);

  OpRegistrationData &ParseParamsFn(const ParseParamFunc &parseParamFn);

  OpRegistrationData &InferShapeAndTypeFn(const InferShapeFunc &inferShapeFn);

  OpRegistrationData &InferShapeAndTypeFn(const InferShapeFuncV2 &inferShapeFn);

  OpRegistrationData &UpdateOpDescFn(const UpdateOpDescFunc &updateOpDescFn);

  OpRegistrationData &GetWorkspaceSizeFn(const GetWorkspaceSizeFunc &getWorkspaceSizeFn);

  OpRegistrationData &TEBinBuildFn(const BuildTeBinFunc &buildTeBinFn);

  OpRegistrationData &ImplyType(const domi::ImplyType &imply_type);

  OpRegistrationData &Formats(const std::initializer_list<domi::tagDomiTensorFormat> &input_formats,
                              const std::initializer_list<domi::tagDomiTensorFormat> &output_formats);

  OpRegistrationData &WeightFormats(const std::initializer_list<domi::tagDomiTensorFormat> &weight_formats);

  OpRegistrationData &InputFormat(const std::initializer_list<std::initializer_list<ge::Format>> &inputFormats);
  OpRegistrationData &OutputFormat(const std::initializer_list<std::initializer_list<ge::Format>> &outputFormats);
  OpRegistrationData &InputDataType(const std::initializer_list<std::initializer_list<ge::DataType>> &inputDataTypes);
  OpRegistrationData &OutputDataType(const std::initializer_list<std::initializer_list<ge::DataType>> &outputDataTypes);
  OpRegistrationData &InputLimitedTensorDescInfo(
      const std::initializer_list<std::initializer_list<ge::TensorDescInfo>> &limitedTensorDescs);
  OpRegistrationData &OutputLimitedTensorDescInfo(
      const std::initializer_list<std::initializer_list<ge::TensorDescInfo>> &limitedTensorDescs);

  OpRegistrationData &MoveInputToAttr(int inputIdx, const std::string &attrName, OmgMoveTypeToAttr moveType);
  OpRegistrationData &DelInputWithCond(int inputIdx, const std::string &attrName, bool attrValue);

 private:
  domi::FrameworkType fmk_type_;                           // Framework type
  std::set<std::string> ori_optype_set_;                   // OP type in the original model, there may be multiple
  std::string om_optype_;                                  // OP type in OM model
  domi::ImplyType imply_type_;                             // Execution type
  std::vector<domi::tagDomiTensorFormat> input_formats_;   // Data formats supported by operator input
  std::vector<domi::tagDomiTensorFormat> output_formats_;  // Data formats supported by operator output
  std::vector<domi::tagDomiTensorFormat> weight_formats_;  // Data format supported by operator weight

  ParseParamFunc parseParamFn_;              // ParseParam function
  InferShapeFunc inferShapeFn_;              // InferShape function
  InferShapeFuncV2 inferShapeFnV2_;          // InferShape function
  GetWorkspaceSizeFunc getWorkspaceSizeFn_;  // GetWorkspaceSizeFunc function
  UpdateOpDescFunc updateOpDescFn_;
  BuildTeBinFunc buildTeBinFn_;
  // Input formats list supported by tbe operators
  std::vector<std::vector<ge::Format>> supportedInputFormats_;
  // Output formats list supported by tbe operators
  std::vector<std::vector<ge::Format>> supportedOutputFormats_;
  // Input datatypes list supported by tbe operators
  std::vector<std::vector<ge::DataType>> supportedInputDataTypes_;
  // Output datatypes list supported by tbe operators
  std::vector<std::vector<ge::DataType>> supportedOutputDataTypes_;
  // Input tensordesinfo list supported by tbe operator
  std::vector<std::vector<ge::TensorDescInfo>> inputLimitedTensorDescs_;
  // Output tensordesinfo list supported by tbe operator
  std::vector<std::vector<ge::TensorDescInfo>> outputLimitedTensorDescs_;

  std::vector<MoveInputToAttrStu> moveInputToAttrVec_;
  friend class OpRegistry;
  friend class OpRegistrationTbe;
  friend class ge::TBEPluginManager;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpReceiver {
 public:
  OpReceiver(OpRegistrationData &reg_data);
  ~OpReceiver() {}
};

#define REGISTER_CUSTOM_OP(name) REGISTER_CUSTOM_OP_UNIQ_HELPER(__COUNTER__, name)
#define REGISTER_CUSTOM_OP_UNIQ_HELPER(ctr, name) REGISTER_CUSTOM_OP_UNIQ(ctr, name)
#define REGISTER_CUSTOM_OP_UNIQ(ctr, name)     \
  static OpReceiver register_op##ctr           \
      __attribute__((unused)) =                \
          OpRegistrationData(name)
}  // namespace domi

namespace ge {
using OpOutput = domi::OpOutput;
using InferShapeContext = domi::InferShapeContext;
using InferShapeOutput = domi::InferShapeOutput;
using OmgMoveTypeToAttr = domi::OmgMoveTypeToAttr;
using MoveInputToAttrStu = domi::MoveInputToAttrStu;
using OpRegistrationData = domi::OpRegistrationData;
using OpReceiver = domi::OpReceiver;
}
#endif  // INC_EXTERNAL_REGISTER_REGISTER_H_
