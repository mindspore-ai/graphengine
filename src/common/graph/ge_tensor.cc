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

#include "graph/ge_tensor.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include "debug/ge_attr_define.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_attr_value.h"
#include "graph/model_serialize.h"
#include "proto/ge_ir.pb.h"
#include "utils/attr_utils.h"
#include "utils/ge_ir_utils.h"
#include "utils/tensor_utils.h"
#include "utils/type_utils.h"

namespace ge {
static const char *const kKeyDataTypeSelfDefined = "__tensor_desc_data_type__";

static const std::map<DataType, ::ge::proto::DataType> kDataTypeMap = {
  {DT_UNDEFINED, proto::DT_UNDEFINED},
  {DT_FLOAT, proto::DT_FLOAT},
  {DT_FLOAT16, proto::DT_FLOAT16},
  {DT_INT8, proto::DT_INT8},
  {DT_UINT8, proto::DT_UINT8},
  {DT_INT16, proto::DT_INT16},
  {DT_UINT16, proto::DT_UINT16},
  {DT_INT32, proto::DT_INT32},
  {DT_INT64, proto::DT_INT64},
  {DT_UINT32, proto::DT_UINT32},
  {DT_UINT64, proto::DT_UINT64},
  {DT_BOOL, proto::DT_BOOL},
  {DT_DOUBLE, proto::DT_DOUBLE},
  {DT_DUAL, proto::DT_DUAL},
  {DT_DUAL_SUB_INT8, proto::DT_DUAL_SUB_INT8},
  {DT_DUAL_SUB_UINT8, proto::DT_DUAL_SUB_UINT8},
  {DT_COMPLEX64, proto::DT_COMPLEX64},
  {DT_COMPLEX128, proto::DT_COMPLEX128},
  {DT_QINT8, proto::DT_QINT8},
  {DT_QINT16, proto::DT_QINT16},
  {DT_QINT32, proto::DT_QINT32},
  {DT_QUINT8, proto::DT_QUINT8},
  {DT_QUINT16, proto::DT_QUINT16},
  {DT_RESOURCE, proto::DT_RESOURCE},
  {DT_STRING_REF, proto::DT_STRING_REF},
  {DT_STRING, proto::DT_STRING},
};

static const std::map<DataType, int> kDataTypeSelfDefinedMap = {
  {DT_DUAL, 13},  {DT_DUAL_SUB_INT8, 14}, {DT_DUAL_SUB_UINT8, 15}, {DT_COMPLEX64, 16}, {DT_COMPLEX128, 17},
  {DT_QINT8, 18}, {DT_QINT16, 19},        {DT_QINT32, 20},         {DT_QUINT8, 21},    {DT_QUINT16, 22},
};

GeShape::GeShape() { shape_def_.InitDefault(); }

// Default
GeShape::GeShape(std::vector<int64_t> s) : GeShape() {
  auto proto_msg = shape_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto i : s) {
      proto_msg->add_dim(i);
    }
  }
}

size_t GeShape::GetDimNum() const {
  auto proto_msg = shape_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    if (proto_msg->dim_size() >= 0) {
      // check whether contain -2, if true, return -1
      for (auto i : proto_msg->dim()) {
        if (i == UNKNOWN_DIM_NUM) {
          return 0;
        }
      }
      return proto_msg->dim_size();
    } else {
      return 0;
    }
  }
  return 0;
}

int64_t GeShape::GetDim(size_t idx) const {
  auto proto_msg = shape_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    if (proto_msg->dim_size() > static_cast<int>(idx)) {
      return proto_msg->dim(static_cast<int>(idx));
    }
  }
  return 0;
}

graphStatus GeShape::SetDim(size_t idx, int64_t value) {
  auto proto_msg = shape_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    auto dims = proto_msg->mutable_dim();
    GE_CHECK_NOTNULL(dims);
    if (dims->empty()) {
      GELOGE(GRAPH_FAILED, "shape is empty");
      return GRAPH_FAILED;
    }
    if (static_cast<int>(idx) >= dims->size()) {
      GELOGE(GRAPH_FAILED, "idx is out of range");
      return GRAPH_FAILED;
    }
    proto_msg->set_dim(static_cast<int>(idx), value);
  }
  return GRAPH_SUCCESS;
}

std::vector<int64_t> GeShape::GetDims() const {
  vector<int64_t> dims;
  auto proto_msg = shape_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto i : proto_msg->dim()) {
      dims.push_back(i);
    }
  }
  return dims;
}

std::string GeShape::ToString() const {
  auto proto_msg = shape_def_.GetProtoMsg();
  if (proto_msg == nullptr) {
    return "";
  }

  std::stringstream ss;
  bool first = true;
  for (auto i : proto_msg->dim()) {
    if (first) {
      first = false;
    } else {
      ss << ",";
    }
    ss << i;
  }
  return ss.str();
}

int64_t GeShape::GetShapeSize() const {
  int64_t res = 1;
  auto proto_msg = shape_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    if (proto_msg->dim().empty()) {
      return 0;
    }
    for (auto i : proto_msg->dim()) {
      // if unknown shape, return -1
      if (i == UNKNOWN_DIM || i == UNKNOWN_DIM_NUM) {
        return UNKNOWN_DIM;
      }
      res *= i;
    }
  }
  return res;
}

///
/// @brief Check is unknown shape
/// @return bool
/// ///
bool GeShape::IsUnknownShape() const {
  auto proto_msg = shape_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto i : proto_msg->dim()) {
      if (i < 0) {
        return true;
      }
    }
  }
  return false;
}

///
/// @brief Check is a scalar
/// @return bool
///
bool GeShape::IsScalar() const {
  auto proto_msg = shape_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return proto_msg->dim().empty();
  }
  return false;
}

const string TENSOR_UTILS_SIZE = "size";
const string TENSOR_UTILS_WEIGHT_SIZE = "weight_size";
const string TENSOR_UTILS_REUSE_INPUT = "reuse_input";
const string TENSOR_UTILS_OUTPUT_TENSOR = "output_tensor";
const string TENSOR_UTILS_DEVICE_TYPE = "device_type";
const string TENSOR_UTILS_INPUT_TENSOR = "input_tensor";
const string TENSOR_UTILS_REAL_DIM_CNT = "real_dim_cnt";
const string TENSOR_UTILS_REUSE_INPUT_INDEX = "reuse_input_index";
const string TENSOR_UTILS_DATA_OFFSET = "data_offset";
const string TENSOR_UTILS_CMPS_SIZE = "cmps_size";
const string TENSOR_UTILS_CMPS_TAB = "cmps_tab";
const string TENSOR_UTILS_CMPS_TAB_OFFSET = "cmps_tab_offset";
const string TENSOR_UTILS_CMPSINFO = "cmps_info";
const string TENSOR_UTILS_ALLOFFSET_QUANTIZE_INFO = "alloffset_quantize_info";
const string TENSOR_UTILS_RC = "rc";
const string TENSOR_UTILS_ORIGIN_SHAPE = "origin_shape";
const string TENSOR_UTILS_ORIGIN_FORMAT = "origin_format";
const string TENSOR_UTILS_ORIGIN_DATA_TYPE = "origin_data_type";
const string TENSOR_UTILS_SHAPE_RANGE = "shape_range";
const string TENSOR_UTILS_REF_PORT_INDEX = "ref_port_index";

GeShape::GeShape(const ProtoMsgOwner &proto_owner, proto::ShapeDef *proto_msg) : shape_def_(proto_owner, proto_msg) {}

GeShape::GeShape(const GeShape &other) : GeShape() { shape_def_.CopyValueFrom(other.shape_def_); }

GeShape::GeShape(GeShape &&other) : GeShape() { shape_def_.MoveValueFrom(std::move(other.shape_def_)); }

GeShape &GeShape::operator=(const GeShape &other) {
  if (&other != this) {
    shape_def_.CopyValueFrom(other.shape_def_);
  }
  return *this;
}

GeShape &GeShape::operator=(GeShape &&other) {
  if (&other != this) {
    shape_def_.CopyValueFrom(std::move(other.shape_def_));
  }
  return *this;
}

GeTensorDesc::GeTensorDesc() {
  tensor_descriptor_.InitDefault();
  SetDataType(DT_FLOAT);
  Init();
}

// Default
GeTensorDesc::GeTensorDesc(GeShape shape, Format format, DataType dt) : GeTensorDesc() {
  SetFormat(format);
  SetDataType(dt);
  ShapeReference() = std::move(shape);
}

// Default
GeTensorDesc::GeTensorDesc(const GeTensorDesc &desc) : GeTensorDesc() {
  tensor_descriptor_.CopyValueFrom(desc.tensor_descriptor_);
}

// Default
GeTensorDesc::GeTensorDesc(GeTensorDesc &&desc) : GeTensorDesc() {
  tensor_descriptor_.MoveValueFrom(std::move(desc.tensor_descriptor_));
}

GeTensorDesc::GeTensorDesc(const ProtoMsgOwner &proto_owner, proto::TensorDescriptor *proto_msg)
    : tensor_descriptor_(proto_owner, proto_msg) {
  if (proto_msg != nullptr && !proto_msg->has_out_attr()) {
    proto_msg->set_has_out_attr(true);

    int64_t size = 0;
    (void)AttrUtils::GetInt(this, TENSOR_UTILS_SIZE, size);
    proto_msg->set_size(size);

    int64_t weight_size = 0;
    (void)AttrUtils::GetInt(this, TENSOR_UTILS_WEIGHT_SIZE, weight_size);
    proto_msg->set_weight_size(weight_size);

    bool reuse_input = false;
    (void)AttrUtils::GetBool(this, TENSOR_UTILS_REUSE_INPUT, reuse_input);
    proto_msg->set_reuse_input(reuse_input);

    bool output_tensor = false;
    (void)AttrUtils::GetBool(this, TENSOR_UTILS_OUTPUT_TENSOR, output_tensor);
    proto_msg->set_output_tensor(output_tensor);

    string device_type = "NPU";
    (void)AttrUtils::GetStr(this, TENSOR_UTILS_DEVICE_TYPE, device_type);
    proto_msg->set_device_type(device_type);

    bool input_tensor = false;
    (void)AttrUtils::GetBool(this, TENSOR_UTILS_INPUT_TENSOR, input_tensor);
    proto_msg->set_input_tensor(input_tensor);

    int64_t real_dim_cnt = 0;
    (void)AttrUtils::GetInt(this, TENSOR_UTILS_REAL_DIM_CNT, real_dim_cnt);
    proto_msg->set_real_dim_cnt(real_dim_cnt);

    int64_t reuse_input_index = 0;
    (void)AttrUtils::GetInt(this, TENSOR_UTILS_REUSE_INPUT_INDEX, reuse_input_index);
    proto_msg->set_reuse_input_index(reuse_input_index);

    int64_t data_offset = 0;
    (void)AttrUtils::GetInt(this, TENSOR_UTILS_DATA_OFFSET, data_offset);
    proto_msg->set_data_offset(data_offset);

    int64_t cmps_size = 0;
    (void)AttrUtils::GetInt(this, TENSOR_UTILS_CMPS_SIZE, cmps_size);
    proto_msg->set_cmps_size(cmps_size);

    string cmps_tab;
    (void)AttrUtils::GetStr(this, TENSOR_UTILS_CMPS_TAB, cmps_tab);
    proto_msg->set_cmps_tab(cmps_tab);

    int64_t cmps_tab_offset = 0;
    (void)AttrUtils::GetInt(this, TENSOR_UTILS_CMPS_TAB_OFFSET, cmps_tab_offset);
    proto_msg->set_cmps_tab_offset(cmps_tab_offset);
  }
}

bool GeTensorDesc::GeTensorDescAttrsAreEqual(const GeTensorDesc &r_ge_tensor_desc) const {
  const auto &tensor_descriptor = this->tensor_descriptor_.GetProtoMsg();
  const auto &r_tensor_descriptor = r_ge_tensor_desc.tensor_descriptor_.GetProtoMsg();
  if ((tensor_descriptor != nullptr) && (r_tensor_descriptor != nullptr)) {
    // Message TensorDescriptor in ge_ir.proto
    return (
      IsEqual(tensor_descriptor->name(), r_tensor_descriptor->name(), "TensorDescriptor.name()") &&
      IsEqual(tensor_descriptor->dtype(), r_tensor_descriptor->dtype(), "TensorDescriptor.dtype()") &&
      // Message ShapeDef in ge_ir.proto
      IsEqual(ToString(tensor_descriptor->shape().dim()), ToString(r_tensor_descriptor->shape().dim()),
              "TensorDescriptor.shape().dim()") &&
      IsEqual(tensor_descriptor->layout(), r_tensor_descriptor->layout(), "TensorDescriptor.layout()") &&
      IsEqual(tensor_descriptor->has_out_attr(), r_tensor_descriptor->has_out_attr(),
              "TensorDescriptor.has_out_attr()") &&
      IsEqual(tensor_descriptor->size(), r_tensor_descriptor->size(), "TensorDescriptor.size()") &&
      IsEqual(tensor_descriptor->weight_size(), r_tensor_descriptor->weight_size(), "TensorDescriptor.weight_size()") &&
      IsEqual(tensor_descriptor->reuse_input(), r_tensor_descriptor->reuse_input(), "TensorDescriptor.reuse_input()") &&
      IsEqual(tensor_descriptor->output_tensor(), r_tensor_descriptor->output_tensor(),
              "TensorDescriptor.output_tensor()") &&
      IsEqual(tensor_descriptor->device_type(), r_tensor_descriptor->device_type(), "TensorDescriptor.device_type()") &&
      IsEqual(tensor_descriptor->input_tensor(), r_tensor_descriptor->input_tensor(),
              "TensorDescriptor.input_tensor()") &&
      IsEqual(tensor_descriptor->real_dim_cnt(), r_tensor_descriptor->real_dim_cnt(),
              "TensorDescriptor.real_dim_cnt()") &&
      IsEqual(tensor_descriptor->reuse_input_index(), r_tensor_descriptor->reuse_input_index(),
              "TensorDescriptor.reuse_input_index()") &&
      IsEqual(tensor_descriptor->data_offset(), r_tensor_descriptor->data_offset(), "TensorDescriptor.data_offset()") &&
      IsEqual(tensor_descriptor->cmps_size(), r_tensor_descriptor->cmps_size(), "TensorDescriptor.cmps_size()") &&
      IsEqual(tensor_descriptor->cmps_tab(), r_tensor_descriptor->cmps_tab(), "TensorDescriptor.cmps_tab()") &&
      IsEqual(tensor_descriptor->cmps_tab_offset(), r_tensor_descriptor->cmps_tab_offset(),
              "TensorDescriptor.cmps_tab_offset()"));
  } else {
    return ((tensor_descriptor == nullptr) && (r_tensor_descriptor == nullptr));
  }
}

bool GeTensorDesc::operator==(const GeTensorDesc &r_ge_tensor_desc) const {
  return GeTensorDescAttrsAreEqual(r_ge_tensor_desc);
}

GeShape &GeTensorDesc::ShapeReference() const {
  if (tensor_descriptor_.GetProtoMsg() != nullptr) {
    GeShape refShape(tensor_descriptor_.GetProtoOwner(), tensor_descriptor_.GetProtoMsg()->mutable_shape());
    __shape_.RefTo(refShape);
  } else {
    GeShape refShape(tensor_descriptor_.GetProtoOwner(), nullptr);
    __shape_.RefTo(refShape);
  }
  return __shape_;
}

void GeTensorDesc::Init() {
  SetFormat(FORMAT_ND);
  SetOriginFormat(FORMAT_ND);
  TensorUtils::SetDeviceType(*this, DeviceType::NPU);
  if (tensor_descriptor_.GetProtoMsg() == nullptr) {
    GELOGE(GRAPH_FAILED, "ProtoType nullptr.");
    return;
  }
  tensor_descriptor_.GetProtoMsg()->set_has_out_attr(true);
}

ProtoAttrMapHelper GeTensorDesc::MutableAttrMap() {
  if (tensor_descriptor_.GetProtoMsg() != nullptr) {
    return ProtoAttrMapHelper(tensor_descriptor_.GetProtoOwner(), tensor_descriptor_.GetProtoMsg()->mutable_attr());
  }
  return ProtoAttrMapHelper(tensor_descriptor_.GetProtoOwner(), nullptr);
}

ConstProtoAttrMapHelper GeTensorDesc::GetAttrMap() const {
  if (tensor_descriptor_.GetProtoMsg() != nullptr) {
    return ConstProtoAttrMapHelper(tensor_descriptor_.GetProtoOwner(),
                                   tensor_descriptor_.GetProtoMsg()->mutable_attr());
  }
  return ConstProtoAttrMapHelper(tensor_descriptor_.GetProtoOwner(), nullptr);
}

void GeTensorDesc::Update(GeShape shape, Format format, DataType dt) {
  ShapeReference() = std::move(shape);
  SetFormat(format);
  SetDataType(dt);
}
GeShape GeTensorDesc::GetShape() const { return ShapeReference(); }

GeShape &GeTensorDesc::MutableShape() { return ShapeReference(); }

void GeTensorDesc::SetShape(GeShape shape) { ShapeReference() = std::move(shape); }

// set shape with -2, it stand for unknown shape
void GeTensorDesc::SetUnknownDimNumShape() { SetShape(GeShape({UNKNOWN_DIM_NUM})); }

// for unknown shape
graphStatus GeTensorDesc::SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  std::vector<vector<int64_t>> shape_range;
  for (const auto &ele : range) {
    shape_range.emplace_back(std::vector<int64_t>({ele.first, ele.second}));
  }
  auto ret = AttrUtils::SetListListInt(this, TENSOR_UTILS_SHAPE_RANGE, shape_range);
  return ret ? GRAPH_SUCCESS : GRAPH_FAILED;
}
graphStatus GeTensorDesc::GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
  std::vector<vector<int64_t>> shape_range;
  (void)AttrUtils::GetListListInt(this, TENSOR_UTILS_SHAPE_RANGE, shape_range);

  for (const auto &ele : shape_range) {
    // here must be only two elemenet because pair
    if (ele.size() != 2) {
      GELOGE(GRAPH_FAILED, "shape_range must contain only 2 value but really is %lu", ele.size());
      return GRAPH_FAILED;
    }
    std::pair<int64_t, int64_t> pair({ele[0], ele[1]});
    range.emplace_back(pair);
  }

  return GRAPH_SUCCESS;
}

GeShape GeTensorDesc::GetOriginShape() const {
  vector<int64_t> origin_shape;
  if (!AttrUtils::GetListInt(this, TENSOR_UTILS_ORIGIN_SHAPE, origin_shape)) {
    return GeShape();
  }
  return GeShape(origin_shape);
}

void GeTensorDesc::SetOriginShape(const GeShape &origin_shape) {
  std::vector<int64_t> origin_shape_tmp = origin_shape.GetDims();
  (void)AttrUtils::SetListInt(this, TENSOR_UTILS_ORIGIN_SHAPE, origin_shape_tmp);
}

Format GeTensorDesc::GetFormat() const {
  auto tensor_descriptor_msg = tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    return TypeUtils::SerialStringToFormat(tensor_descriptor_msg->layout());
  }
  return FORMAT_RESERVED;
}

void GeTensorDesc::SetFormat(Format format) {
  auto tensor_descriptor_msg = tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_layout(TypeUtils::FormatToSerialString(format));
  }
}

void GeTensorDesc::SetName(const std::string &name) {
  auto tensor_descriptor_msg = tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_name(name);
    return;
  }
  GELOGW("[SetName]tensor_descriptor_msg is null.");
}

const std::string GeTensorDesc::GetName() const {
  auto tensor_descriptor_msg = tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    return tensor_descriptor_msg->name();
  }
  GELOGW("[GetName]tensor_descriptor_msg is null.");
  return "";
}

Format GeTensorDesc::GetOriginFormat() const {
  std::string origin_format_str;
  if (!AttrUtils::GetStr(this, TENSOR_UTILS_ORIGIN_FORMAT, origin_format_str)) {
    // Can not get the certificate and it's not set, return directly
    return FORMAT_RESERVED;
  }
  if (origin_format_str == "RESERVED") {
    return FORMAT_RESERVED;
  }
  return TypeUtils::SerialStringToFormat(origin_format_str);
}

void GeTensorDesc::SetOriginFormat(Format origin_format) {
  std::string origin_format_str = "RESERVED";
  if (origin_format != FORMAT_RESERVED) {
    origin_format_str = TypeUtils::FormatToSerialString(origin_format);
  }
  (void)AttrUtils::SetStr(this, TENSOR_UTILS_ORIGIN_FORMAT, origin_format_str);
}

DataType GeTensorDesc::GetDataType() const {
  auto tensor_descriptor_msg = tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg == nullptr) {
    return DT_UNDEFINED;
  }
  auto &attr_map = *(tensor_descriptor_msg->mutable_attr());
  // Data type
  auto it_data_type = attr_map.find(kKeyDataTypeSelfDefined);
  if (it_data_type != attr_map.end()) {
    int64_t data_type_proto = it_data_type->second.i();
    for (auto it : kDataTypeSelfDefinedMap) {
      if (it.second == data_type_proto) {
        return it.first;
      }
    }
  } else {
    auto data_type_proto = tensor_descriptor_msg->dtype();
    for (auto it : kDataTypeMap) {
      if (it.second == data_type_proto) {
        return it.first;
      }
    }
  }
  return DT_UNDEFINED;
}

void GeTensorDesc::SetDataType(DataType dataType) {
  auto tensor_descriptor_msg = tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg == nullptr) {
    return;
  }
  auto &attr_maps = *(tensor_descriptor_msg->mutable_attr());
  (void)attr_maps.erase(kKeyDataTypeSelfDefined);

  // Data type
  auto it = kDataTypeMap.find(dataType);
  if (it != kDataTypeMap.end()) {
    tensor_descriptor_msg->set_dtype(it->second);
    return;
  }
  auto it2 = kDataTypeSelfDefinedMap.find(dataType);
  if (it2 != kDataTypeSelfDefinedMap.end()) {
    attr_maps[kKeyDataTypeSelfDefined].set_i(it2->second);
  }
}

void GeTensorDesc::SetOriginDataType(DataType origin_data_type) {
  std::string origin_data_type_str = "RESERVED";
  if (origin_data_type != DT_UNDEFINED) {
    origin_data_type_str = TypeUtils::DataTypeToSerialString(origin_data_type);
  }
  (void)AttrUtils::SetStr(this, TENSOR_UTILS_ORIGIN_DATA_TYPE, origin_data_type_str);
}

DataType GeTensorDesc::GetOriginDataType() const {
  std::string origin_data_type_str;
  if (!AttrUtils::GetStr(this, TENSOR_UTILS_ORIGIN_DATA_TYPE, origin_data_type_str)) {
    return DT_UNDEFINED;
  }
  if (origin_data_type_str == "RESERVED") {
    return DT_UNDEFINED;
  }
  return TypeUtils::SerialStringToDataType(origin_data_type_str);
}

std::vector<uint32_t> GeTensorDesc::GetRefPortIndex() const {
  vector<uint32_t> ref_port_index;
  (void)AttrUtils::GetListInt(this, TENSOR_UTILS_REF_PORT_INDEX, ref_port_index);
  return ref_port_index;
}

void GeTensorDesc::SetRefPortByIndex(const std::vector<uint32_t> &index) {
  (void)AttrUtils::SetListInt(this, TENSOR_UTILS_REF_PORT_INDEX, index);
}

graphStatus GeTensorDesc::IsValid() const {
  auto dtype = this->GetDataType();
  auto format = this->GetFormat();
  if (dtype == DT_UNDEFINED && format == FORMAT_RESERVED) {
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

GeTensorDesc GeTensorDesc::Clone() const { return *this; }

GeTensorDesc &GeTensorDesc::operator=(const GeTensorDesc &desc) {
  if (&desc != this) {
    tensor_descriptor_.CopyValueFrom(desc.tensor_descriptor_);
  }
  return *this;
}

GeTensorDesc &GeTensorDesc::operator=(GeTensorDesc &&desc) {
  if (&desc != this) {
    tensor_descriptor_.CopyValueFrom(std::move(desc.tensor_descriptor_));
  }
  return *this;
}

GeTensor::GeTensor::GeTensor() {
  tensor_def_.InitDefault();
  // Default init desc
  DescReference() = GeTensorDesc();
}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc) : GeTensor() { DescReference() = tensor_desc; }

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const vector<uint8_t> &data) : GeTensor() {
  DescReference() = tensor_desc;
  auto proto_msg = tensor_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_data(data.data(), data.size());
  }
}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const uint8_t *data, size_t size) : GeTensor() {
  DescReference() = tensor_desc;
  auto proto_msg = tensor_def_.GetProtoMsg();
  if (proto_msg != nullptr && data != nullptr) {
    proto_msg->set_data(data, size);
  }
}

GeTensor::GeTensor(GeTensorDesc &&tensor_desc, vector<uint8_t> &&data) : GeTensor() {
  DescReference() = std::move(tensor_desc);
  auto proto_msg = tensor_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_data(data.data(), data.size());
  }
}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const Buffer &data) : GeTensor() {
  DescReference() = tensor_desc;
  auto proto_msg = tensor_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    if (data.size() == 0) {
      GELOGI("GetSize res is 0.");
    }
    if (data.data() == nullptr) {
      GELOGI("data addr is null.");
    }
    proto_msg->set_data(data.GetData(), data.GetSize());
  }
}

GeTensor::GeTensor(const ProtoMsgOwner &proto_owner, proto::TensorDef *proto_msg)
    : tensor_def_(proto_owner, proto_msg) {}

GeTensorDesc GeTensor::GetTensorDesc() const { return DescReference(); }

GeTensorDesc &GeTensor::MutableTensorDesc() { return DescReference(); }

GeTensorDesc &GeTensor::DescReference() const {
  if (tensor_def_.GetProtoMsg() != nullptr) {
    GeTensorDesc tensor_desc(tensor_def_.GetProtoOwner(), tensor_def_.GetProtoMsg()->mutable_desc());
    __desc_.RefTo(tensor_desc);
  } else {
    GeTensorDesc tensor_desc(tensor_def_.GetProtoOwner(), nullptr);
    __desc_.RefTo(tensor_desc);
  }
  return __desc_;
}

void GeTensor::SetTensorDesc(const GeTensorDesc &tensor_desc) { DescReference() = tensor_desc; }

const Buffer GeTensor::GetData() const {
  auto proto_msg = tensor_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return Buffer(tensor_def_.GetProtoOwner(), proto_msg->mutable_data());
  }
  return Buffer();
}

Buffer GeTensor::MutableData() {
  auto proto_msg = tensor_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return Buffer(tensor_def_.GetProtoOwner(), proto_msg->mutable_data());
  }
  return Buffer();
}

graphStatus GeTensor::SetData(vector<uint8_t> &&data) {
  auto proto_msg = tensor_def_.GetProtoMsg();
  GE_CHECK_NOTNULL(proto_msg);
  proto_msg->set_data(data.data(), data.size());
  return GRAPH_SUCCESS;
}

graphStatus GeTensor::SetData(const vector<uint8_t> &data) {
  auto proto_msg = tensor_def_.GetProtoMsg();
  GE_CHECK_NOTNULL(proto_msg);
  proto_msg->set_data(data.data(), data.size());
  return GRAPH_SUCCESS;
}

graphStatus GeTensor::SetData(const uint8_t *data, size_t size) {
  GE_CHECK_NOTNULL(data);
  auto proto_msg = tensor_def_.GetProtoMsg();
  GE_CHECK_NOTNULL(proto_msg);
  proto_msg->set_data(data, size);
  return GRAPH_SUCCESS;
}

graphStatus GeTensor::SetData(const Buffer &data) {
  auto proto_msg = tensor_def_.GetProtoMsg();
  GE_CHECK_NOTNULL(proto_msg);
  if (data.size() == 0) {
    GELOGI("GetSize res is 0.");
  }
  if (data.data() == nullptr) {
    GELOGI("data addr is null.");
  }
  proto_msg->set_data(data.data(), data.size());
  return GRAPH_SUCCESS;
}

GeTensor GeTensor::Clone() const {
  GeTensor tensor;
  tensor.tensor_def_.CopyValueFrom(tensor_def_);
  return tensor;
}

GeTensor::GeTensor(const GeTensor &other) { tensor_def_ = other.tensor_def_; }

GeTensor &GeTensor::operator=(const GeTensor &other) {
  if (&other != this) {
    tensor_def_ = other.tensor_def_;
  }
  return *this;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetSize(const GeTensorDesc &tensor_desc,
                                                                                int64_t &size) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  GE_CHECK_NOTNULL(tensor_descriptor_msg);
  size = static_cast<int64_t>(tensor_descriptor_msg->size());
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetSize(GeTensorDesc &tensor_desc, int64_t size) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_size(size);
  }
}

uint32_t TensorUtils::GetWeightSize(const GeTensorDesc &tensor_desc) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    return static_cast<uint32_t>(tensor_descriptor_msg->weight_size());
  }
  return 0;
}

uint32_t TensorUtils::GetWeightSize(const GeTensor &tensor) { return GetWeightSize(tensor.GetTensorDesc()); }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t TensorUtils::GetWeightSize(const ConstGeTensorPtr &tensor_ptr) {
  if (tensor_ptr == nullptr) {
    return 0;
  }
  return GetWeightSize(*tensor_ptr);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint8_t *TensorUtils::GetWeightAddr(const ConstGeTensorPtr &tensor_ptr,
                                                                                   uint8_t *base) {
  if (tensor_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "tensor_ptr is null.");
    return nullptr;
  }
  return GetWeightAddr(*tensor_ptr, base);
}

uint8_t *TensorUtils::GetWeightAddr(const GeTensor &tensor, uint8_t *base) {
  if (base == nullptr) {
    GELOGE(GRAPH_FAILED, "base is null.");
    return nullptr;
  }
  int64_t weight_data_offset = 0;
  if (GetDataOffset(tensor.GetTensorDesc(), weight_data_offset) != GRAPH_SUCCESS) return nullptr;

  if (weight_data_offset == 0) {
    // The weight of offset 0 is still in const op, still get from ATTR_NAME_WEIGHTS.
    return const_cast<uint8_t *>(tensor.GetData().data());
  }

  return base + weight_data_offset;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetWeightSize(GeTensorDesc &tensor_desc,
                                                                               uint32_t size) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_weight_size(size);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetReuseInput(const GeTensorDesc &tensor_desc,
                                                                                      bool &flag) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  GE_CHECK_NOTNULL(tensor_descriptor_msg);
  flag = tensor_descriptor_msg->reuse_input();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetReuseInput(GeTensorDesc &tensor_desc, bool flag) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_reuse_input(flag);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetOutputTensor(const GeTensorDesc &tensor_desc,
                                                                                        bool &flag) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  GE_CHECK_NOTNULL(tensor_descriptor_msg);
  flag = tensor_descriptor_msg->output_tensor();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetOutputTensor(GeTensorDesc &tensor_desc, bool flag) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_output_tensor(flag);
  }
}

static map<uint32_t, string> device_to_str_map{
  {0, "NPU"},
  {1, "CPU"},
};
static map<string, uint32_t> str_to_device_map{
  {"NPU", 0},
  {"CPU", 1},
};

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetDeviceType(const GeTensorDesc &tensor_desc,
                                                                                      DeviceType &type) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  GE_CHECK_NOTNULL(tensor_descriptor_msg);
  string type_str = tensor_descriptor_msg->device_type();
  type = DeviceType(str_to_device_map[type_str]);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetDeviceType(GeTensorDesc &tensor_desc,
                                                                               DeviceType type) {
  auto type_str = device_to_str_map[type];
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_device_type(type_str);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetInputTensor(const GeTensorDesc &tensor_desc,
                                                                                       bool &flag) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  GE_CHECK_NOTNULL(tensor_descriptor_msg);
  flag = tensor_descriptor_msg->input_tensor();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetInputTensor(GeTensorDesc &tensor_desc, bool flag) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_input_tensor(flag);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetRealDimCnt(const GeTensorDesc &tensor_desc,
                                                                                      uint32_t &cnt) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  GE_CHECK_NOTNULL(tensor_descriptor_msg);
  cnt = static_cast<uint32_t>(tensor_descriptor_msg->real_dim_cnt());
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetRealDimCnt(GeTensorDesc &tensor_desc,
                                                                               uint32_t cnt) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_real_dim_cnt(cnt);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetReuseInputIndex(const GeTensorDesc &tensor_desc, uint32_t &idx) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  GE_CHECK_NOTNULL(tensor_descriptor_msg);

  idx = static_cast<uint32_t>(tensor_descriptor_msg->reuse_input_index());
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetReuseInputIndex(GeTensorDesc &tensor_desc,
                                                                                    uint32_t idx) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_reuse_input_index(idx);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetDataOffset(const GeTensorDesc &tensor_desc,
                                                                                      int64_t &offset) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    offset = tensor_descriptor_msg->data_offset();
    return GRAPH_SUCCESS;
  } else {
    GELOGW("tensor_descriptor_msg is nullptr.");
    return GRAPH_FAILED;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetDataOffset(GeTensorDesc &tensor_desc,
                                                                               int64_t offset) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_data_offset(offset);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetCmpsSize(const GeTensorDesc &tensor_desc,
                                                                                    uint32_t &cmp_size) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    cmp_size = static_cast<uint32_t>(tensor_descriptor_msg->cmps_size());
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetCmpsSize(GeTensorDesc &tensor_desc,
                                                                             uint32_t cmp_size) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_cmps_size(cmp_size);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetCmpsTab(const GeTensorDesc &tensor_desc,
                                                                                   vector<uint8_t> &vec) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    string str = tensor_descriptor_msg->cmps_tab();
    vec.assign(str.begin(), str.end());
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetCmpsTab(GeTensorDesc &tensor_desc,
                                                                            const uint8_t *data, size_t size) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    GE_CHK_BOOL_EXEC(data != nullptr, return, "data is null.");
    string str((const char *)data, size);
    tensor_descriptor_msg->set_cmps_tab(str);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetCmpsTabOffset(const GeTensorDesc &tensor_desc, int64_t &tab_offset) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tab_offset = tensor_descriptor_msg->cmps_tab_offset();
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetCmpsTabOffset(GeTensorDesc &tensor_desc,
                                                                                  int64_t tab_offset) {
  auto tensor_descriptor_msg = tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_cmps_tab_offset(tab_offset);
  }
}

graphStatus TensorUtils::GetCmpsInfo(const GeTensorDesc &tensor_desc, CompressInfo &info) {
  GeAttrValue attr_value;
  if (tensor_desc.GetAttr(TENSOR_UTILS_CMPSINFO, attr_value) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return attr_value.GetValue(info);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetCmpsInfo(GeTensorDesc &tensor_desc,
                                                                             const CompressInfo &info) {
  (void)tensor_desc.SetAttr(TENSOR_UTILS_CMPSINFO, GeAttrValue::CreateFrom(info));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool TensorUtils::HasAlloffsetQuantizeInfo(
  const GeTensorDesc &tensor_desc) {
  return tensor_desc.HasAttr(TENSOR_UTILS_ALLOFFSET_QUANTIZE_INFO);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetAlloffsetQuantizeInfo(const GeTensorDesc &tensor_desc, AllOffsetQuantizeInfo &info) {
  GeAttrValue attr_value;
  if (tensor_desc.GetAttr(TENSOR_UTILS_ALLOFFSET_QUANTIZE_INFO, attr_value) != GRAPH_SUCCESS) {
    GELOGW("get attr alloffset_quantize_info fail.");
  }
  return attr_value.GetValue(info);
}

void TensorUtils::SetAlloffsetQuantizeInfo(GeTensorDesc &tensor_desc, const AllOffsetQuantizeInfo &info) {
  (void)tensor_desc.SetAttr(TENSOR_UTILS_ALLOFFSET_QUANTIZE_INFO, GeAttrValue::CreateFrom(info));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetRC(const GeTensorDesc &tensor_desc,
                                                                              uint32_t &rc) {
  return AttrUtils::GetInt(&tensor_desc, TENSOR_UTILS_RC, rc) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetRC(GeTensorDesc &tensor_desc, uint32_t rc) {
  (void)AttrUtils::SetInt(&tensor_desc, TENSOR_UTILS_RC, rc);
}
}  // namespace ge
