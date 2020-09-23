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

#include "graph/utils/type_utils.h"
#include "debug/ge_util.h"

using domi::domiTensorFormat_t;

namespace ge {
static const std::map<Format, std::string> kFormatToStringMap = {
  {FORMAT_NCHW, "NCHW"},
  {FORMAT_NHWC, "NHWC"},
  {FORMAT_ND, "ND"},
  {FORMAT_NC1HWC0, "NC1HWC0"},
  {FORMAT_FRACTAL_Z, "FRACTAL_Z"},
  {FORMAT_NC1C0HWPAD, "NC1C0HWPAD"},
  {FORMAT_NHWC1C0, "NHWC1C0"},
  {FORMAT_FSR_NCHW, "FSR_NCHW"},
  {FORMAT_FRACTAL_DECONV, "FRACTAL_DECONV"},
  {FORMAT_C1HWNC0, "C1HWNC0"},
  {FORMAT_FRACTAL_DECONV_TRANSPOSE, "FRACTAL_DECONV_TRANSPOSE"},
  {FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS, "FRACTAL_DECONV_SP_STRIDE_TRANS"},
  {FORMAT_NC1HWC0_C04, "NC1HWC0_C04"},
  {FORMAT_FRACTAL_Z_C04, "FRACTAL_Z_C04"},
  {FORMAT_CHWN, "CHWN"},
  {FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS, "DECONV_SP_STRIDE8_TRANS"},
  {FORMAT_NC1KHKWHWC0, "NC1KHKWHWC0"},
  {FORMAT_BN_WEIGHT, "BN_WEIGHT"},
  {FORMAT_FILTER_HWCK, "FILTER_HWCK"},
  {FORMAT_HWCN, "HWCN"},
  {FORMAT_HASHTABLE_LOOKUP_LOOKUPS, "LOOKUP_LOOKUPS"},
  {FORMAT_HASHTABLE_LOOKUP_KEYS, "LOOKUP_KEYS"},
  {FORMAT_HASHTABLE_LOOKUP_VALUE, "LOOKUP_VALUE"},
  {FORMAT_HASHTABLE_LOOKUP_OUTPUT, "LOOKUP_OUTPUT"},
  {FORMAT_HASHTABLE_LOOKUP_HITS, "LOOKUP_HITS"},
  {FORMAT_MD, "MD"},
  {FORMAT_NDHWC, "NDHWC"},
  {FORMAT_NCDHW, "NCDHW"},
  {FORMAT_DHWCN, "DHWCN"},
  {FORMAT_DHWNC, "DHWNC"},
  {FORMAT_NDC1HWC0, "NDC1HWC0"},
  {FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"},
  {FORMAT_FRACTAL_Z_3D_TRANSPOSE, "FRACTAL_Z_3D_TRANSPOSE"},
  {FORMAT_C1HWNCoC0, "C1HWNCoC0"},
  {FORMAT_FRACTAL_NZ, "FRACTAL_NZ"},
  {FORMAT_CN, "CN"},
  {FORMAT_NC, "NC"},
  {FORMAT_FRACTAL_ZN_LSTM, "FRACTAL_ZN_LSTM"},
  {FORMAT_FRACTAL_Z_G, "FRACTAL_Z_G"},
  {FORMAT_RESERVED, "FORMAT_RESERVED"},
  {FORMAT_ALL, "ALL"}};

static const std::map<domiTensorFormat_t, Format> kDomiFormatToGeFormat = {
  {domi::DOMI_TENSOR_NCHW, FORMAT_NCHW},
  {domi::DOMI_TENSOR_NHWC, FORMAT_NHWC},
  {domi::DOMI_TENSOR_ND, FORMAT_ND},
  {domi::DOMI_TENSOR_NC1HWC0, FORMAT_NC1HWC0},
  {domi::DOMI_TENSOR_FRACTAL_Z, FORMAT_FRACTAL_Z},
  {domi::DOMI_TENSOR_NC1C0HWPAD, FORMAT_NC1C0HWPAD},
  {domi::DOMI_TENSOR_NHWC1C0, FORMAT_NHWC1C0},
  {domi::DOMI_TENSOR_FSR_NCHW, FORMAT_FSR_NCHW},
  {domi::DOMI_TENSOR_FRACTAL_DECONV, FORMAT_FRACTAL_DECONV},
  {domi::DOMI_TENSOR_BN_WEIGHT, FORMAT_BN_WEIGHT},
  {domi::DOMI_TENSOR_CHWN, FORMAT_CHWN},
  {domi::DOMI_TENSOR_FILTER_HWCK, FORMAT_FILTER_HWCK},
  {domi::DOMI_TENSOR_NDHWC, FORMAT_NDHWC},
  {domi::DOMI_TENSOR_NCDHW, FORMAT_NCDHW},
  {domi::DOMI_TENSOR_DHWCN, FORMAT_DHWCN},
  {domi::DOMI_TENSOR_DHWNC, FORMAT_DHWNC},
  {domi::DOMI_TENSOR_RESERVED, FORMAT_RESERVED}};

static const std::unordered_set<std::string> kInternalFormat = {"NC1HWC0",
                                                                "FRACTAL_Z",
                                                                "NC1C0HWPAD",
                                                                "NHWC1C0",
                                                                "FRACTAL_DECONV",
                                                                "C1HWNC0",
                                                                "FRACTAL_DECONV_TRANSPOSE",
                                                                "FRACTAL_DECONV_SP_STRIDE_TRANS",
                                                                "NC1HWC0_C04",
                                                                "FRACTAL_Z_C04",
                                                                "FRACTAL_DECONV_SP_STRIDE8_TRANS",
                                                                "NC1KHKWHWC0",
                                                                "C1HWNCoC0",
                                                                "FRACTAL_ZZ",
                                                                "FRACTAL_NZ",
                                                                "NDC1HWC0",
                                                                "FORMAT_FRACTAL_Z_3D",
                                                                "FORMAT_FRACTAL_Z_3D_TRANSPOSE",
                                                                "FORMAT_FRACTAL_ZN_LSTM",
                                                                "FORMAT_FRACTAL_Z_G"};

static const std::map<std::string, Format> kDataFormatMap = {
  {"NCHW", FORMAT_NCHW}, {"NHWC", FORMAT_NHWC}, {"NDHWC", FORMAT_NDHWC}, {"NCDHW", FORMAT_NCDHW}, {"ND", FORMAT_ND}};

static const std::map<std::string, Format> kStringToFormatMap = {
  {"NCHW", FORMAT_NCHW},
  {"NHWC", FORMAT_NHWC},
  {"ND", FORMAT_ND},
  {"NC1HWC0", FORMAT_NC1HWC0},
  {"FRACTAL_Z", FORMAT_FRACTAL_Z},
  {"NC1C0HWPAD", FORMAT_NC1C0HWPAD},
  {"NHWC1C0", FORMAT_NHWC1C0},
  {"FSR_NCHW", FORMAT_FSR_NCHW},
  {"FRACTAL_DECONV", FORMAT_FRACTAL_DECONV},
  {"C1HWNC0", FORMAT_C1HWNC0},
  {"FRACTAL_DECONV_TRANSPOSE", FORMAT_FRACTAL_DECONV_TRANSPOSE},
  {"FRACTAL_DECONV_SP_STRIDE_TRANS", FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
  {"NC1HWC0_C04", FORMAT_NC1HWC0_C04},
  {"FRACTAL_Z_C04", FORMAT_FRACTAL_Z_C04},
  {"CHWN", FORMAT_CHWN},
  {"DECONV_SP_STRIDE8_TRANS", FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
  {"NC1KHKWHWC0", FORMAT_NC1KHKWHWC0},
  {"BN_WEIGHT", FORMAT_BN_WEIGHT},
  {"FILTER_HWCK", FORMAT_FILTER_HWCK},
  {"HWCN", FORMAT_HWCN},
  {"LOOKUP_LOOKUPS", FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
  {"LOOKUP_KEYS", FORMAT_HASHTABLE_LOOKUP_KEYS},
  {"LOOKUP_VALUE", FORMAT_HASHTABLE_LOOKUP_VALUE},
  {"LOOKUP_OUTPUT", FORMAT_HASHTABLE_LOOKUP_OUTPUT},
  {"LOOKUP_HITS", FORMAT_HASHTABLE_LOOKUP_HITS},
  {"MD", FORMAT_MD},
  {"C1HWNCoC0", FORMAT_C1HWNCoC0},
  {"FRACTAL_NZ", FORMAT_FRACTAL_NZ},
  {"NDHWC", FORMAT_NDHWC},
  {"NCDHW", FORMAT_NCDHW},
  {"DHWCN", FORMAT_DHWCN},
  {"DHWNC", FORMAT_DHWNC},
  {"NDC1HWC0", FORMAT_NDC1HWC0},
  {"FRACTAL_Z_3D", FORMAT_FRACTAL_Z_3D},
  {"FRACTAL_Z_3D_TRANSPOSE", FORMAT_FRACTAL_Z_3D_TRANSPOSE},
  {"CN", FORMAT_CN},
  {"NC", FORMAT_NC},
  {"FRACTAL_ZN_LSTM", FORMAT_FRACTAL_ZN_LSTM},
  {"FRACTAL_Z_G", FORMAT_FRACTAL_Z_G},
  {"FORMAT_RESERVED", FORMAT_RESERVED},
  {"ALL", FORMAT_ALL},
  {"NULL", FORMAT_NULL}};

static const std::map<DataType, std::string> kDataTypeToStringMap = {
  {DT_UNDEFINED, "DT_UNDEFINED"},            // Used to indicate a DataType field has not been set.
  {DT_FLOAT, "DT_FLOAT"},                    // float type
  {DT_FLOAT16, "DT_FLOAT16"},                // fp16 type
  {DT_INT8, "DT_INT8"},                      // int8 type
  {DT_INT16, "DT_INT16"},                    // int16 type
  {DT_UINT16, "DT_UINT16"},                  // uint16 type
  {DT_UINT8, "DT_UINT8"},                    // uint8 type
  {DT_INT32, "DT_INT32"},                    // uint32 type
  {DT_INT64, "DT_INT64"},                    // int64 type
  {DT_UINT32, "DT_UINT32"},                  // unsigned int32
  {DT_UINT64, "DT_UINT64"},                  // unsigned int64
  {DT_BOOL, "DT_BOOL"},                      // bool type
  {DT_DOUBLE, "DT_DOUBLE"},                  // double type
  {DT_DUAL, "DT_DUAL"},                      // dual output type
  {DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},    // dual output int8 type
  {DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},  // dual output uint8 type
  {DT_COMPLEX64, "DT_COMPLEX64"},            // complex64 type
  {DT_COMPLEX128, "DT_COMPLEX128"},          // complex128 type
  {DT_QINT8, "DT_QINT8"},                    // qint8 type
  {DT_QINT16, "DT_QINT16"},                  // qint16 type
  {DT_QINT32, "DT_QINT32"},                  // qint32 type
  {DT_QUINT8, "DT_QUINT8"},                  // quint8 type
  {DT_QUINT16, "DT_QUINT16"},                // quint16 type
  {DT_RESOURCE, "DT_RESOURCE"},              // resource type
  {DT_STRING_REF, "DT_STRING_REF"},          // string ref type
  {DT_STRING, "DT_STRING"},                  // string type
};

static const std::map<std::string, DataType> kStringTodataTypeMap = {
  {"DT_UNDEFINED", DT_UNDEFINED},  // Used to indicate a DataType field has not been set.
  {"DT_FLOAT", DT_FLOAT},          // float type
  {
    "DT_FLOAT16",
    DT_FLOAT16,
  },                                         // fp16 type
  {"DT_INT8", DT_INT8},                      // int8 type
  {"DT_INT16", DT_INT16},                    // int16 type
  {"DT_UINT16", DT_UINT16},                  // uint16 type
  {"DT_UINT8", DT_UINT8},                    // uint8 type
  {"DT_INT32", DT_INT32},                    // uint32 type
  {"DT_INT64", DT_INT64},                    // int64 type
  {"DT_UINT32", DT_UINT32},                  // unsigned int32
  {"DT_UINT64", DT_UINT64},                  // unsigned int64
  {"DT_BOOL", DT_BOOL},                      // bool type
  {"DT_DOUBLE", DT_DOUBLE},                  // double type
  {"DT_DUAL", DT_DUAL},                      // dual output type
  {"DT_DUAL_SUB_INT8", DT_DUAL_SUB_INT8},    // dual output int8 type
  {"DT_DUAL_SUB_UINT8", DT_DUAL_SUB_UINT8},  // dual output uint8 type
  {"DT_COMPLEX64", DT_COMPLEX64},            // complex64 type
  {"DT_COMPLEX128", DT_COMPLEX128},          // complex128 type
  {"DT_QINT8", DT_QINT8},                    // qint8 type
  {"DT_QINT16", DT_QINT16},                  // qint16 type
  {"DT_QINT32", DT_QINT32},                  // qint32 type
  {"DT_QUINT8", DT_QUINT8},                  // quint8 type
  {"DT_QUINT16", DT_QUINT16},                // quint16 type
  {"DT_RESOURCE", DT_RESOURCE},              // resource type
  {"DT_STRING_REF", DT_STRING_REF},          // string ref type
  {"DT_STRING", DT_STRING},                  // string type
};

static const std::map<ge::DataType, uint32_t> kDataTypeToLength = {
  {DT_BOOL, sizeof(bool)},
  {DT_INT64, sizeof(int64_t)},
  {DT_UINT64, sizeof(int64_t)},
  {DT_FLOAT, sizeof(float)},
  {DT_INT32, sizeof(int32_t)},
  {DT_UINT32, sizeof(int32_t)},
  {DT_INT8, sizeof(char)},
  {DT_UINT8, sizeof(char)},
  {DT_INT16, sizeof(int16_t)},
  {DT_UINT16, sizeof(int16_t)},
  {DT_FLOAT16, sizeof(int16_t)},
  {DT_DOUBLE, sizeof(double)},
  {DT_DUAL, sizeof(float) + sizeof(int8_t)},
  {DT_DUAL_SUB_INT8, sizeof(int8_t)},
  {DT_DUAL_SUB_UINT8, sizeof(uint8_t)},
  {DT_COMPLEX64, sizeof(int64_t)},
  {DT_COMPLEX128, sizeof(int64_t) * 2},
  {DT_QINT8, sizeof(int8_t)},
  {DT_QINT16, sizeof(int16_t)},
  {DT_QINT32, sizeof(int32_t)},
  {DT_QUINT8, sizeof(uint8_t)},
  {DT_QUINT16, sizeof(uint16_t)},
  {DT_STRING_REF, sizeof(uint64_t) * 2},
  {DT_STRING, sizeof(uint64_t)},
  {DT_RESOURCE, sizeof(uint64_t)},
};

static const std::map<domi::FrameworkType, std::string> kFmkTypeToString = {
  {domi::CAFFE, "caffe"},           {domi::MINDSPORE, "mindspore"}, {domi::TENSORFLOW, "tensorflow"},
  {domi::ANDROID_NN, "android_nn"}, {domi::ONNX, "onnx"},           {domi::FRAMEWORK_RESERVED, "framework_reserved"},
};

static const std::map<domi::ImplyType, std::string> kImplyTypeToString = {
  {domi::ImplyType::BUILDIN, "buildin"}, {domi::ImplyType::TVM, "tvm"},        {domi::ImplyType::CUSTOM, "custom"},
  {domi::ImplyType::AI_CPU, "ai_cpu"},   {domi::ImplyType::CCE, "cce"},        {domi::ImplyType::GELOCAL, "gelocal"},
  {domi::ImplyType::HCCL, "hccl"},       {domi::ImplyType::INVALID, "invalid"}};

std::string TypeUtils::ImplyTypeToSerialString(domi::ImplyType imply_type) {
  auto it = kImplyTypeToString.find(imply_type);
  if (it != kImplyTypeToString.end()) {
    return it->second;
  } else {
    GELOGE(GRAPH_FAILED, "ImplyTypeToSerialString: imply_type not support %u", imply_type);
    return "UNDEFINED";
  }
}

bool TypeUtils::IsDataTypeValid(DataType dt) {
  uint32_t num = static_cast<uint32_t>(dt);
  GE_CHK_BOOL_EXEC((num <= DT_UNDEFINED), return false, "The DataType is invalid");
  return true;
}

std::string TypeUtils::DataTypeToSerialString(DataType data_type) {
  auto it = kDataTypeToStringMap.find(data_type);
  if (it != kDataTypeToStringMap.end()) {
    return it->second;
  } else {
    GELOGE(GRAPH_FAILED, "DataTypeToSerialString: datatype not support %u", data_type);
    return "UNDEFINED";
  }
}

DataType TypeUtils::SerialStringToDataType(const std::string &str) {
  auto it = kStringTodataTypeMap.find(str);
  if (it != kStringTodataTypeMap.end()) {
    return it->second;
  } else {
    GELOGE(GRAPH_FAILED, "SerialStringToDataType: datatype not support %s", str.c_str());
    return DT_UNDEFINED;
  }
}

bool TypeUtils::IsFormatValid(Format format) {
  uint32_t num = static_cast<uint32_t>(format);
  GE_CHK_BOOL_EXEC((num <= FORMAT_RESERVED), return false, "The Format is invalid");
  return true;
}

bool TypeUtils::IsInternalFormat(Format format) {
  std::string serial_format = FormatToSerialString(format);
  auto iter = kInternalFormat.find(serial_format);
  bool result = (iter == kInternalFormat.end()) ? false : true;
  return result;
}

std::string TypeUtils::FormatToSerialString(Format format) {
  auto it = kFormatToStringMap.find(format);
  if (it != kFormatToStringMap.end()) {
    return it->second;
  } else {
    GELOGE(GRAPH_FAILED, "Format not support %u", format);
    return "RESERVED";
  }
}
Format TypeUtils::SerialStringToFormat(const std::string &str) {
  auto it = kStringToFormatMap.find(str);
  if (it != kStringToFormatMap.end()) {
    return it->second;
  } else {
    GELOGE(GRAPH_FAILED, "Format not support %s", str.c_str());
    return FORMAT_RESERVED;
  }
}

Format TypeUtils::DataFormatToFormat(const std::string &str) {
  auto it = kDataFormatMap.find(str);
  if (it != kDataFormatMap.end()) {
    return it->second;
  } else {
    GELOGE(GRAPH_FAILED, "Format not support %s", str.c_str());
    return FORMAT_RESERVED;
  }
}

Format TypeUtils::DomiFormatToFormat(domi::domiTensorFormat_t domi_format) {
  auto it = kDomiFormatToGeFormat.find(domi_format);
  if (it != kDomiFormatToGeFormat.end()) {
    return it->second;
  }
  GELOGE(GRAPH_FAILED, "do not find domi Format %d from map", domi_format);
  return FORMAT_RESERVED;
}

std::string TypeUtils::FmkTypeToSerialString(domi::FrameworkType fmk_type) {
  auto it = kFmkTypeToString.find(fmk_type);
  if (it != kFmkTypeToString.end()) {
    return it->second;
  } else {
    GELOGW("Framework type not support %d.", fmk_type);
    return "";
  }
}

static inline void CopyDataFromBuffer(vector<uint8_t> &data, const Buffer &buffer) {
  data.clear();
  if (buffer.GetData() != nullptr && buffer.GetSize() != 0) {
    data.assign(buffer.GetData(), buffer.GetData() + buffer.GetSize());
  }
}

graphStatus Usr2DefQuantizeFactor(const UsrQuantizeFactor &usr, QuantizeFactor &def) {
  def.scale_mode = uint32_t(usr.scale_mode);
  def.set_scale_value(usr.scale_value.data(), usr.scale_value.size());
  def.scale_offset = usr.scale_offset;
  def.set_offset_data_value(usr.offset_data_value.data(), usr.offset_data_value.size());
  def.offset_data_offset = usr.offset_data_offset;
  def.set_offset_weight_value(usr.offset_weight_value.data(), usr.offset_weight_value.size());
  def.offset_weight_offset = usr.offset_weight_offset;
  def.set_offset_pad_value(usr.offset_pad_value.data(), usr.offset_pad_value.size());
  def.offset_pad_offset = usr.offset_pad_offset;
  return GRAPH_SUCCESS;
}
graphStatus Def2UsrQuantizeFactor(const QuantizeFactor &def, UsrQuantizeFactor &usr) {
  usr.scale_mode = UsrQuantizeScaleMode(def.scale_mode);
  CopyDataFromBuffer(usr.scale_value, def.scale_value);
  usr.scale_offset = def.scale_offset;
  CopyDataFromBuffer(usr.offset_data_value, def.offset_data_value);
  usr.offset_data_offset = def.offset_data_offset;
  CopyDataFromBuffer(usr.offset_weight_value, def.offset_weight_value);
  usr.offset_weight_offset = def.offset_weight_offset;
  CopyDataFromBuffer(usr.offset_pad_value, def.offset_pad_value);
  usr.offset_pad_offset = def.offset_pad_offset;
  return GRAPH_SUCCESS;
}
graphStatus Usr2DefUsrQuantizeCalcFactor(const UsrQuantizeCalcFactor &usr, QuantizeCalcFactor &def) {
  def.set_offsetw(usr.offsetw.data(), usr.offsetw.size());
  def.offsetw_offset = usr.offsetw_offset;
  def.set_offsetd(usr.offsetd.data(), usr.offsetd.size());
  def.offsetd_offset = usr.offsetd_offset;
  def.set_scalereq(usr.scalereq.data(), usr.scalereq.size());
  def.scaledreq_offset = usr.scaledreq_offset;
  def.set_offsetdnext(usr.offsetdnext.data(), usr.offsetdnext.size());
  def.offsetdnext_offset = usr.offsetdnext_offset;
  return GRAPH_SUCCESS;
}
graphStatus Def2UsrQuantizeCalcFactor(const QuantizeCalcFactor &def, UsrQuantizeCalcFactor &usr) {
  CopyDataFromBuffer(usr.offsetw, def.offsetw);
  usr.offsetw_offset = def.offsetw_offset;
  CopyDataFromBuffer(usr.offsetd, def.offsetd);
  usr.offsetd_offset = def.offsetd_offset;
  CopyDataFromBuffer(usr.scalereq, def.scalereq);
  usr.scaledreq_offset = def.scaledreq_offset;
  CopyDataFromBuffer(usr.offsetdnext, def.offsetdnext);
  usr.offsetdnext_offset = def.offsetdnext_offset;
  return GRAPH_SUCCESS;
}
graphStatus TypeUtils::Usr2DefQuantizeFactorParams(const UsrQuantizeFactorParams &usr, QuantizeFactorParams &def) {
  def.quantize_algo = uint32_t(usr.quantize_algo);
  def.scale_type = uint32_t(usr.scale_type);
  GE_RETURN_WITH_LOG_IF_ERROR(Usr2DefQuantizeFactor(usr.quantize_param, def.quantize_param),
                              "Usr2DefQuantizeFactor quantize_param failed");
  GE_RETURN_WITH_LOG_IF_ERROR(Usr2DefQuantizeFactor(usr.dequantize_param, def.dequantize_param),
                              "Usr2DefQuantizeFactor dequantize_param failed");
  GE_RETURN_WITH_LOG_IF_ERROR(Usr2DefQuantizeFactor(usr.requantize_param, def.requantize_param),
                              "Usr2DefQuantizeFactor requantize_param failed");
  GE_RETURN_WITH_LOG_IF_ERROR(Usr2DefUsrQuantizeCalcFactor(usr.quantizecalc_param, def.quantizecalc_param),
                              "Usr2DefQuantizeFactor quantizecalc_param failed");
  return GRAPH_SUCCESS;
}
graphStatus TypeUtils::Def2UsrQuantizeFactorParams(const QuantizeFactorParams &def, UsrQuantizeFactorParams &usr) {
  usr.quantize_algo = UsrQuantizeAlgorithm(def.quantize_algo);
  usr.scale_type = UsrQuantizeScaleType(def.scale_type);
  GE_RETURN_WITH_LOG_IF_ERROR(Def2UsrQuantizeFactor(def.quantize_param, usr.quantize_param),
                              "Def2UsrQuantizeFactor quantize_param failed");
  GE_RETURN_WITH_LOG_IF_ERROR(Def2UsrQuantizeFactor(def.dequantize_param, usr.dequantize_param),
                              "Def2UsrQuantizeFactor dequantize_param failed");
  GE_RETURN_WITH_LOG_IF_ERROR(Def2UsrQuantizeFactor(def.requantize_param, usr.requantize_param),
                              "Def2UsrQuantizeFactor requantize_param failed");
  GE_RETURN_WITH_LOG_IF_ERROR(Def2UsrQuantizeCalcFactor(def.quantizecalc_param, usr.quantizecalc_param),
                              "Def2UsrQuantizeCalcFactor quantizecalc_param failed");
  return GRAPH_SUCCESS;
}
bool TypeUtils::GetDataTypeLength(ge::DataType data_type, uint32_t &length) {
  auto it = kDataTypeToLength.find(data_type);
  if (it != kDataTypeToLength.end()) {
    length = it->second;
    return true;
  } else {
    GELOGE(GRAPH_FAILED, "data_type not support %d", data_type);
    return false;
  }
}
bool TypeUtils::CheckUint64MulOverflow(uint64_t a, uint32_t b) {
  // Not overflow
  if (a == 0) {
    return false;
  }
  if ((ULLONG_MAX / a) >= b) {
    return false;
  }
  return true;
}
}  // namespace ge
