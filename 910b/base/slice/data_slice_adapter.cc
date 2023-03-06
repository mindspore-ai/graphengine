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
#include "slice/data_slice_adapter.h"
#include <sstream>
#include <map>
#include <set>
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
const std::map<Format, std::vector<std::string>> FORMAT_MAP = {
  {Format::FORMAT_NCHW, {"N", "C", "H", "W"}},
  {Format::FORMAT_NHWC, {"N", "H", "W", "C"}},
  {Format::FORMAT_CHWN, {"C", "H", "W", "N"}},
  {Format::FORMAT_HWCN, {"H", "W", "C", "N"}},
  {Format::FORMAT_NC1HWC0, {"N", "C1", "H", "W", "C0"}},
  {Format::FORMAT_NC1HWC0_C04, {"N", "C1", "H", "W", "C0"}},
  {Format::FORMAT_NCDHW, {"N", "C", "D", "H", "W"}},
  {Format::FORMAT_NDHWC, {"N", "D", "H", "W", "C"}},
  {Format::FORMAT_DHWCN, {"D", "H", "W", "C", "N"}},
  {Format::FORMAT_DHWNC, {"D", "H", "W", "N", "C"}},
  {Format::FORMAT_NDC1HWC0, {"N", "D", "C1", "H", "W", "C0"}}
};
const std::set<Format> FORMAT_4D_SET = {
  Format::FORMAT_NCHW,
  Format::FORMAT_NHWC,
  Format::FORMAT_CHWN,
  Format::FORMAT_HWCN
};
const std::set<Format> FORMAT_5D_SET = {
  Format::FORMAT_NCDHW,
  Format::FORMAT_NDHWC,
  Format::FORMAT_DHWCN,
  Format::FORMAT_DHWNC
};
const std::map<Format, std::string> FORMAT_MAP_STR = {
  {Format::FORMAT_NCHW, "NCHW"},
  {Format::FORMAT_NHWC, "NHWC"},
  {Format::FORMAT_CHWN, "CHWN"},
  {Format::FORMAT_HWCN, "HWCN"},
  {Format::FORMAT_NC1HWC0, "NC1HWC0"},
  {Format::FORMAT_NC1HWC0_C04, "NC1HWC0"},
  {Format::FORMAT_NCHW, "NCHW"},
  {Format::FORMAT_NCDHW, "NCDHW"},
  {Format::FORMAT_NDHWC, "NDHWC"},
  {Format::FORMAT_DHWCN, "DHWCN"},
  {Format::FORMAT_DHWNC, "DHWNC"},
  {Format::FORMAT_NDC1HWC0, "NDC1HWC0"},
  {Format::FORMAT_FRACTAL_NZ, "NZ"},
  {Format::FORMAT_ND, "ND"},
};
constexpr int64_t AXIS_INDEX_1 = 1;
constexpr int64_t AXIS_INDEX_2 = 2;
constexpr int64_t AXIS_INDEX_4 = 4;
constexpr int64_t AXIS_INDEX_5 = 5;
constexpr int64_t DIM_NUM_1 = 1;
constexpr int64_t DIM_NUM_2 = 2;
constexpr int64_t DIM_NUM_3 = 3;
constexpr size_t DIM_NUM_4 = 4;
constexpr size_t DIM_NUM_5 = 5;
constexpr size_t MAX_TYPE_SIZE = 2;
constexpr size_t RANGE_NUM_SIZE = 2;

void DataSliceAdapter::PrintAxisItem(const AxisTypeInfo &axis_type, bool print_ori, std::stringstream &ss)
{
  ss << "{type:" << static_cast<int>(axis_type.GetAxisType());
  ss << ",relate_inputs:[";
  for (const auto &relate_input : axis_type.GetRelateInputs()) {
    ss << "{" << relate_input.first << ",{";
    for (const auto &axis : relate_input.second) {
      ss << axis << ",";
    }
    ss << "}}";
  }
  ss << "],relate_outputs:[";
  for (const auto &relate_output : axis_type.GetRelateOutputs()) {
    ss << "{" << relate_output.first << ",{";
    for (const auto &axis : relate_output.second) {
      ss << axis << ",";
    }
    ss << "}}";
  }
  ss << "],";
  if (print_ori) {
    ss << "ori_relate_inputs:[";
    for (const auto &relate_input : axis_type.GetOriRelateInputs()) {
      ss << "{" << relate_input.first << ",{";
      for (const auto &axis : relate_input.second) {
        ss << axis << ",";
      }
      ss << "}}";
    }
    ss << "],ori_relate_outputs:[";
    for (const auto &relate_output : axis_type.GetOriRelateOutputs()) {
      ss << "{" << relate_output.first << ",{";
      for (const auto &axis : relate_output.second) {
        ss << axis << ",";
      }
      ss << "}}";
    }
    ss << "]";
  }
  ss << "},";
}

void DataSliceAdapter::PrintAxis(const OpDescPtr &op, const std::vector<AxisTypeInfo> &axis_type_vec,
    const std::string &type, bool print_ori)
{
  if (!IsLogEnable(GE, DLOG_DEBUG)) {
    return;
  }
  std::stringstream ss;
  ss << "OpName:[" << op->GetName() << "] " << type << " op_axis_type info:[";
  for (const auto &axis_type : axis_type_vec) {
    PrintAxisItem(axis_type, print_ori, ss);
  }
  ss << "]";
  GELOGD("%s", ss.str().c_str());
}

void DataSliceAdapter::PrintSlice(const OpDescPtr &op, const DataSliceType &slice_info,
    const std::string &tensor_type, const std::string &tag)
{
  if (!IsLogEnable(GE, DLOG_DEBUG)) {
    return;
  }
  std::stringstream ss;
  ss << "OpName[" << op->GetName() << "] " << tag << ":";
  for (size_t tensor_idx = 0; tensor_idx < slice_info.size(); tensor_idx++) {
    ss << tensor_type << "[" << tensor_idx << "]={";
    for (size_t axis_idx = 0; axis_idx < slice_info[tensor_idx].size(); axis_idx++) {
      ss << "{";
      for (size_t range_idx = 0; range_idx < slice_info[tensor_idx][axis_idx].size(); range_idx++) {
        ss << slice_info[tensor_idx][axis_idx][range_idx] << ",";
      }
      ss << "},";
    }
    ss << "};";
  }
  GELOGD("%s", ss.str().c_str());
}

void DataSliceAdapter::PrintOp(const OpDescPtr &op)
{
  if (!IsLogEnable(GE, DLOG_DEBUG)) {
    return;
  }
  OpDesc &op_desc = *(op.get());
  std::string input_str = GetTensorStr(op_desc.GetAllInputsDescPtr());
  GELOGD("OpName[%s] input:%s", op->GetName().c_str(), input_str.c_str());
  std::string output_str = GetTensorStr(op_desc.GetAllOutputsDescPtr());
  GELOGD("OpName[%s] output:%s", op->GetName().c_str(), output_str.c_str());
}

std::string DataSliceAdapter::GetTensorStr(const OpDesc::Vistor<ge::GeTensorDescPtr> all_tensor_desc)
{
  std::stringstream ss;
  for (const auto &tensor : all_tensor_desc) {
    const Format ori_format = tensor->GetOriginFormat();
    const GeShape ori_shape = tensor->GetOriginShape();
    const Format format = static_cast<Format>(GetPrimaryFormat(tensor->GetFormat()));
    const GeShape shape = tensor->GetShape();
    auto iter_ori = FORMAT_MAP_STR.find(ori_format);
    auto iter = FORMAT_MAP_STR.find(format);
    std::string reshape_type;
    (void)AttrUtils::GetStr(tensor, ATTR_NAME_RESHAPE_INFER_TYPE, reshape_type);
    if (iter_ori == FORMAT_MAP_STR.cend() || iter == FORMAT_MAP_STR.cend()) {
      ss << "ori_fomat:" << ori_format << ",ori_shape:" << ori_shape.ToString();
      ss << ",fomat:" << format << ",shape:" << shape.ToString() << ",reshape_type:" << reshape_type << ";";
      continue;
    }
    ss << "ori_fomat:" << iter_ori->second << ",ori_shape:" << ori_shape.ToString();
    ss << ",fomat:" << iter->second << ",shape:" << shape.ToString() << ",reshape_type:" << reshape_type << ";";
  }
  return ss.str();
}

AxisTypeInfo DataSliceAdapter::GetTmpAxisTypeInfo(const AxisTypeInfo &slice_info)
{
  AxisTypeInfo axis_type_info = slice_info;
  axis_type_info.SetRelateInputs(slice_info.GetOriRelateInputs());
  axis_type_info.SetRelateOutputs(slice_info.GetOriRelateOutputs());
  return axis_type_info;
}

Status DataSliceAdapter::GetOriOutputSlice(const OpDescPtr &op, const AxisTypeInfo &slice_info,
    DataSliceType &ori_output_slice)
{
  DataSliceType output_slice;
  for (const auto &tensor_slice : slice_info.GetRelateOutputs()) {
    GeTensorDesc tensor_desc = op->GetOutputDesc(tensor_slice.first);
    std::vector<std::vector<int64_t>> infer_range_vec_res;
    (void)AttrUtils::GetListListInt(tensor_desc, ATTR_NAME_DATA_SLICE, infer_range_vec_res);
    output_slice.emplace_back(infer_range_vec_res);
  }
  PrintSlice(op, output_slice, "output", "current");
  if (TransSliceInfo(op, slice_info, TransType::CUR_TO_ORI, output_slice, ori_output_slice) != SUCCESS) {
    GELOGE(FAILED, "Failed to trans slice info from cur to ori, op_name = %s", op->GetName().c_str());
    return FAILED;
  }
  PrintSlice(op, ori_output_slice, "output", "origin");
  return SUCCESS;
}

Status DataSliceAdapter::GetCurInputSlice(const OpDescPtr &op, const AxisTypeInfo &slice_info,
    const DataSliceType &ori_input_slice, DataSliceType &cur_input_slice)
{
  PrintSlice(op, ori_input_slice, "input", "origin");
  if (TransSliceInfo(op, slice_info, TransType::ORI_TO_CUR, ori_input_slice, cur_input_slice) != SUCCESS) {
    GELOGE(FAILED, "Failed to trans slice info from cur to ori, op_name = %s", op->GetName().c_str());
    return FAILED;
  }
  PrintSlice(op, cur_input_slice, "input", "current");
  return SUCCESS;
}

bool DataSliceAdapter::CheckOriInfo(const OpDescPtr &op)
{
  for (size_t idx = 0; idx < op->GetAllInputsDescPtr().size(); idx++) {
    auto cur_tensor = op->MutableInputDesc(idx);
    if (cur_tensor == nullptr) {
      GELOGW("op_name = %s, input_tensor[%zu] is nullptr", op->GetName().c_str(), idx);
      continue;
    }
    auto ori_shape = cur_tensor->GetOriginShape();
    auto shape = cur_tensor->GetShape();
    if (ori_shape.GetShapeSize() == 0 && shape.GetShapeSize() != 0) {
      GELOGW("op_name = %s, input_tensor[%zu] ori_shape is empty", op->GetName().c_str(), idx);
      return false;
    }
  }
  for (size_t idx = 0; idx < op->GetAllOutputsDescPtr().size(); idx++) {
    auto cur_tensor = op->MutableOutputDesc(idx);
    if (cur_tensor == nullptr) {
      GELOGW("op_name = %s, output_tensor[%zu] is nullptr", op->GetName().c_str(), idx);
      continue;
    }
    auto ori_shape = cur_tensor->GetOriginShape();
    auto shape = cur_tensor->GetShape();
    if (ori_shape.GetShapeSize() == 0 && shape.GetShapeSize() != 0) {
      GELOGW("op_name = %s, output_tensor[%zu] ori_shape is empty", op->GetName().c_str(), idx);
      return false;
    }
  }
  return true;
}

void DataSliceAdapter::SetOriOpInfo(OpDescPtr &op,
    std::vector<std::pair<Format, GeShape>> &cache_input_info,
    std::vector<std::pair<Format, GeShape>> &cache_output_info)
{
  uint32_t input_size = static_cast<uint32_t>(op->GetAllInputsDescPtr().size());
  for (uint32_t idx = 0; idx < input_size; idx++) {
    auto cur_tensor = op->MutableInputDesc(idx);
    if (cur_tensor == nullptr) {
      GELOGW("op_name = %s, input_tensor[%u] is nullptr", op->GetName().c_str(), idx);
      continue;
    }
    cache_input_info.emplace_back(static_cast<Format>(GetPrimaryFormat(cur_tensor->GetFormat())),
                                  cur_tensor->GetShape());
    cur_tensor->SetFormat(cur_tensor->GetOriginFormat());
    cur_tensor->SetShape(cur_tensor->GetOriginShape());
  }
  uint32_t output_size = static_cast<uint32_t>(op->GetAllOutputsDescPtr().size());
  for (uint32_t idx = 0; idx < output_size; idx++) {
    auto cur_tensor = op->MutableOutputDesc(idx);
    if (cur_tensor == nullptr) {
      GELOGW("op_name = %s, output_tensor[%u] is nullptr", op->GetName().c_str(), idx);
      continue;
    }
    cache_output_info.emplace_back(static_cast<Format>(GetPrimaryFormat(cur_tensor->GetFormat())),
                                   cur_tensor->GetShape());
    cur_tensor->SetFormat(cur_tensor->GetOriginFormat());
    cur_tensor->SetShape(cur_tensor->GetOriginShape());
  }
}

void DataSliceAdapter::SetCurOpInfo(OpDescPtr &op,
    const std::vector<std::pair<Format, GeShape>> &cache_input_info,
    const std::vector<std::pair<Format, GeShape>> &cache_output_info)
{
  size_t item_idx = 0;
  uint32_t input_size = static_cast<uint32_t>(op->GetAllInputsDescPtr().size());
  for (uint32_t idx = 0; idx < input_size; idx++) {
    auto cur_tensor = op->MutableInputDesc(static_cast<uint32_t>(idx));
    if (cur_tensor == nullptr) {
      GELOGW("op_name = %s, input_tensor[%u] is nullptr", op->GetName().c_str(), idx);
      continue;
    }
    cur_tensor->SetFormat(cache_input_info[item_idx].first);
    cur_tensor->SetShape(cache_input_info[item_idx].second);
    item_idx++;
  }
  item_idx = 0;
  uint32_t output_size = static_cast<uint32_t>(op->GetAllOutputsDescPtr().size());
  for (uint32_t idx = 0; idx < output_size; idx++) {
    auto cur_tensor = op->MutableOutputDesc(static_cast<uint32_t>(idx));
    if (cur_tensor == nullptr) {
      GELOGW("op_name = %s, output_tensor[%u] is nullptr", op->GetName().c_str(), idx);
      continue;
    }
    cur_tensor->SetFormat(cache_output_info[item_idx].first);
    cur_tensor->SetShape(cache_output_info[item_idx].second);
    item_idx++;
  }
}

std::vector<int64_t> DataSliceAdapter::TransAxisToNZ(const GeTensorDescPtr &tensor, int64_t axis)
{
  const auto ori_shape = tensor->GetOriginShape();
  const int64_t rank = static_cast<int64_t>(ori_shape.GetDims().size());
  std::vector<int64_t> axis_vec;
  if (axis <= rank - DIM_NUM_3) {
    axis_vec.push_back(axis);
  } else if (axis == rank - DIM_NUM_2) {
    axis_vec.push_back(rank - DIM_NUM_1);
    axis_vec.push_back(rank);
  } else if (axis == rank - DIM_NUM_1) {
    axis_vec.push_back(rank - DIM_NUM_2);
    axis_vec.push_back(rank + DIM_NUM_1);
  }
  return axis_vec;
}

bool DataSliceAdapter::CheckReshape(const GeTensorDescPtr &tensor, const std::string &reshape_type,
    int64_t axis, int64_t &format_match_axis)
{
  if (axis >= static_cast<int64_t>(reshape_type.size())) {
    GELOGW("The axis [%ld] >= reshape_type size [%zu]", axis, reshape_type.size());
    return false;
  }
  const auto format = tensor->GetOriginFormat();
  auto iter = FORMAT_MAP.find(format);
  const std::vector<std::string> format_vec = iter->second;
  const std::string reshape_char = reshape_type.substr(axis, 1);
  format_match_axis = std::find(format_vec.cbegin(), format_vec.cend(), reshape_char) - format_vec.cbegin();
  return true;
}

bool DataSliceAdapter::CheckRank(size_t rank, size_t dim_num, const std::string &reshape_type)
{
  if (rank > dim_num) {
    return false;
  } else if (rank < dim_num && rank != reshape_type.size()) {
    return false;
  }
  return true;
}

std::vector<int64_t> DataSliceAdapter::TransAxisForSplit(const GeTensorDescPtr &tensor, const int64_t axis,
    size_t dim_num)
{
  const auto ori_shape = tensor->GetOriginShape();
  const size_t rank = ori_shape.GetDims().size();
  std::string reshape_type;
  (void)AttrUtils::GetStr(tensor, ATTR_NAME_RESHAPE_INFER_TYPE, reshape_type);
  std::vector<int64_t> axis_vec;
  if (!(CheckRank(rank, dim_num, reshape_type))) {
    GELOGW("Failed to CheckRank rank = %zu, dim_num = %zu, reshape_type = %s", rank, dim_num, reshape_type.c_str());
    return axis_vec;
  }

  int64_t format_match_axis = axis;
  if (rank != dim_num && !CheckReshape(tensor, reshape_type, axis, format_match_axis)) {
    GELOGW("Failed to CheckReshape");
    return axis_vec;
  }

  const auto ori_format = tensor->GetOriginFormat();
  const auto format = static_cast<Format>(GetPrimaryFormat(tensor->GetFormat()));
  auto iter = FORMAT_MAP.find(ori_format);
  auto iter_dst = FORMAT_MAP.find(format);
  if (iter == FORMAT_MAP.cend() || iter_dst == FORMAT_MAP.cend()) {
    GELOGW("Cannot find ori_format[%d] or format[%d] in FORMAT_MAP", static_cast<int>(ori_format),
        static_cast<int>(format));
    return axis_vec;
  }

  const std::vector<std::string> ori_format_vec = iter->second;
  if (format_match_axis >= static_cast<int64_t>(ori_format_vec.size())) {
    GELOGW("format_match_axis[%ld] is out of range of format_vec_size[%zu]", format_match_axis, ori_format_vec.size());
    return axis_vec;
  }
  const std::string format_char = ori_format_vec[format_match_axis];
  if (format_char == "C") {
    const std::vector<int64_t> vec_4d = {AXIS_INDEX_1, AXIS_INDEX_4};
    const std::vector<int64_t> vec_5d = {AXIS_INDEX_2, AXIS_INDEX_5};
    axis_vec = (dim_num == DIM_NUM_4) ? vec_4d : vec_5d;
    return axis_vec;
  }

  std::vector<std::string> dst_fmt_vec = iter_dst->second;
  axis_vec.push_back(std::find(dst_fmt_vec.cbegin(), dst_fmt_vec.cend(), format_char) - dst_fmt_vec.cbegin());
  return axis_vec;
}

std::vector<int64_t> DataSliceAdapter::TransAxisForNoSplit(const GeTensorDescPtr &tensor, const int64_t axis,
    size_t dim_num)
{
  const auto ori_shape = tensor->GetOriginShape();
  const size_t rank = ori_shape.GetDims().size();
  std::vector<int64_t> axis_vec;
  if (rank != dim_num) {
    GELOGW("rank[%zu] != to dim_num[%zu] in non_split_axis scene", rank, dim_num);
    return axis_vec;
  }
  const auto ori_format = tensor->GetOriginFormat();
  auto iter = FORMAT_MAP.find(ori_format);
  if (iter == FORMAT_MAP.cend()) {
    GELOGW("Cannot find ori_format[%d] in FORMAT_MAP", static_cast<int>(ori_format));
    return axis_vec;
  }
  const std::vector<std::string> ori_format_vec = iter->second;
  if (axis >= static_cast<int64_t>(ori_format_vec.size())) {
    GELOGW("axis[%ld] is out of range of format_vec_size[%zu]", axis, ori_format_vec.size());
    return axis_vec;
  }
  const std::string format_char = ori_format_vec[axis];
  auto format = static_cast<Format>(GetPrimaryFormat(tensor->GetFormat()));
  auto iter_dst = FORMAT_MAP.find(format);
  if (iter_dst == FORMAT_MAP.cend()) {
    GELOGW("Cannot find format[%d] in FORMAT_MAP", format);
    return axis_vec;
  }
  std::vector<std::string> dst_fmt_vec = iter_dst->second;
  axis_vec.push_back(std::find(dst_fmt_vec.cbegin(), dst_fmt_vec.cend(), format_char) - dst_fmt_vec.cbegin());
  return axis_vec;
}

bool DataSliceAdapter::IsFormatInSet(const Format format, const std::set<Format> &format_set)
{
  auto iter = format_set.find(format);
  return iter != format_set.cend();
}

std::vector<int64_t> DataSliceAdapter::TransAxis(const GeTensorDescPtr &tensor, int64_t ori_axis)
{
  std::vector<int64_t> axis_vec;
  auto ori_format = tensor->GetOriginFormat();
  auto format = static_cast<Format>(GetPrimaryFormat(tensor->GetFormat()));
  if (format == ori_format) {
    axis_vec.push_back(ori_axis);
    return axis_vec;
  } else if (format == Format::FORMAT_FRACTAL_NZ) {
    return TransAxisToNZ(tensor, ori_axis);
  } else if (format == Format::FORMAT_NC1HWC0 || format == Format::FORMAT_NC1HWC0_C04) {
    return TransAxisForSplit(tensor, ori_axis, DIM_NUM_4);
  } else if (format == Format::FORMAT_NDC1HWC0) {
    return TransAxisForSplit(tensor, ori_axis, DIM_NUM_5);
  } else if (IsFormatInSet(format, FORMAT_4D_SET)) {
    return TransAxisForNoSplit(tensor, ori_axis, DIM_NUM_4);
  } else if (IsFormatInSet(format, FORMAT_5D_SET)) {
    return TransAxisForNoSplit(tensor, ori_axis, DIM_NUM_5);
  }
  return axis_vec;
}

Status DataSliceAdapter::FixAxisTypeInfoToOne(AxisTypeInfo &axis_type_info)
{
  size_t count = 0;
  std::vector<CutInfo> input_cut_info_vec = axis_type_info.GetRelateInputs();
  for (const auto &item : input_cut_info_vec) {
    count = (count == 0) ? item.second.size() : count;
    if (count != item.second.size()) {
      GELOGW("The split axis size is not same in all input tensors.");
      return FAILED;
    }
  }
  std::vector<CutInfo> output_cut_info_vec = axis_type_info.GetRelateOutputs();
  for (const auto &item : output_cut_info_vec) {
    bool is_reduce = (axis_type_info.GetAxisType() == AxisType::REDUCESUM) ||
                     (axis_type_info.GetAxisType() == AxisType::REDUCEMAX) ||
                     (axis_type_info.GetAxisType() == AxisType::REDUCEMIN);
    if (item.second.empty() && is_reduce) {
      continue;
    }
    count = (count == 0) ? item.second.size() : count;
    if (count != item.second.size()) {
      GELOGW("The split axis size is not same in all input output tensors.");
      return FAILED;
    }
  }
  for (auto &item : input_cut_info_vec) {
    std::vector<int64_t> &axis_vec = item.second;
    axis_vec = {axis_vec[0]};
  }
  for (auto &item : output_cut_info_vec) {
    std::vector<int64_t> &axis_vec = item.second;
    if (!axis_vec.empty()) {
      axis_vec = {axis_vec[0]};
    }
  }
  axis_type_info.SetRelateInputs(input_cut_info_vec);
  axis_type_info.SetRelateOutputs(output_cut_info_vec);
  return SUCCESS;
}

Status DataSliceAdapter::TransAxisForInputTensor(const OpDescPtr &op, const std::string &axis_type_str,
    AxisTypeInfo &axis_type_info)
{
  std::vector<CutInfo> tmp_relate_puts;
  for (const auto &item : axis_type_info.GetRelateInputs()) {
    std::vector<int64_t> trans_axis_list;
    for (const int64_t axis : item.second) {
      GeTensorDescPtr cur_tensor = nullptr;
      cur_tensor = op->MutableInputDesc(static_cast<uint32_t>(item.first));
      if (cur_tensor == nullptr) {
        GELOGW("op_name = %s, input_tensor[%ld] is nullptr", op->GetName().c_str(), item.first);
        return FAILED;
      }
      const std::vector<int64_t> axis_vec = TransAxis(cur_tensor, axis);
      if (axis_type_str == "reduce_type" && axis_vec.size() > 1) {
        GELOGW("axis_type is reduce_type and axis_vec_size > 1");
        return FAILED;
      }
      if (axis_vec.empty()) {
        GELOGW("TransAxis failed: op_name = %s, input_tensor[%ld], ori_axis[%ld]",
            op->GetName().c_str(), item.first, axis);
        return FAILED;
      }
      trans_axis_list.insert(trans_axis_list.cend(), axis_vec.cbegin(), axis_vec.cend());
    }
    tmp_relate_puts.emplace_back(item.first, trans_axis_list);
  }
  axis_type_info.SetRelateInputs(tmp_relate_puts);
  return SUCCESS;
}

Status DataSliceAdapter::TransAxisForOutputTensor(const OpDescPtr &op, const std::string &axis_type_str,
    AxisTypeInfo &axis_type_info)
{
  std::vector<CutInfo> tmp_relate_puts;
  for (const auto &item : axis_type_info.GetRelateOutputs()) {
    std::vector<int64_t> trans_axis_list;
    for (const int64_t axis : item.second) {
      GeTensorDescPtr cur_tensor = nullptr;
      cur_tensor = op->MutableOutputDesc(static_cast<uint32_t>(item.first));
      if (cur_tensor == nullptr) {
        GELOGW("op_name = %s, output_tensor[%ld] is nullptr", op->GetName().c_str(), item.first);
        return FAILED;
      }
      const std::vector<int64_t> axis_vec = TransAxis(cur_tensor, axis);
      if (axis_type_str == "reduce_type" && axis_vec.size() > 1) {
        GELOGW("axis_type is reduce_type and axis_vec_size > 1");
        return FAILED;
      }
      if (axis_vec.empty()) {
        GELOGW("TransAxis failed: op_name = %s, output_tensor[%ld], ori_axis[%ld]",
            op->GetName().c_str(), item.first, axis);
        return FAILED;
      }
      trans_axis_list.insert(trans_axis_list.cend(), axis_vec.cbegin(), axis_vec.cend());
    }
    tmp_relate_puts.emplace_back(item.first, trans_axis_list);
  }
  axis_type_info.SetRelateOutputs(tmp_relate_puts);
  return SUCCESS;
}

Status DataSliceAdapter::TransByAxisTypeStr(const OpDescPtr &op, const std::string &axis_type_str,
    AxisTypeInfo &axis_type_info)
{
  if (TransAxisForInputTensor(op, axis_type_str, axis_type_info) != SUCCESS) {
    GELOGW("Failed to trans axis type for input tensor.");
    return FAILED;
  }
  if (TransAxisForOutputTensor(op, axis_type_str, axis_type_info) != SUCCESS) {
    GELOGW("Failed to trans axis type for output tensor.");
    return FAILED;
  }
  if (axis_type_str == "element_type" && FixAxisTypeInfoToOne(axis_type_info) != SUCCESS) {
    GELOGW("Fix axis type info to on for element_type failed");
    return FAILED;
  }

  return SUCCESS;
}

void DataSliceAdapter::BackupOriAxisTypeInfo(AxisTypeInfo &axis_type_info)
{
  axis_type_info.SetOriRelateInputs(axis_type_info.GetRelateInputs());
  axis_type_info.SetOriRelateOutputs(axis_type_info.GetRelateOutputs());
}

void DataSliceAdapter::ResetOriAxisTypeInfo(AxisTypeInfo &axis_type_info)
{
  std::vector<CutInfo> relat_inputs;
  std::vector<CutInfo> relat_outputs;
  axis_type_info.SetOriRelateInputs(relat_inputs);
  axis_type_info.SetOriRelateOutputs(relat_outputs);
}

bool DataSliceAdapter::ValidateRelateInputOutput(const AxisTypeInfo &axis_type_info)
{
  if (axis_type_info.GetRelateInputs().size() > 0 && axis_type_info.GetRelateOutputs().size() > 0) {
    return true;
  }
  return false;
}

Status DataSliceAdapter::TransAxisByType(const AxisType axis_type, const OpDescPtr &op,
    AxisTypeInfo &axis_type_info)
{
  if (!ValidateRelateInputOutput(axis_type_info)) {
    GELOGW("ValidateRelateInputOutput failed");
    return FAILED;
  }
  Status ret = SUCCESS;
  BackupOriAxisTypeInfo(axis_type_info);
  switch (axis_type) {
    case AxisType::ELEMENTWISE:
    case AxisType::TRANSPOSE:
    case AxisType::REDUCESUM:
    case AxisType::REDUCEMAX:
    case AxisType::REDUCEMIN:
      ret = TransByAxisTypeStr(op, "element_type", axis_type_info);
      break;
    case AxisType::REDUCEMEAN:
    case AxisType::REDUCEGATHER:
    case AxisType::ELEMENTWITHSHAPEVALUE:
      ret = TransByAxisTypeStr(op, "reduce_type", axis_type_info);
      break;
    case AxisType::SLIDINGWINDOW:
    case AxisType::SLIDINGWINDOWGRAD:
      ret = TransByAxisTypeStr(op, "other_type", axis_type_info);
      break;
    default:
      ret = FAILED;
      GELOGW("Unsupport axis_type = %d", static_cast<int>(axis_type));
      break;
  }
  if (ret != SUCCESS) {
    ResetOriAxisTypeInfo(axis_type_info);
  }
  return ret;
}

AxisType DataSliceAdapter::GetAxisTypeForTransAxis(const AxisTypeInfo &axis_type_info)
{
  const std::vector<AxisType> tmp_vec = axis_type_info.GetAxisTypes();
  std::set<AxisType> tmp_set(tmp_vec.begin(), tmp_vec.end());
  if (tmp_set.size() > MAX_TYPE_SIZE) {
    return AxisType::UNSPLIT;
  }
  if (tmp_set.size() == MAX_TYPE_SIZE &&
      std::find(tmp_set.cbegin(), tmp_set.cend(), AxisType::ELEMENTWISE) != tmp_set.cend() &&
      std::find(tmp_set.cbegin(), tmp_set.cend(), AxisType::REDUCESUM) != tmp_set.cend()) {
    GELOGI("axis_type is ELEMENTWISE+REDUCESUM.");
    return AxisType::SLIDINGWINDOW;
  }
  if (tmp_set.size() == 1) {
    return *tmp_set.cbegin();
  }
  return axis_type_info.GetAxisType();
}

void DataSliceAdapter::TransAxisInfo(const OpDescPtr &op, std::vector<AxisTypeInfo> &axis_type_vec)
{
  for (auto iter = axis_type_vec.begin(); iter != axis_type_vec.end();) {
    AxisType axis_type = GetAxisTypeForTransAxis(*iter);
    if (TransAxisByType(axis_type, op, *iter) == SUCCESS) {
      iter++;
    } else {
      GELOGI("remove one axis type info");
      axis_type_vec.erase(iter);
    }
  }
}

int64_t DataSliceAdapter::SearchOriAxis(const std::vector<CutInfo> &ori_relate, int64_t tensor_idx,
    int64_t axis_idx)
{
  for (const auto &item : ori_relate) {
    if (item.first == tensor_idx) {
      if (axis_idx < static_cast<int64_t>(item.second.size())) {
        return item.second[axis_idx];
      }
    }
  }
  return -1;
}

bool DataSliceAdapter::ValidateAxisIndex(int64_t from_axis,
    const std::vector<std::vector<int64_t>> &slice_info,
    int64_t to_axis, const std::vector<std::vector<int64_t>> &cur_tensor_range)
{
  if (from_axis >= static_cast<int64_t>(slice_info.size()) || slice_info[from_axis].size() != RANGE_NUM_SIZE ||
    to_axis >= static_cast<int64_t>(cur_tensor_range.size())) {
    GELOGE(FAILED, "from_axis:%ld,slice_info_size:%zu,slice_info[%ld].size:%zu,to_axis:%ld, cur_tensor_range_size:%zu",
        from_axis, slice_info.size(), from_axis, slice_info[from_axis].size(), to_axis, cur_tensor_range.size());
    return false;
  }
  return true;
}

Status DataSliceAdapter::TransSliceInfoToOriForElement(const OpDescPtr &op, const AxisTypeInfo &axis_type_info,
    const DataSliceType &slice_info_list, DataSliceType &ori_slice_info_list)
{
  const std::vector<CutInfo> ori_relate_outputs = axis_type_info.GetOriRelateOutputs();
  const std::vector<CutInfo> relate_outputs = axis_type_info.GetRelateOutputs();
  if (ori_relate_outputs.size() == 0 || relate_outputs.size() != slice_info_list.size()) {
    GELOGW("op_name = %s, ori_relate_outputs_size[%zu], relate_outputs_size[%zu], slice_info_list_size[%zu]",
        op->GetName().c_str(), ori_relate_outputs.size(), relate_outputs.size(), slice_info_list.size());
    return FAILED;
  }
  for (size_t index = 0; index < relate_outputs.size(); index++) {
    const int64_t tensor_idx = relate_outputs[index].first;
    const auto output_tensor = op->MutableOutputDesc(static_cast<uint32_t>(tensor_idx));
    if (output_tensor == nullptr) {
      GELOGW("op_name = %s, output_tensor[%ld] is nullptr", op->GetName().c_str(), tensor_idx);
      return FAILED;
    }
    const auto ori_shape = output_tensor->GetOriginShape();
    const size_t rank = ori_shape.GetDims().size();
    std::vector<int64_t> tmp;
    std::vector<std::vector<int64_t>> cur_tensor_range(rank, tmp);
    std::vector<int64_t> axis_vec = relate_outputs[index].second;
    for (size_t idx = 0; idx < axis_vec.size(); idx++) {
      const int64_t ori_axis = SearchOriAxis(ori_relate_outputs, tensor_idx, idx);
      if (ori_axis < 0) {
        GELOGW("op_name = %s, get_ori_axis for output_tensor[%ld] axis[%ld] return ori_axis [-1]",
            op->GetName().c_str(), tensor_idx, idx);
        return FAILED;
      }
      const int64_t cur_axis = axis_vec[idx];
      if (!ValidateAxisIndex(cur_axis, slice_info_list[index], ori_axis, cur_tensor_range)) {
        return FAILED;
      }
      const std::vector<int64_t> transed_axis_list = TransAxis(output_tensor, ori_axis);
      if (transed_axis_list.size() == 0) {
        GELOGW("op_name = %s, TransAxis failed for output_tensor[%ld] ori_axis[%ld]",
            op->GetName().c_str(), tensor_idx, ori_axis);
        return FAILED;
      }
      GeShape cur_shape = output_tensor->GetShape();
      size_t prod_rest_axis = 1;
      for (size_t i = 1; i < transed_axis_list.size(); i++) {
        prod_rest_axis *= (cur_shape.GetDim(transed_axis_list[i]));
      }

      cur_tensor_range[ori_axis] = slice_info_list[index][cur_axis];
      std::vector<int64_t> &ori_slice_piece = cur_tensor_range[ori_axis];
      ori_slice_piece[0] = ori_slice_piece[0] * prod_rest_axis;
      ori_slice_piece[1] = ori_slice_piece[1] * prod_rest_axis + prod_rest_axis - 1;
      ori_slice_piece[1] = std::min(ori_slice_piece[1], ori_shape.GetDim(ori_axis) -1);
    }
    ori_slice_info_list.emplace_back(cur_tensor_range);
  }
  return SUCCESS;
}

Status DataSliceAdapter::TransSliceInfoToCurForElement(const OpDescPtr &op, const AxisTypeInfo &axis_type_info,
    const DataSliceType &slice_info_list, DataSliceType &cur_slice_info_list)
{
  const std::vector<CutInfo> ori_relate_inputs = axis_type_info.GetOriRelateInputs();
  const std::vector<CutInfo> relate_inputs = axis_type_info.GetRelateInputs();
  if (ori_relate_inputs.size() == 0 || relate_inputs.size() != slice_info_list.size()) {
    GELOGW("op_name = %s, ori_relate_inputs_size[%zu], relate_inputs_size[%zu], slice_info_list_size[%zu]",
        op->GetName().c_str(), ori_relate_inputs.size(), relate_inputs.size(), slice_info_list.size());
    return FAILED;
  }
  for (size_t index = 0; index < relate_inputs.size(); index++) {
    const int64_t tensor_idx = relate_inputs[index].first;
    const auto input_tensor = op->MutableInputDesc(static_cast<uint32_t>(tensor_idx));
    if (input_tensor == nullptr) {
      GELOGW("op_name = %s, input_tensor[%ld] is nullptr", op->GetName().c_str(), tensor_idx);
      return FAILED;
    }
    const auto cur_shape = input_tensor->GetShape();
    const size_t rank = cur_shape.GetDims().size();
    std::vector<int64_t> tmp;
    std::vector<std::vector<int64_t>> cur_tensor_range(rank, tmp);
    std::vector<int64_t> axis_vec = relate_inputs[index].second;
    for (size_t idx = 0; idx < axis_vec.size(); idx++) {
      const int64_t ori_axis = SearchOriAxis(ori_relate_inputs, tensor_idx, idx);
      if (ori_axis < 0) {
        GELOGW("op_name = %s, get_ori_axis for input_tensor[%ld] axis[%ld] return ori_axis [-1]",
            op->GetName().c_str(), tensor_idx, idx);
        return FAILED;
      }
      const int64_t cur_axis = axis_vec[idx];
      if (!ValidateAxisIndex(ori_axis, slice_info_list[index], cur_axis, cur_tensor_range)) {
        return FAILED;
      }
      const std::vector<int64_t> transed_axis_list = TransAxis(input_tensor, ori_axis);
      if (transed_axis_list.size() == 0) {
        GELOGW("op_name = %s, TransAxis failed for input_tensor[%ld] ori_axis[%ld]",
            op->GetName().c_str(), tensor_idx, ori_axis);
        return FAILED;
      }

      size_t prod_rest_axis = 1;
      for (size_t i = 1; i < transed_axis_list.size(); i++) {
        prod_rest_axis *= (cur_shape.GetDim(transed_axis_list[i]));
      }
      cur_tensor_range[cur_axis] = slice_info_list[index][ori_axis];
      std::vector<int64_t> &cur_slice_piece = cur_tensor_range[cur_axis];
      cur_slice_piece[0] /= prod_rest_axis;
      cur_slice_piece[1] /= prod_rest_axis;
    }
    cur_slice_info_list.emplace_back(cur_tensor_range);
  }
  return SUCCESS;
}

AxisType DataSliceAdapter::GetAxisTypeForTransSlice(const AxisTypeInfo &axis_type_info)
{
  const std::vector<AxisType> tmp_vec = axis_type_info.GetAxisTypes();
  std::set<AxisType> tmp_set(tmp_vec.begin(), tmp_vec.end());
  if (tmp_set.size() >= MAX_TYPE_SIZE) {
    return AxisType::UNSPLIT;
  }
  if (tmp_set.size() == 1) {
    return *tmp_set.cbegin();
  }
  return axis_type_info.GetAxisType();
}

Status DataSliceAdapter::TransSliceInfo(const OpDescPtr &op, const AxisTypeInfo &axis_type_info,
    TransType trans_type, const DataSliceType &slice_info_list, DataSliceType &out_slice_info_list)
{
  const AxisType axis_type = GetAxisTypeForTransSlice(axis_type_info);
  Status ret = SUCCESS;
  switch (axis_type) {
    case AxisType::ELEMENTWISE:
    case AxisType::REDUCESUM:
    case AxisType::REDUCEMAX:
    case AxisType::REDUCEMIN:
    case AxisType::REDUCEMEAN:
      if (trans_type == TransType::CUR_TO_ORI) {
        ret = TransSliceInfoToOriForElement(op, axis_type_info, slice_info_list, out_slice_info_list);
      } else {
        ret = TransSliceInfoToCurForElement(op, axis_type_info, slice_info_list, out_slice_info_list);
      }
      break;
    case AxisType::SLIDINGWINDOW:
    case AxisType::SLIDINGWINDOWGRAD:
    case AxisType::ELEMENTWITHSHAPEVALUE:
      out_slice_info_list = slice_info_list;
      GELOGI("op_name[%s], axis_type[%d], keep slice info.", op->GetName().c_str(), static_cast<int>(axis_type));
      break;
    default:
      GELOGW("op_name[%s], unsupport axis_type[%d]", op->GetName().c_str(), static_cast<int>(axis_type));
      ret = FAILED;
      break;
  }
  return ret;
}
}
