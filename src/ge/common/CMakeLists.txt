# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# libge_common.so
file(GLOB PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "../../proto/om.proto"
        "../../proto/ge_ir.proto"
        "../../proto/task.proto"
        "../../proto/insert_op.proto"
        )

file(GLOB SRC_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "../model/ge_model.cc"
        "auth/file_saver.cc"
        "context/ctx.cc"
        "cust_aicpu_kernel_store.cc"
        "debug/memory_dumper.cc"
        "dump/dump_properties.cc"
        "fmk_error_codes.cc"
        "formats/format_transfers/datatype_transfer.cc"
        "formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc"
        "formats/format_transfers/format_transfer_dhwcn_fracz3D.cc"
        "formats/format_transfers/format_transfer_dhwnc_fracz3D_transpose.cc"
        "formats/format_transfers/format_transfer_fractal_nz.cc"
        "formats/format_transfers/format_transfer_fractal_z.cc"
        "formats/format_transfers/format_transfer_fractal_zz.cc"
        "formats/format_transfers/format_transfer_fracz_hwcn.cc"
        "formats/format_transfers/format_transfer_fracz_nchw.cc"
        "formats/format_transfers/format_transfer_fracz_nhwc.cc"
        "formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc"
        "formats/format_transfers/format_transfer_nc1hwc0_nchw.cc"
        "formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc"
        "formats/format_transfers/format_transfer_nchw_fz_c04.cc"
        "formats/format_transfers/format_transfer_nchw_nc1hwc0.cc"
        "formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc"
        "formats/format_transfers/format_transfer_transpose.cc"
        "formats/formats.cc"
        "formats/utils/formats_trans_utils.cc"
        "fp16_t.cc"
        "ge/datatype_util.cc"
        "ge/tbe_plugin_manager.cc"
        "ge_format_util.cc"
        "helper/model_helper.cc"
        "helper/om_file_helper.cc"
        "kernel_store.cc"
        "math/fp16_math.cc"
        "model_parser/base.cc"
        "model_saver.cc"
        "op/attr_value_util.cc"
        "op/ge_op_utils.cc"
        "properties_manager.cc"
        "tbe_kernel_store.cc"
        "thread_pool.cc"
        "types.cc"
        "util.cc"
        )

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/op)
include_directories(${GE_SOURCE_DIR}/src/ge)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/common/util)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/external/graph)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/inc/graph)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/cce)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

############ libge_common.so ################
add_library(ge_common SHARED ${SRC_LIST} ${PROTO_HDRS})
target_compile_definitions(ge_common PUBLIC
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        HOST_VISIBILITY
        OS_CENTOS)
target_link_libraries(ge_common
        graph
	#${PROTOBUF_LIBRARY}
	protobuf
        ${register}
        ${c_sec}
        ${slog}
        ${mmpa}
        ${resource}
        ${error_manager}
        rt
        dl)
