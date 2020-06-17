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

# libge_executor.so
# add all proto files, generate corresponding .h and .cc files
# add src files
file(GLOB PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "../../proto/task.proto"
        "../../proto/om.proto"
        "../../proto/insert_op.proto"
        "../../proto/op_mapping_info.proto"
        "../../proto/ge_ir.proto"
        )

file(GLOB SRC_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "ge_executor.cc"
        "../common/ge/plugin_manager.cc"
        "../common/profiling/profiling_manager.cc"
        "../graph/execute/graph_execute.cc"
        "../graph/load/graph_loader.cc"
        "../graph/load/new_model_manager/cpu_queue_schedule.cc"
        "../graph/load/new_model_manager/data_dumper.cc"
        "../graph/load/new_model_manager/data_inputer.cc"
        "../graph/load/new_model_manager/davinci_model.cc"
        "../graph/load/new_model_manager/davinci_model_parser.cc"
        "../graph/load/new_model_manager/model_manager.cc"
        "../graph/load/new_model_manager/model_utils.cc"
        "../graph/load/new_model_manager/task_info/end_graph_task_info.cc"
        "../graph/load/new_model_manager/task_info/event_record_task_info.cc"
        "../graph/load/new_model_manager/task_info/event_wait_task_info.cc"
        "../graph/load/new_model_manager/task_info/fusion_start_task_info.cc"
        "../graph/load/new_model_manager/task_info/fusion_stop_task_info.cc"
        "../graph/load/new_model_manager/task_info/kernel_ex_task_info.cc"
        "../graph/load/new_model_manager/task_info/kernel_task_info.cc"
        "../graph/load/new_model_manager/task_info/label_goto_ex_task_info.cc"
        "../graph/load/new_model_manager/task_info/label_set_task_info.cc"
        "../graph/load/new_model_manager/task_info/label_switch_by_index_task_info.cc"
        "../graph/load/new_model_manager/task_info/memcpy_addr_async_task_info.cc"
        "../graph/load/new_model_manager/task_info/memcpy_async_task_info.cc"
        "../graph/load/new_model_manager/task_info/profiler_trace_task_info.cc"
        "../graph/load/new_model_manager/task_info/stream_active_task_info.cc"
        "../graph/load/new_model_manager/task_info/stream_switch_task_info.cc"
        "../graph/load/new_model_manager/task_info/stream_switchn_task_info.cc"
        "../graph/load/new_model_manager/task_info/super_kernel/super_kernel.cc"
        "../graph/load/new_model_manager/task_info/super_kernel/super_kernel_factory.cc"
        "../graph/load/new_model_manager/task_info/task_info.cc"
        "../graph/load/new_model_manager/tbe_handle_store.cc"
        "../graph/load/new_model_manager/zero_copy_task.cc"
        "../graph/load/output/output.cc"
        "../graph/manager/graph_caching_allocator.cc"
        "../graph/manager/graph_manager_utils.cc"
        "../graph/manager/graph_mem_allocator.cc"
        "../graph/manager/graph_var_manager.cc"
        "../graph/manager/trans_var_data_utils.cc"
        "../graph/manager/util/debug.cc"
        "../hybrid/hybrid_davinci_model_stub.cc"
        "../model/ge_model.cc"
        "../model/ge_root_model.cc"
        "../omm/csa_interact.cc"
        "../single_op/single_op.cc"
        "../single_op/single_op_manager.cc"
        "../single_op/single_op_model.cc"
        "../single_op/stream_resource.cc"
        "../single_op/task/aicpu_task_builder.cc"
        "../single_op/task/build_task_utils.cc"
        "../single_op/task/op_task.cc"
        "../single_op/task/tbe_task_builder.cc"
        )

ge_protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_SOURCE_DIR}/src/ge)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/external/graph)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/graph)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/cce)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

######## libge_executor.so ########
add_library(ge_executor SHARED ${SRC_LIST} ${PROTO_HDRS})
target_compile_definitions(ge_executor PRIVATE
        Werror
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        DAVINCI_SUPPORT_PROFILING
        FMK_HOST_INFER)
target_link_libraries(ge_executor
        ge_common
        graph
        ${PROTOBUF_LIBRARY}
        ${register}
        ${c_sec}
        ${runtime}
        ${slog}
        ${mmpa}
        ${msprof}
        rt
        dl)

