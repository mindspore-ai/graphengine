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

# libge_compiler.so & libge_runner.so
# will later be integrated into libgraph_runner.so, works for both training and inference
# compiling proto files generates some warnings, use no-unused-variable to suppress them
set(CMAKE_CXX_FLAGS "-Wno-unused-variable ${CMAKE_CXX_FLAGS}")
file(GLOB PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "../proto/fusion_model.proto"
        "../proto/optimizer_priority.proto"
        )
file(GLOB PROTO_CLIENT_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "../proto/ge_api.proto"
        )
file(GLOB PROTO_HEADER_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "../proto/om.proto"
        "../proto/task.proto"
        "../proto/insert_op.proto"
        "../proto/ge_ir.proto"
        "../proto/fwk_adapter.proto"
        "../proto/op_mapping_info.proto"
        )
ge_protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})
ge_protobuf_generate(ge PROTO_CLIENT_SRCS PROTO_CLIENT_HDRS ${PROTO_CLIENT_LIST})
ge_protobuf_generate(ge PROTO_HEADER_SRCS PROTO_HEADER_HDRS ${PROTO_HEADER_LIST})
# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_SOURCE_DIR})
include_directories(${GE_SOURCE_DIR}/src)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/common/util)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/external/graph)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/inc/framework/common)
include_directories(${GE_SOURCE_DIR}/inc/runtime)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/cce)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

######### libge_runner.so #############
# need to remove dependencies on pb files later
file(GLOB TRAIN_SRC_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "client/ge_api.cc"
        "common/formats/format_transfers/*.cc"
        "common/formats/formats.cc"
        "common/formats/utils/formats_trans_utils.cc"
        "common/fp16_t.cc"
        "common/ge/plugin_manager.cc"
        "common/helper/model_cache_helper.cc"
        "common/profiling/profiling_manager.cc"
        "engine_manager/dnnengine_manager.cc"
        "ge_local_engine/engine/host_cpu_engine.cc"
        "generator/ge_generator.cc"
        "generator/generator_api.cc"
        "graph/build/*.cc"
        "graph/common/*.cc"
        "graph/execute/graph_execute.cc"
        "graph/label/*.cc"
        "graph/load/graph_loader.cc"
        "graph/load/new_model_manager/*.cc"
        "graph/load/new_model_manager/task_info/end_graph_task_info.cc"
        "graph/load/new_model_manager/task_info/event_record_task_info.cc"
        "graph/load/new_model_manager/task_info/event_wait_task_info.cc"
        "graph/load/new_model_manager/task_info/fusion_start_task_info.cc"
        "graph/load/new_model_manager/task_info/fusion_stop_task_info.cc"
        "graph/load/new_model_manager/task_info/hccl_task_info.cc"
        "graph/load/new_model_manager/task_info/kernel_ex_task_info.cc"
        "graph/load/new_model_manager/task_info/kernel_task_info.cc"
        "graph/load/new_model_manager/task_info/label_goto_ex_task_info.cc"
        "graph/load/new_model_manager/task_info/label_set_task_info.cc"
        "graph/load/new_model_manager/task_info/label_switch_by_index_task_info.cc"
        "graph/load/new_model_manager/task_info/memcpy_addr_async_task_info.cc"
        "graph/load/new_model_manager/task_info/memcpy_async_task_info.cc"
        "graph/load/new_model_manager/task_info/profiler_trace_task_info.cc"
        "graph/load/new_model_manager/task_info/stream_active_task_info.cc"
        "graph/load/new_model_manager/task_info/stream_switch_task_info.cc"
        "graph/load/new_model_manager/task_info/stream_switchn_task_info.cc"
        "graph/load/new_model_manager/task_info/super_kernel/super_kernel.cc"
        "graph/load/new_model_manager/task_info/super_kernel/super_kernel_factory.cc"
        "graph/load/new_model_manager/task_info/task_info.cc"
        "graph/load/output/output.cc"
        "graph/manager/*.cc"
        "graph/manager/model_manager/event_manager.cc"
        "graph/manager/util/debug.cc"
        "graph/manager/util/hcom_util.cc"
        "graph/manager/util/rt_context_util.cc"
        "graph/manager/util/variable_accelerate_ctrl.cc"
        "graph/optimize/graph_optimize.cc"
        "graph/optimize/optimizer/allreduce_fusion_pass.cc"
        "graph/optimize/summary_optimize.cc"
        "graph/partition/dynamic_shape_partition.cc"
        "graph/partition/engine_place.cc"
        "graph/partition/graph_partition.cc"
        "graph/passes/*.cc"
        "graph/passes/folding_kernel/add_kernel.cc"
        "graph/passes/folding_kernel/broadcast_args_kernel.cc"
        "graph/passes/folding_kernel/broadcast_gradient_args_kernel.cc"
        "graph/passes/folding_kernel/cast_kernel.cc"
        "graph/passes/folding_kernel/concat_offset_kernel.cc"
        "graph/passes/folding_kernel/concat_v2_kernel.cc"
        "graph/passes/folding_kernel/dynamic_stitch_kernel.cc"
        "graph/passes/folding_kernel/empty_kernel.cc"
        "graph/passes/folding_kernel/expanddims_kernel.cc"
        "graph/passes/folding_kernel/fill_kernel.cc"
        "graph/passes/folding_kernel/floordiv_kernel.cc"
        "graph/passes/folding_kernel/floormod_kernel.cc"
        "graph/passes/folding_kernel/gather_v2_kernel.cc"
        "graph/passes/folding_kernel/greater_kernel.cc"
        "graph/passes/folding_kernel/kernel_utils.cc"
        "graph/passes/folding_kernel/maximum_kernel.cc"
        "graph/passes/folding_kernel/mul_kernel.cc"
        "graph/passes/folding_kernel/pack_kernel.cc"
        "graph/passes/folding_kernel/permute_kernel.cc"
        "graph/passes/folding_kernel/range_kernel.cc"
        "graph/passes/folding_kernel/rank_kernel.cc"
        "graph/passes/folding_kernel/reduce_prod_kernel.cc"
        "graph/passes/folding_kernel/reshape_kernel.cc"
        "graph/passes/folding_kernel/rsqrt_kernel.cc"
        "graph/passes/folding_kernel/shape_kernel.cc"
        "graph/passes/folding_kernel/shape_n_kernel.cc"
        "graph/passes/folding_kernel/size_kernel.cc"
        "graph/passes/folding_kernel/slice_d_kernel.cc"
        "graph/passes/folding_kernel/slice_kernel.cc"
        "graph/passes/folding_kernel/squeeze_kernel.cc"
        "graph/passes/folding_kernel/ssd_prior_box_kernel.cc"
        "graph/passes/folding_kernel/strided_slice_kernel.cc"
        "graph/passes/folding_kernel/sub_kernel.cc"
        "graph/passes/folding_kernel/transdata_kernel.cc"
        "graph/passes/folding_kernel/unpack_kernel.cc"
        "graph/preprocess/graph_preprocess.cc"
        "graph/preprocess/insert_op/ge_aipp_op.cc"
        "graph/preprocess/insert_op/util_insert_aipp_op.cc"
        "graph/preprocess/multi_batch_copy_graph.cc"
        "init/gelib.cc"
        "model/ge_model.cc"
        "omm/csa_interact.cc"
        "opskernel_manager/ops_kernel_manager.cc"
        "session/inner_session.cc"
        "session/session_manager.cc"
        "single_op/*.cc"
        "single_op/task/*.cc"
        )


######### libge_runner.so #############
add_library(ge_runner SHARED ${TRAIN_SRC_LIST} ${PROTO_SRCS} ${PROTO_CLIENT_SRCS} ${PROTO_HEADER_HDRS})
target_compile_definitions(ge_runner PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        DAVINCI_SUPPORT_PROFILING
        REUSE_MEMORY=1
        DAVINCI_CLOUD)
target_link_libraries(ge_runner
        graph
        ge_common
        ge_memory
        ${PROTOBUF_LIBRARY}
        ${register}
        ${c_sec}
        ${slog}
        ${mmpa}
        ${hccl}
        ${msprof}
        ${runtime}
        ${resouce}
        rt
        dl)

######### libge_compiler.so #############
# need to remove dependencies on pb files later
file(GLOB INFER_SRC_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "common/formats/format_transfers/*.cc"
        "common/formats/formats.cc"
        "common/formats/utils/formats_trans_utils.cc"
        "common/fp16_t.cc"
        "common/ge/plugin_manager.cc"
        "common/helper/model_cache_helper.cc"
        "common/profiling/profiling_manager.cc"
        "engine_manager/dnnengine_manager.cc"
        "ge_local_engine/engine/host_cpu_engine.cc"
        "generator/ge_generator.cc"
        "generator/generator_api.cc"
        "graph/build/*.cc"
        "graph/common/*.cc"
        "graph/execute/graph_execute.cc"
        "graph/label/*.cc"
        "graph/load/graph_loader.cc"
        "graph/load/new_model_manager/*.cc"
        "graph/load/new_model_manager/task_info/end_graph_task_info.cc"
        "graph/load/new_model_manager/task_info/event_record_task_info.cc"
        "graph/load/new_model_manager/task_info/event_wait_task_info.cc"
        "graph/load/new_model_manager/task_info/fusion_start_task_info.cc"
        "graph/load/new_model_manager/task_info/fusion_stop_task_info.cc"
        "graph/load/new_model_manager/task_info/kernel_ex_task_info.cc"
        "graph/load/new_model_manager/task_info/kernel_task_info.cc"
        "graph/load/new_model_manager/task_info/label_goto_ex_task_info.cc"
        "graph/load/new_model_manager/task_info/label_set_task_info.cc"
        "graph/load/new_model_manager/task_info/label_switch_by_index_task_info.cc"
        "graph/load/new_model_manager/task_info/memcpy_addr_async_task_info.cc"
        "graph/load/new_model_manager/task_info/memcpy_async_task_info.cc"
        "graph/load/new_model_manager/task_info/profiler_trace_task_info.cc"
        "graph/load/new_model_manager/task_info/stream_active_task_info.cc"
        "graph/load/new_model_manager/task_info/stream_switch_task_info.cc"
        "graph/load/new_model_manager/task_info/stream_switchn_task_info.cc"
        "graph/load/new_model_manager/task_info/super_kernel/super_kernel.cc"
        "graph/load/new_model_manager/task_info/super_kernel/super_kernel_factory.cc"
        "graph/load/new_model_manager/task_info/task_info.cc"
        "graph/load/output/output.cc"
        "graph/manager/*.cc"
        "graph/manager/model_manager/event_manager.cc"
        "graph/manager/util/debug.cc"
        "graph/manager/util/rt_context_util.cc"
        "graph/manager/util/variable_accelerate_ctrl.cc"
        "graph/optimize/graph_optimize.cc"
        "graph/optimize/summary_optimize.cc"
        "graph/partition/dynamic_shape_partition.cc"
        "graph/partition/engine_place.cc"
        "graph/partition/graph_partition.cc"
        "graph/passes/*.cc"
        "graph/passes/folding_kernel/add_kernel.cc"
        "graph/passes/folding_kernel/broadcast_args_kernel.cc"
        "graph/passes/folding_kernel/broadcast_gradient_args_kernel.cc"
        "graph/passes/folding_kernel/cast_kernel.cc"
        "graph/passes/folding_kernel/concat_offset_kernel.cc"
        "graph/passes/folding_kernel/concat_v2_kernel.cc"
        "graph/passes/folding_kernel/dynamic_stitch_kernel.cc"
        "graph/passes/folding_kernel/empty_kernel.cc"
        "graph/passes/folding_kernel/expanddims_kernel.cc"
        "graph/passes/folding_kernel/fill_kernel.cc"
        "graph/passes/folding_kernel/floordiv_kernel.cc"
        "graph/passes/folding_kernel/floormod_kernel.cc"
        "graph/passes/folding_kernel/gather_v2_kernel.cc"
        "graph/passes/folding_kernel/greater_kernel.cc"
        "graph/passes/folding_kernel/kernel_utils.cc"
        "graph/passes/folding_kernel/maximum_kernel.cc"
        "graph/passes/folding_kernel/mul_kernel.cc"
        "graph/passes/folding_kernel/pack_kernel.cc"
        "graph/passes/folding_kernel/permute_kernel.cc"
        "graph/passes/folding_kernel/range_kernel.cc"
        "graph/passes/folding_kernel/rank_kernel.cc"
        "graph/passes/folding_kernel/reduce_prod_kernel.cc"
        "graph/passes/folding_kernel/reshape_kernel.cc"
        "graph/passes/folding_kernel/rsqrt_kernel.cc"
        "graph/passes/folding_kernel/shape_kernel.cc"
        "graph/passes/folding_kernel/shape_n_kernel.cc"
        "graph/passes/folding_kernel/size_kernel.cc"
        "graph/passes/folding_kernel/slice_d_kernel.cc"
        "graph/passes/folding_kernel/slice_kernel.cc"
        "graph/passes/folding_kernel/squeeze_kernel.cc"
        "graph/passes/folding_kernel/ssd_prior_box_kernel.cc"
        "graph/passes/folding_kernel/strided_slice_kernel.cc"
        "graph/passes/folding_kernel/sub_kernel.cc"
        "graph/passes/folding_kernel/transdata_kernel.cc"
        "graph/passes/folding_kernel/transpose_kernel.cc"
        "graph/passes/folding_kernel/unpack_kernel.cc"
        "graph/preprocess/graph_preprocess.cc"
        "graph/preprocess/insert_op/ge_aipp_op.cc"
        "graph/preprocess/insert_op/util_insert_aipp_op.cc"
        "graph/preprocess/multi_batch_copy_graph.cc"
        "init/gelib.cc"
        "ir_build/atc_ir_common.cc"
        "ir_build/ge_ir_build.cc"
        "model/ge_model.cc"
        "omm/csa_interact.cc"
        "opskernel_manager/ops_kernel_manager.cc"
        "session/inner_session.cc"
        "session/session_manager.cc"
        "single_op/*.cc"
        "single_op/task/*.cc"
        )

add_library(ge_compiler SHARED ${INFER_SRC_LIST} ${PROTO_SRCS} ${PROTO_HEADER_HDRS})
target_compile_definitions(ge_compiler PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        REUSE_MEMORY=1
        FMK_HOST_INFER)
target_link_libraries(ge_compiler
        graph
        ge_common
        ge_memory
        ${PROTOBUF_LIBRARY}
        ${register}
        ${c_sec}
        ${slog}
        ${mmpa}
        ${msprof}
        ${runtime}
        ${resouce}
        ${error_manager}
        rt
        dl)
