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

project(ut_ge)

set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${GE_SOURCE_DIR}/src/proto/om.proto"
        "${GE_SOURCE_DIR}/src/proto/ge_ir.proto"
        "${GE_SOURCE_DIR}/src/proto/ge_api.proto"
        "${GE_SOURCE_DIR}/src/proto/fwk_adapter.proto"
        "${GE_SOURCE_DIR}/src/proto/op_mapping_info.proto"
        "${GE_SOURCE_DIR}/src/proto/ge_api.proto"
        "${onnx_INC}/onnx/onnx.proto"
        )

ge_protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_SOURCE_DIR})
include_directories(${GE_SOURCE_DIR}/src/ge/inc)
include_directories(${GE_SOURCE_DIR}/src)
include_directories(${GE_SOURCE_DIR}/src/ge)
include_directories(${GE_SOURCE_DIR}/src/common)
include_directories(${GE_SOURCE_DIR}/src/common/graph)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/external/graph)
include_directories(${GE_SOURCE_DIR}/inc/graph)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/inc/common)
include_directories(${GE_SOURCE_DIR}/inc/ops)
include_directories(${GE_SOURCE_DIR}/third_party/eigen)
include_directories(${GE_SOURCE_DIR}/third_party/securec/include)
include_directories(${GE_SOURCE_DIR}/third_party/json/include)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/ops)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/cce)
include_directories(${GE_SOURCE_DIR}/tests/ut/ge)
include_directories(/usr/local/HiAI/opp/op_proto/built-in/inc)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

file(GLOB_RECURSE COMMON_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/common/properties_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/common/ge/plugin_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/common/ge/tbe_plugin_manager.cc"
    "${GE_SOURCE_DIR}/src/common/graph/option/ge_local_context.cc"
    "${GE_SOURCE_DIR}/src/common/graph/option/ge_context.cc"
    "${GE_SOURCE_DIR}/src/ge/common/types.cc"
    "${GE_SOURCE_DIR}/src/ge/common/op_map.cc"
    "${GE_SOURCE_DIR}/src/ge/common/fmk_error_codes.cc"
    "${GE_SOURCE_DIR}/src/ge/common/op/ge_op_utils.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/util/node_searcher/need_rebuild_node_searcher.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/util/variable_accelerate_ctrl.cc"
    "${GE_SOURCE_DIR}/src/ge/opskernel_manager/ops_kernel_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/generator/ge_generator.cc"
    "${GE_SOURCE_DIR}/src/ge/generator/generator_api.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/common/omg_util.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/common/bcast.cc"
    "${GE_SOURCE_DIR}/src/ge/common/util.cc"
    "${GE_SOURCE_DIR}/src/common/graph/ge_attr_define.cc"
    "${GE_SOURCE_DIR}/src/common/graph/anchor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/ge_attr_value.cc"
    "${GE_SOURCE_DIR}/src/common/graph/attr_value.cc"
    "${GE_SOURCE_DIR}/src/common/graph/buffer.cc"
    "${GE_SOURCE_DIR}/src/common/graph/compute_graph.cc"
    "${GE_SOURCE_DIR}/src/common/graph/graph.cc"
    "${GE_SOURCE_DIR}/src/common/graph/inference_context.cc"
    "${GE_SOURCE_DIR}/src/common/graph/shape_refiner.cc"
    "${GE_SOURCE_DIR}/src/common/graph/model.cc"
    "${GE_SOURCE_DIR}/src/common/graph/model_serialize.cc"
    "${GE_SOURCE_DIR}/src/common/graph/node.cc"
    "${GE_SOURCE_DIR}/src/common/graph/op_desc.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator_factory.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator_factory_impl.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator_reg.cc"
    "${GE_SOURCE_DIR}/src/common/graph/range_vistor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/ge_tensor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/tensor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/detail/attributes_holder.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/anchor_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/graph_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/ge_ir_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/node_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/op_desc_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/tensor_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/type_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/debug/graph_debug.cc"
    "${GE_SOURCE_DIR}/src/common/graph/opsproto/opsproto_manager.cc"
    "${GE_SOURCE_DIR}/src/common/graph/op_imp.cc"
    "${GE_SOURCE_DIR}/src/common/register/register.cc"
    "${GE_SOURCE_DIR}/src/common/register/auto_mapping_util.cc"
    "${GE_SOURCE_DIR}/src/common/register/tensor_assign.cc"
    "${GE_SOURCE_DIR}/src/common/graph/format_refiner.cc"
)

file(GLOB_RECURSE COMMON_FORMAT_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/common/fp16_t.cc"
    "${GE_SOURCE_DIR}/src/ge/common/ge_format_util.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/formats.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/datatype_transfer.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_transpose.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_nchw_nc1hwc0.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_fractal_z.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_fractal_nz.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_fractal_zz.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_nc1hwc0_nchw.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_fracz_nchw.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_fracz_nhwc.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/format_transfers/format_transfer_fracz_hwcn.cc"
    "${GE_SOURCE_DIR}/src/ge/common/formats/utils/formats_trans_utils.cc"   
)

file(GLOB_RECURSE GRAPH_OPTIMIZE_COMMON_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/graph/optimize/graph_optimize.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/optimize/summary_optimize.cc"
)


file(GLOB_RECURSE GRAPH_PREPARE_COMMON_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/graph/preprocess/graph_preprocess.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/preprocess/insert_op/util_insert_aipp_op.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/preprocess/insert_op/ge_aipp_op.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/preprocess/insert_op/base_insert_op.cc"
)

file(GLOB_RECURSE GRAPH_PARTITION_COMMON_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/graph/partition/graph_partition.cc"
    "${GE_SOURCE_DIR}/src/ge/plugin/engine/dnnengines.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/partition/engine_place.cc"
)

file(GLOB_RECURSE GRAPH_LOAD_COMMON_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/graph/load/graph_loader.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/graph_manager_utils.cc"
    "${GE_SOURCE_DIR}/src/ge/omm/csa_interact.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/graph_mem_allocator.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/graph_var_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/trans_var_data_utils.cc"
    "${GE_SOURCE_DIR}/src/ge/common/util.cc"
)

file(GLOB_RECURSE DISTINCT_GRAPH_LOAD_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/graph/manager/util/hcom_util.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/util/debug.cc"
    "${GE_SOURCE_DIR}/src/ge/common/properties_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/common/profiling/profiling_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/common/model_parser/base.cc"
    "${GE_SOURCE_DIR}/src/ge/common/tbe_kernel_store.cc"
    "${GE_SOURCE_DIR}/src/ge/common/util.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/data_dumper.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/data_inputer.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/davinci_model.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/davinci_model_parser.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/model_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/model_output.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/model_utils.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/tbe_handle_store.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/event_record_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/event_wait_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/fusion_start_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/fusion_stop_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/hccl_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/kernel_ex_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/kernel_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/label_goto_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/label_set_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/memcpy_async_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/profiler_trace_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/stream_active_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/stream_switch_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/new_model_manager/task_info/end_graph_task_info.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/load/output/output.cc"
    "${GE_SOURCE_DIR}/src/ge/model/ge_model.cc"
    "${GE_SOURCE_DIR}/src/ge/common/helper/model_helper.cc"
    "${GE_SOURCE_DIR}/src/ge/common/helper/om_file_helper.cc"
    "${GE_SOURCE_DIR}/src/ge/common/debug/memory_dumper.cc"
    "${GE_SOURCE_DIR}/src/ge/executor/ge_executor.cc"
    "${GE_SOURCE_DIR}/src/ge/common/auth/file_saver.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/model_manager/event_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/custom/custom_op.cc"
        )

file(GLOB_RECURSE GRAPH_EXECUTE_COMMON_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/graph/execute/graph_execute.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/graph_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/graph_context.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/util/rt_context_util.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/manager/graph_context.h"
    "${GE_SOURCE_DIR}/src/ge/common/thread_pool.cc"
)

file(GLOB_RECURSE GRAPH_BUILD_COMMON_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/graph/build/graph_build.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/build/new_model/task_generator.cc"
    "${GE_SOURCE_DIR}/src/ge/init/gelib.cc"
    "${GE_SOURCE_DIR}/src/ge/client/ge_api.cc"
    "${GE_SOURCE_DIR}/src/ge/session/inner_session.cc"
    "${GE_SOURCE_DIR}/src/ge/session/session_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/engine_manager/dnnengine_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/plugin/engine/engine_manage.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/build/logical_stream_allocator.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/build/stream_allocator.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/build/memory/block_mem_assigner.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/build/memory/binary_block_mem_assigner.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/build/memory/hybrid_mem_assigner.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/build/memory/max_block_mem_assigner.cc"
    "${GE_SOURCE_DIR}/src/ge/model/ge_model.cc"
    "${GE_SOURCE_DIR}/src/ge/common/helper/model_helper.cc"
    "${GE_SOURCE_DIR}/src/ge/common/helper/om_file_helper.cc"
    "${GE_SOURCE_DIR}/src/ge/common/tbe_kernel_store.cc"
    "${GE_SOURCE_DIR}/src/ge/common/model_parser/base.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/build/run_context.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/build/optimize_stream_graph.cc"
)

file(GLOB_RECURSE GRAPH_PASS_COMMON_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/graph/passes/pass_manager.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/base_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/variable_prepare_op_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/variable_ref_delete_op_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/atomic_addr_clean_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/constant_folding_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/iterator_fusion_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/iterator_op_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/net_output_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/update_net_output_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/end_graph_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/node_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/print_op_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/dimension_compute_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/dimension_adjust_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/get_original_format_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/stop_gradient_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/guarantee_const_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/prevent_gradient_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/identity_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/placeholder_with_default_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/snapshot_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/shape_operate_op_remove_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/unused_op_remove_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/isolated_op_remove_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/permute_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/var_is_initialized_op_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/cast_translate_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/dropout_scope_split_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/prune_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/enter_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/next_iteration_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/switch_op_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/pass_utils.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/addn_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/save_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/merge_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/switch_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/switch_logic_remove_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/assert_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/dropout_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/unused_const_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/reshape_remove_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/resource_pair_add_control_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/resource_pair_remove_control_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/transop_breadth_fusion_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/transop_without_reshape_fusion_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/transop_depth_fusion_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/transop_nearby_allreduce_fusion_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/same_transdata_breadth_fusion_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/compile_nodes_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/common/transop_util.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/flow_ctrl_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/optimize/optimizer/allreduce_fusion_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/folding_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/variable_op_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/transpose_transdata_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/hccl_memcpy_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/no_reshape_op_remove_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/no_use_reshape_remove_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/control_op_attr_pass.cc"
    "${GE_SOURCE_DIR}/src/ge/graph/passes/infershape_pass.cc"
)

file(GLOB_RECURSE KERNEL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/broadcast_gradient_args_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/broadcast_args_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/greater_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/gather_v2_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/maximum_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/floormod_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/floordiv_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/range_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/mul_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/shape_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/shape_n_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/size_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/rank_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/fill_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/empty_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/cast_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/transdata_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/permute_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/squeeze_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/reshape_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/reformat_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/kernel_utils.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/expanddims_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/ssd_prior_box_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/pack_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/strided_slice_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/concat_v2_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/add_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/sub_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/reduce_prod_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/rsqrt_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/concat_offset_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/slice_kernel.cc"
"${GE_SOURCE_DIR}/src/ge/graph/passes/folding_kernel/dynamic_stitch_kernel.cc"
)

file(GLOB_RECURSE SINGLE_OP_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/ge/single_op/task/build_task_utils.cc"
    "${GE_SOURCE_DIR}/src/ge/single_op/task/op_task.cc"
    "${GE_SOURCE_DIR}/src/ge/single_op/task/tbe_task_builder.cc"
    "${GE_SOURCE_DIR}/src/ge/single_op/single_op.cc"
    "${GE_SOURCE_DIR}/src/ge/single_op/single_op_model.cc"
    "${GE_SOURCE_DIR}/src/ge/single_op/stream_resource.cc"
    "${GE_SOURCE_DIR}/src/ge/single_op/single_op_manager.cc"
)

# test files
file(GLOB_RECURSE COMMON_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "graph/passes/graph_builder_utils.cc"
    "test.cc"
)

file(GLOB_RECURSE DISTINCT_GRAPH_LOAD_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}
     "graph/load/data_dumper_unittest.cc"
     "graph/load/new_model_manager_data_inputer_unittest.cc"
    "graph/load/new_model_manager_davinci_model_unittest.cc"
    "graph/load/new_model_manager_model_manager_unittest.cc"
    "graph/load/new_model_manager_task_build_unittest.cc"
    "graph/load/end_graph_task_unittest.cc"
    "graph/load/new_model_manager_event_manager_unittest.cc"
    "graph/load/output_net_output_unittest.cc"
    "graph/load/tbe_handle_store_unittest.cc"
    "graph/graph_load_unittest.cc"
    "graph/ge_executor_unittest.cc"
)

file(GLOB_RECURSE PASS_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "graph/passes/prune_pass_unittest.cc"
    "graph/passes/enter_pass_unittest.cc"
    "graph/passes/switch_op_pass_unittest.cc"
    "graph/passes/get_original_format_pass_unittest.cc"
    "graph/passes/pass_manager_unittest.cc"
    "graph/passes/permute_pass_unittest.cc"
    "graph/passes/print_op_pass_unittest.cc"
    "graph/passes/shape_operate_op_remove_pass_unittest.cc"
    "graph/passes/unused_and_isolated_op_remove_pass_unittest.cc"
    "graph/passes/variable_op_pass_unittest.cc"
    "graph/passes/base_pass_unittest.cc"
    "graph/passes/addn_pass_unittest.cc"
    "graph/passes/save_pass_unittest.cc"
    "graph/passes/merge_pass_unittest.cc"
    "graph/passes/switch_pass_unittest.cc"
    "graph/passes/switch_logic_remove_pass_unittest.cc"
    "graph/passes/assert_pass_unittest.cc"
    "graph/passes/dropout_pass_unittest.cc"
    "graph/passes/unused_const_pass_unittest.cc"
    "graph/passes/reshape_remove_pass_unittest.cc"
    "graph/passes/resource_pair_control_pass_unittest.cc"
    "graph/passes/trans_op_breadth_fusion_pass_unittest.cc"
    "graph/passes/trans_op_depth_fusion_pass_unittest.cc"
    "graph/passes/transop_nearby_allreduce_fusion_pass_unittest.cc"
    "graph/passes/constant_folding_pass_unittest.cc"
    "graph/passes/stop_gradient_pass_unittest.cc"
    "graph/passes/prevent_gradient_pass_unittest.cc"
    "graph/passes/identity_pass_unittest.cc"
    "graph/passes/placeholder_with_default_pass_unittest.cc"
    "graph/passes/snapshot_pass_unittest.cc"
    "graph/passes/guarantee_const_pass_unittest.cc"
    "graph/passes/flow_ctrl_pass_unittest.cc"
    "graph/passes/dimension_compute_pass_unittest.cc"
    "graph/passes/variable_prepare_pass_unittest.cc"
    "graph/passes/variable_ref_delete_pass_unittest.cc"
    "graph/passes/dimension_adjust_pass_unittest.cc"
    "graph/passes/pass_utils_unittest.cc"
    "graph/passes/net_output_pass_unittest.cc"
    "graph/passes/update_net_output_pass_unittest.cc"
    "graph/passes/no_reshape_op_remove_pass_unittest.cc"
    "graph/passes/no_use_reshape_remove_pass_unittest.cc"
    "graph/passes/infershape_pass_unittest.cc"
)

file(GLOB_RECURSE KERNEL_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}
    "graph/passes/folding_kernel/greater_kernel_unittest.cc"
    "graph/passes/folding_kernel/maximum_kernel_unittest.cc"
    "graph/passes/folding_kernel/floormod_kernel_unittest.cc"
    "graph/passes/folding_kernel/floordiv_kernel_unittest.cc"
    "graph/passes/folding_kernel/range_kernel_unittest.cc"
    "graph/passes/folding_kernel/mul_kernel_unittest.cc"
    "graph/passes/folding_kernel/shape_kernel_unittest.cc"
    "graph/passes/folding_kernel/shape_n_kernel_unittest.cc"
    "graph/passes/folding_kernel/size_kernel_unittest.cc"
    "graph/passes/folding_kernel/broadcast_args_kernel_unittest.cc"
    "graph/passes/folding_kernel/broadcast_gradient_args_kernel_unittest.cc"
    "graph/passes/folding_kernel/fill_kernel_unittest.cc"
    "graph/passes/folding_kernel/empty_kernel_unittest.cc"
    "graph/passes/folding_kernel/rank_kernel_unittest.cc"
    "graph/passes/folding_kernel/expanddims_kernel_unittest.cc"
    "graph/passes/folding_kernel/reshape_kernel_unittest.cc"
    "graph/passes/folding_kernel/reformat_kernel_unittest.cc"
    "graph/passes/folding_kernel/squeeze_kernel_unittest.cc"
    "graph/passes/folding_kernel/transdata_kernel_unittest.cc"
    "graph/passes/folding_kernel/permute_kernel_unittest.cc"
    "graph/passes/folding_kernel/cast_kernel_unittest.cc"
    "graph/passes/folding_kernel/ssd_prior_box_kernel_unittest.cc"
    "graph/passes/folding_kernel/strided_slice_kernel_unittest.cc"
    "graph/passes/folding_kernel/pack_kernel_unittest.cc"
    "graph/passes/folding_kernel/concat_v2_kernel_unittest.cc"
    "graph/passes/folding_kernel/add_kernel_unittest.cc"
    "graph/passes/folding_kernel/sub_kernel_unittest.cc"
    "graph/passes/folding_kernel/reduce_prod_kernel_unittest.cc"
    "graph/passes/folding_kernel/rsqrt_kernel_unittest.cc"
    "graph/passes/folding_kernel/concat_offset_kernel_unittest.cc"
    "graph/passes/folding_kernel/gather_v2_kernel_unittest.cc"
    "graph/passes/folding_kernel/slice_kernel_unittest.cc"
    "graph/passes/folding_kernel/dynamic_stitch_kernel_unittest.cc"
)

file(GLOB_RECURSE MULTI_PARTS_TEST_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "graph_ir/ge_operator_factory_unittest.cc"
    "graph/transop_util_unittest.cc"
    "common/datatype_transfer_unittest.cc"
    "common/format_transfer_unittest.cc"
    "common/format_transfer_transpose_unittest.cc"
    "common/format_transfer_nchw_5d_unittest.cc"
    "common/format_transfer_nchw_fractalz_unittest.cc"
    "common/format_transfer_hwcn_fractalz_unittest.cc"
    "common/format_transfer_nhwc_fractalz_unittest.cc"
    "common/format_transfer_fractal_nz_unittest.cc"
    "common/format_transfer_fractal_zz_unittest.cc"
    "common/format_transfer_nhwc_5d_unittest.cc"
    "common/format_transfer_5d_nchw_unittest.cc"
    "common/format_transfer_5d_nhwc_unittest.cc"
    "common/format_transfer_hwcn_c1hwncoc0_unittest.cc"
    "common/format_transfer_c1hwncoc0_hwcn_unittest.cc"
    "common/format_transfer_fracz_nchw_unittest.cc"
    "common/format_transfer_fracz_nhwc_unittest.cc"
    "common/format_transfer_fracz_hwcn_unittest.cc"
    "common/ge_format_util_unittest.cc"
    "graph/variable_accelerate_ctrl_unittest.cc"
    "graph/build/logical_stream_allocator_unittest.cc"
    "graph/build/mem_assigner_unittest.cc"
)

file(GLOB_RECURSE SINGLE_OP_TEST_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "single_op/single_op_model_unittest.cc"
    "single_op/single_op_manager_unittest.cc"
    "single_op/stream_resource_unittest.cc"
)

file(GLOB_RECURSE PROFILING_MNG_TEST_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "profiling/ge_profiling_manager_unittest.cc"
)

file(GLOB_RECURSE OTHERS_TEST_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "plugin_manager/ge_util_unittest.cc"
)

list(APPEND COMMON_SHARED_LIBRARIES
    omg_stub
    ${c_sec}
    slog_stub
    cce_ge_stub
    runtime_stub
    profiler_stub
    mmpa_stub
    hccl_stub
)
# build common
add_library(ge_ut_common STATIC ${COMMON_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# build common format
add_library(ge_ut_common_format STATIC ${COMMON_SRC_FILES} ${COMMON_FORMAT_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# build graph prepare common
add_library(ge_prepare_common STATIC ${GRAPH_PREPARE_COMMON_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# build graph optimize common
add_library(ge_optimize_common STATIC ${GRAPH_OPTIMIZE_COMMON_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# build graph partition common
add_library(ge_partition_common STATIC ${GRAPH_PARTITION_COMMON_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# build build graph load common
add_library(ge_load_common STATIC ${GRAPH_LOAD_COMMON_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# build graph execute common
add_library(ge_execute_common STATIC ${GRAPH_EXECUTE_COMMON_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# build graph build common
add_library(ge_build_common STATIC ${GRAPH_BUILD_COMMON_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# build graph pass common
add_library(ge_pass_common STATIC ${GRAPH_PASS_COMMON_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# build single_op common
add_library(ge_single_op STATIC ${SINGLE_OP_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# ut binary

# libge_mutiparts_utest
add_executable(ut_libge_multiparts_utest
    ${COMMON_TEST_FILES}
    ${COMMON_FORMAT_SRC_FILES}
    ${MULTI_PARTS_TEST_FILES}
)
target_link_libraries(ut_libge_multiparts_utest
        ge_build_common ge_load_common ge_build_common ge_execute_common ge_optimize_common ge_partition_common ge_pass_common
    ge_prepare_common ge_single_op ge_ut_common
        graphengine::gtest graphengine::gtest_main ge_protobuf::protobuf rt dl
)
target_link_libraries(ut_libge_multiparts_utest  ${COMMON_SHARED_LIBRARIES} ge_protobuf::protobuf)

# libge_others_utest
add_executable(ut_libge_others_utest
    ${COMMON_TEST_FILES}
    ${COMMON_FORMAT_SRC_FILES}
    ${PASS_TEST_FILES}
    ${EXECUTE_TEST_FILES}
    ${OTHERS_TEST_FILES}
)
target_link_libraries(ut_libge_others_utest
        ge_execute_common ge_load_common ge_pass_common ge_ut_common
        graphengine::gtest graphengine::gtest_main ge_protobuf::protobuf rt dl
)
target_link_libraries(ut_libge_others_utest  ${COMMON_SHARED_LIBRARIES} ge_protobuf::protobuf)

# libge_kernel_utest
add_executable(ut_libge_kernel_utest
        ${COMMON_TEST_FILES}
        ${COMMON_FORMAT_SRC_FILES}
        ${KERNEL_TEST_FILES}
        ${KERNEL_SRC_FILES}
)
target_link_libraries(ut_libge_kernel_utest
        ge_pass_common ge_ut_common  ge_load_common
        graphengine::gtest graphengine::gtest_main ge_protobuf::protobuf rt dl
)
target_link_libraries(ut_libge_kernel_utest  ${COMMON_SHARED_LIBRARIES} ge_protobuf::protobuf)

# libge_distinct_load_utest
add_executable(ut_libge_distinct_load_utest
        ${COMMON_TEST_FILES}
        ${DISTINCT_GRAPH_LOAD_TEST_FILES}
        ${DISTINCT_GRAPH_LOAD_SRC_FILES}
        ${SINGLE_OP_TEST_FILES}
        ${PROFILING_MNG_TEST_FILES}
)
target_link_libraries(ut_libge_distinct_load_utest ${COMMON_SHARED_LIBRARIES}
        ge_execute_common ge_ut_common  ge_ut_common_format  ge_pass_common ge_load_common
        ge_single_op   ge_prepare_common
        ge_optimize_common  ge_build_common ge_partition_common
        graphengine::gtest graphengine::gtest_main ge_protobuf::protobuf rt dl pthread
)
