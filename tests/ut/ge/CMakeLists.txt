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

set(PROTO_LIST
    "${GE_CODE_DIR}/metadef/proto/om.proto"
    "${GE_CODE_DIR}/metadef/proto/ge_ir.proto"
    "${GE_CODE_DIR}/metadef/proto/ge_api.proto"
    "${GE_CODE_DIR}/metadef/proto/insert_op.proto"
    "${GE_CODE_DIR}/metadef/proto/dump_task.proto"
    "${GE_CODE_DIR}/metadef/proto/fwk_adapter.proto"
    "${GE_CODE_DIR}/metadef/proto/op_mapping.proto"
    "${GE_CODE_DIR}/metadef/proto/optimizer_priority.proto"
    "${GE_CODE_DIR}/metadef/proto/ge_api.proto"
    "${GE_CODE_DIR}/metadef/proto/tensorflow/attr_value.proto"
    "${GE_CODE_DIR}/metadef/proto/tensorflow/tensor.proto"
    "${GE_CODE_DIR}/metadef/proto/tensorflow/resource_handle.proto"
    "${GE_CODE_DIR}/metadef/proto/tensorflow/tensor_shape.proto"
    "${GE_CODE_DIR}/metadef/proto/tensorflow/types.proto"
    "${GE_CODE_DIR}/metadef/proto/tensorflow/node_def.proto"
    "${GE_CODE_DIR}/metadef/proto/onnx/ge_onnx.proto"
    )

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_CODE_DIR})
include_directories(${GE_CODE_DIR}/inc)
include_directories(${GE_CODE_DIR}/metadef/inc)
include_directories(${GE_CODE_DIR}/ge)
include_directories(${GE_CODE_DIR}/ge/inc)
include_directories(${GE_CODE_DIR}/ge/ir_build)
include_directories(${GE_CODE_DIR}/metadef)
include_directories(${GE_CODE_DIR}/metadef/graph)
include_directories(${GE_CODE_DIR}/inc/external)
include_directories(${GE_CODE_DIR}/metadef/inc/external)
include_directories(${GE_CODE_DIR}/metadef/inc/external/graph)
include_directories(${GE_CODE_DIR}/metadef/inc/graph)
include_directories(${GE_CODE_DIR}/inc/framework)
include_directories(${GE_CODE_DIR}/metadef/inc/common)
include_directories(${GE_CODE_DIR}/metadef/third_party)
include_directories(${GE_CODE_DIR}/metadef/third_party/transformer/inc)
include_directories(${GE_CODE_DIR}/parser)
include_directories(${GE_CODE_DIR}/parser/parser)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc/cce)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc/ops)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc/opt_info)
include_directories(${GE_CODE_DIR}/tests/ut/ge)
include_directories(${GE_CODE_DIR}/tests/ut/common)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)

set(GRAPH_SRC_FILES
    "${GE_CODE_DIR}/metadef/graph/option/ge_local_context.cc"
    "${GE_CODE_DIR}/metadef/graph/option/ge_context.cc"
    "${GE_CODE_DIR}/metadef/graph/ge_attr_define.cc"
    "${GE_CODE_DIR}/metadef/graph/anchor.cc"
    "${GE_CODE_DIR}/metadef/graph/ge_attr_value.cc"
    "${GE_CODE_DIR}/metadef/graph/attr_value.cc"
    "${GE_CODE_DIR}/metadef/graph/buffer.cc"
    "${GE_CODE_DIR}/metadef/graph/aligned_ptr.cc"
    "${GE_CODE_DIR}/metadef/graph/compute_graph.cc"
    "${GE_CODE_DIR}/metadef/graph/graph.cc"
    "${GE_CODE_DIR}/metadef/graph/gnode.cc"
    "${GE_CODE_DIR}/metadef/graph/ascend_string.cc"
    "${GE_CODE_DIR}/metadef/graph/inference_context.cc"
    "${GE_CODE_DIR}/metadef/graph/shape_refiner.cc"
    "${GE_CODE_DIR}/metadef/graph/model.cc"
    "${GE_CODE_DIR}/metadef/graph/model_serialize.cc"
    "${GE_CODE_DIR}/metadef/graph/node.cc"
    "${GE_CODE_DIR}/metadef/graph/runtime_inference_context.cc"
    "${GE_CODE_DIR}/metadef/graph/op_desc.cc"
    "${GE_CODE_DIR}/metadef/third_party/transformer/src/transfer_shape_according_to_format.cc"
    "${GE_CODE_DIR}/metadef/third_party/transformer/src/axis_util.cc"
    "${GE_CODE_DIR}/metadef/third_party/transformer/src/expand_dimension.cc"
    "${GE_CODE_DIR}/metadef/graph/operator.cc"
    "${GE_CODE_DIR}/metadef/graph/operator_factory.cc"
    "${GE_CODE_DIR}/metadef/graph/operator_factory_impl.cc"
    "${GE_CODE_DIR}/metadef/graph/ge_tensor.cc"
    "${GE_CODE_DIR}/metadef/graph/ref_relation.cc"
    "${GE_CODE_DIR}/metadef/graph/tensor.cc"
    "${GE_CODE_DIR}/metadef/graph/types.cc"
    "${GE_CODE_DIR}/metadef/graph/detail/attributes_holder.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/anchor_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/graph_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/ge_ir_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/node_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/op_desc_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/tensor_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/type_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/transformer_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/debug/graph_debug.cc"
    "${GE_CODE_DIR}/metadef/graph/opsproto/opsproto_manager.cc"
    "${GE_CODE_DIR}/metadef/ops/op_imp.cpp"
    "${GE_CODE_DIR}/metadef/register/register.cpp"
    "${GE_CODE_DIR}/metadef/register/register_pass.cpp"
    "${GE_CODE_DIR}/metadef/register/op_kernel_registry.cpp"
    "${GE_CODE_DIR}/metadef/register/auto_mapping_util.cpp"
    "${GE_CODE_DIR}/metadef/register/tensor_assign.cpp"
    "${GE_CODE_DIR}/metadef/register/register_format_transfer.cc"
    "${GE_CODE_DIR}/metadef/graph/format_refiner.cc"
    "${GE_CODE_DIR}/metadef/register/ops_kernel_builder_registry.cc"
    "${GE_CODE_DIR}/metadef/register/op_tiling.cpp"
    "${GE_CODE_DIR}/metadef/graph/utils/tuning_utils.cc"
    "${GE_CODE_DIR}/metadef/register/op_tiling_registry.cpp"
    "${GE_CODE_DIR}/metadef/register/op_tiling_registry_impl.cpp"
)

set(PARSER_SRC_FILES
    "${GE_CODE_DIR}/parser/parser/common/op_map.cc"
    "${GE_CODE_DIR}/parser/parser/common/pre_checker.cc"
    "${GE_CODE_DIR}/parser/parser/common/convert/pb2json.cc"
    "${GE_CODE_DIR}/parser/parser/common/parser_factory.cc"
    "${GE_CODE_DIR}/parser/parser/common/model_saver.cc"
    "${GE_CODE_DIR}/parser/parser/common/parser_types.cc"
    "${GE_CODE_DIR}/parser/parser/common/parser_inner_ctx.cc"
)

set(COMMON_SRC_FILES
    "${GE_CODE_DIR}/ge/common/properties_manager.cc"
    "${GE_CODE_DIR}/ge/common/ge/plugin_manager.cc"
    "${GE_CODE_DIR}/ge/common/ge/tbe_plugin_manager.cc"
    "${GE_CODE_DIR}/ge/common/types.cc"
    "${GE_CODE_DIR}/ge/common/fmk_error_codes.cc"
    "${GE_CODE_DIR}/ge/common/op/ge_op_utils.cc"
    "${GE_CODE_DIR}/ge/common/context/ctx.cc"
    "${GE_CODE_DIR}/ge/graph/manager/util/variable_accelerate_ctrl.cc"
    "${GE_CODE_DIR}/ge/opskernel_manager/ops_kernel_manager.cc"
    "${GE_CODE_DIR}/ge/generator/ge_generator.cc"
    "${GE_CODE_DIR}/ge/generator/generator_api.cc"
    "${GE_CODE_DIR}/ge/graph/common/omg_util.cc"
    "${GE_CODE_DIR}/ge/graph/common/bcast.cc"
    "${GE_CODE_DIR}/ge/common/util.cc"
    "${GE_CODE_DIR}/ge/common/ge/op_tiling_manager.cc"
    "${GE_CODE_DIR}/ge/init/gelib.cc"
    "${GE_CODE_DIR}/ge/engine_manager/dnnengine_manager.cc"
    "${GE_CODE_DIR}/ge/opskernel_manager/ops_kernel_manager.cc"
    "${GE_CODE_DIR}/ge/session/session_manager.cc"
    "${GE_CODE_DIR}/ge/opskernel_manager/ops_kernel_builder_manager.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/model_manager.cc"
    "${GE_CODE_DIR}/ge/common/profiling/profiling_manager.cc"
    "${GE_CODE_DIR}/ge/common/profiling/ge_profiling.cc"
    "${GE_CODE_DIR}/ge/graph/manager/host_mem_manager.cc"
	"${GE_CODE_DIR}/ge/graph/manager/memory_api.cc"
    "${GE_CODE_DIR}/ge/session/inner_session.cc"
    "${GE_CODE_DIR}/ge/graph/manager/util/rt_context_util.cc"
    "${GE_CODE_DIR}/ge/graph/execute/graph_execute.cc"
    "${GE_CODE_DIR}/ge/graph/preprocess/graph_preprocess.cc"
    "${GE_CODE_DIR}/ge/hybrid/hybrid_davinci_model_stub.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/davinci_model.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/data_inputer.cc"
    "${GE_CODE_DIR}/ge/common/dump/dump_properties.cc"
    "${GE_CODE_DIR}/ge/common/helper/model_helper.cc"
    "${GE_CODE_DIR}/ge/common/dump/dump_manager.cc"
    "${GE_CODE_DIR}/ge/common/dump/exception_dumper.cc"
    "${GE_CODE_DIR}/ge/common/dump/opdebug_register.cc"
    "${GE_CODE_DIR}/ge/common/dump/dump_op.cc"
    "${GE_CODE_DIR}/ge/common/ge_opt_info.cc"
    "${GE_CODE_DIR}/ge/common/helper/om_file_helper.cc"
    "${GE_CODE_DIR}/ge/model/ge_root_model.cc"
    "${GE_CODE_DIR}/ge/common/model_parser/model_parser.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/data_dumper.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_manager.cc"
    "${GE_CODE_DIR}/ge/common/dump/dump_server.cc"
    "${GE_CODE_DIR}/ge/graph/preprocess/insert_op/util_insert_aipp_op.cc"
    "${GE_CODE_DIR}/ge/graph/preprocess/multi_batch_copy_graph.cc"
    "${GE_CODE_DIR}/ge/graph/optimize/mem_rw_conflict_optimize.cc"
    "${GE_CODE_DIR}/ge/graph/passes/pass_manager.cc"
    "${GE_CODE_DIR}/ge/graph/passes/resource_pair_add_control_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/resource_pair_remove_control_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/pass_utils.cc"
    "${GE_CODE_DIR}/ge/graph/passes/base_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/bitcast_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/constant_folding_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/aicpu_constant_folding_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/reshape_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/reshape_recovery_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transop_breadth_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transop_depth_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transop_nearby_allreduce_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/same_transdata_breadth_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transop_without_reshape_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/compile_nodes_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/variable_prepare_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/variable_ref_delete_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/variable_ref_useless_control_out_delete_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/subgraph_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/data_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/net_output_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/replace_transshape_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/constant_fuse_same_pass.cc"
	"${GE_CODE_DIR}/ge/graph/passes/fuse_data_nodes_with_common_input_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/print_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/no_use_reshape_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/iterator_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/input_output_connection_identify_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/atomic_addr_clean_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/mark_same_addr_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/mark_graph_unknown_status_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/mark_agnostic_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/dimension_compute_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/dimension_adjust_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/get_original_format_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/shape_operate_op_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/assert_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/dropout_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/infershape_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/unused_const_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/permute_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/ctrl_edge_transfer_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/end_of_sequence_add_control_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/stop_gradient_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/prevent_gradient_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/identity_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/ref_identity_delete_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/placeholder_with_default_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/snapshot_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/guarantee_const_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/var_is_initialized_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/parallel_concat_start_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/folding_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/cast_translate_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/prune_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/merge_to_stream_merge_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/merge_input_memcpy_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/switch_to_stream_switch_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/mark_force_unknown_for_cond_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/attach_stream_label_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/multi_batch_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/multi_batch_clone_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/subexpression_migration_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/subgraph_const_migration_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/unused_args_clean_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/next_iteration_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/control_trigger_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/cond_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/cond_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/for_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/enter_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/assign_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/inplace_support_check_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/addn_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/common_subexpression_elimination_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transop_symmetry_elimination_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/save_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/switch_dead_branch_elimination.cc"
    "${GE_CODE_DIR}/ge/graph/passes/switch_logic_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/switch_data_edges_bypass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/merge_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/variable_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/cast_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transpose_transdata_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/hccl_memcpy_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/hccl_continuous_memcpy_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/flow_ctrl_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/global_step_insert_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/link_gen_mask_nodes_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/replace_with_empty_const_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/hccl_group_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/hccl_tailing_optimization_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/memcpy_addr_async_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/set_input_output_offset_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/remove_same_const_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/useless_control_out_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/parallel_group_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/buffer_pool_memory_pass.cc"
	"${GE_CODE_DIR}/ge/graph/passes/mark_node_unknown_shape_pass.cc"
    "${GE_CODE_DIR}/ge/model/ge_model.cc"
    "${GE_CODE_DIR}/ge/common/cust_aicpu_kernel_store.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/model_utils.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/zero_copy_offset.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/zero_copy_task.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/cpu_queue_schedule.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/aipp_utils.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/tbe_handle_store.cc"
    "${GE_CODE_DIR}/ge/common/kernel_store.cc"
    "${GE_CODE_DIR}/ge/common/tbe_kernel_store.cc"
    "${GE_CODE_DIR}/ge/common/auth/file_saver.cc"
    "${GE_CODE_DIR}/ge/graph/manager/util/debug.cc"
    "${GE_CODE_DIR}/ge/common/debug/memory_dumper.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_context.cc"
    "${GE_CODE_DIR}/ge/graph/load/graph_loader.cc"
    "${GE_CODE_DIR}/ge/graph/optimize/graph_optimize.cc"
    "${GE_CODE_DIR}/ge/graph/build/graph_builder.cc"
    "${GE_CODE_DIR}/ge/graph/partition/graph_partition.cc"
    "${GE_CODE_DIR}/ge/common/helper/model_cache_helper.cc"
    "${GE_CODE_DIR}/ge/ir_build/ge_ir_build.cc"
    "${GE_CODE_DIR}/ge/ir_build/attr_options/utils.cc"
    "${GE_CODE_DIR}/ge/ir_build/attr_options/keep_dtype_option.cc"
    "${GE_CODE_DIR}/ge/ir_build/attr_options/weight_compress_option.cc"
    "${GE_CODE_DIR}/ge/graph/build/label_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/partition/stage_partition.cc"
    "${GE_CODE_DIR}/ge/graph/partition/dynamic_shape_partition.cc"
    "${GE_CODE_DIR}/ge/graph/optimize/summary_optimize.cc"
    "${GE_CODE_DIR}/ge/ir_build/option_utils.cc"
    "${GE_CODE_DIR}/ge/graph/preprocess/insert_op/ge_aipp_op.cc"
    "${GE_CODE_DIR}/ge/graph/preprocess/multi_batch_options.cc"
    "${GE_CODE_DIR}/ge/graph/build/model_builder.cc"
    "${GE_CODE_DIR}/ge/graph/build/run_context.cc"
    "${GE_CODE_DIR}/ge/graph/build/stream_graph_optimizer.cc"
    "${GE_CODE_DIR}/ge/graph/build/task_generator.cc"
    "${GE_CODE_DIR}/ge/graph/partition/graph_partition.cc"
    "${GE_CODE_DIR}/ge/graph/partition/engine_place.cc"
    "${GE_CODE_DIR}/ge/graph/build/stream_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/memory_assigner.cc"
    "${GE_CODE_DIR}/ge/graph/build/logical_stream_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/graph_mem_assigner.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/var_mem_assign_util.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/hybrid_mem_assigner.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/block_mem_assigner.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/binary_block_mem_assigner.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/max_block_mem_assigner.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/buffer_pool_mem_assigner.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_mem_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_var_manager.cc"
    "${GE_CODE_DIR}/ge/analyzer/analyzer.cc"
    "${GE_CODE_DIR}/ge/common/thread_pool.cc"
    "${GE_CODE_DIR}/ge/graph/common/transop_util.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_manager_utils.cc"
    "${GE_CODE_DIR}/ge/graph/manager/trans_var_data_utils.cc"
    "${GE_CODE_DIR}/ge/graph/common/local_context.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_caching_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/manager/session_scope_mem_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/manager/rdma_pool_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/manager/host_mem_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_mem_manager.cc"
    "${GE_CODE_DIR}/ge/common/dump/dump_op.cc"
    "${GE_CODE_DIR}/ge/common/model_saver.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/aicpu/aicpu_ext_info.cc"
    "${GE_CODE_DIR}/ge/common/ge/datatype_util.cc"
    "${GE_CODE_DIR}/ge/ge_local_engine/engine/host_cpu_engine.cc"
    "${GE_CODE_DIR}/ge/session/omg.cc"
)

set(COMMON_FORMAT_SRC_FILES
    "${GE_CODE_DIR}/ge/common/fp16_t.cc"
    "${GE_CODE_DIR}/ge/common/ge_format_util.cc"
    "${GE_CODE_DIR}/ge/common/formats/formats.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/datatype_transfer.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_transpose.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_nchw_nc1hwc0.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fractal_z.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fractal_nz.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fractal_zz.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_nc1hwc0_nchw.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fracz_nchw.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fracz_nhwc.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fracz_hwcn.cc"
    "${GE_CODE_DIR}/ge/common/formats/utils/formats_trans_utils.cc"
    "${GE_CODE_DIR}/ge/graph/manager/util/hcom_util.cc"
    "${GE_CODE_DIR}/ge/common/dump/dump_manager.cc"
)

set(GRAPH_OPTIMIZE_COMMON_SRC_FILES
    "${GE_CODE_DIR}/ge/graph/optimize/graph_optimize.cc"
    "${GE_CODE_DIR}/ge/graph/optimize/summary_optimize.cc"
    "${GE_CODE_DIR}/ge/graph/optimize/mem_rw_conflict_optimize.cc"
)


set(GRAPH_PREPARE_COMMON_SRC_FILES
    "${GE_CODE_DIR}/ge/graph/preprocess/graph_preprocess.cc"
    "${GE_CODE_DIR}/ge/graph/preprocess/insert_op/util_insert_aipp_op.cc"
    "${GE_CODE_DIR}/ge/graph/preprocess/insert_op/ge_aipp_op.cc"
    #"${GE_CODE_DIR}/ge/graph/preprocess/insert_op/base_insert_op.cc"
)

set(GRAPH_PARTITION_COMMON_SRC_FILES
    "${GE_CODE_DIR}/ge/graph/partition/graph_partition.cc"
    "${GE_CODE_DIR}/ge/plugin/engine/dnnengines.cc"
    "${GE_CODE_DIR}/ge/graph/partition/engine_place.cc"
)

set(GRAPH_LOAD_COMMON_SRC_FILES
    "${GE_CODE_DIR}/ge/graph/load/graph_loader.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_manager_utils.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_mem_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_var_manager.cc"
    "${GE_CODE_DIR}/ge/graph/manager/trans_var_data_utils.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_caching_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/manager/session_scope_mem_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/manager/rdma_pool_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/manager/host_mem_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_mem_manager.cc"
    "${GE_CODE_DIR}/ge/common/thread_pool.cc"
)

set(DISTINCT_GRAPH_LOAD_SRC_FILES
    "${GE_CODE_DIR}/ge/graph/manager/util/hcom_util.cc"
    "${GE_CODE_DIR}/ge/graph/manager/util/debug.cc"
    "${GE_CODE_DIR}/ge/common/properties_manager.cc"
    "${GE_CODE_DIR}/ge/common/profiling/profiling_manager.cc"
    "${GE_CODE_DIR}/ge/common/model_parser/model_parser.cc"
    "${GE_CODE_DIR}/ge/common/tbe_kernel_store.cc"
    "${GE_CODE_DIR}/ge/common/util.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/cpu_queue_schedule.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/data_dumper.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/data_inputer.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/davinci_model.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/model_manager.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/model_utils.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/tbe_handle_store.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/event_record_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/event_wait_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/fusion_start_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/fusion_stop_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/hccl_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/kernel_ex_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/kernel_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/label_set_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/label_goto_ex_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/label_switch_by_index_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/memcpy_addr_async_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/memcpy_async_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/profiler_trace_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/stream_active_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/end_graph_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/model_exit_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/ffts_task_info.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/super_kernel/super_kernel.cc"
    "${GE_CODE_DIR}/ge/graph/load/model_manager/task_info/super_kernel/super_kernel_factory.cc"
    "${GE_CODE_DIR}/ge/model/ge_model.cc"
    "${GE_CODE_DIR}/ge/common/helper/om_file_helper.cc"
    "${GE_CODE_DIR}/ge/common/debug/memory_dumper.cc"
    "${GE_CODE_DIR}/ge/executor/ge_executor.cc"
    "${GE_CODE_DIR}/ge/common/auth/file_saver.cc"
    "${GE_CODE_DIR}/ge/graph/manager/model_manager/event_manager.cc"
)

set(GRAPH_EXECUTE_COMMON_SRC_FILES
    "${GE_CODE_DIR}/ge/graph/execute/graph_execute.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_manager.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_context.cc"
    "${GE_CODE_DIR}/ge/graph/manager/util/rt_context_util.cc"
    "${GE_CODE_DIR}/ge/graph/manager/graph_context.h"
)

set(GRAPH_BUILD_COMMON_SRC_FILES
    "${GE_CODE_DIR}/ge/graph/build/graph_builder.cc"
    "${GE_CODE_DIR}/ge/graph/build/task_generator.cc"
    "${GE_CODE_DIR}/ge/client/ge_api.cc"
    "${GE_CODE_DIR}/ge/session/inner_session.cc"
    "${GE_CODE_DIR}/ge/session/session_manager.cc"
    "${GE_CODE_DIR}/ge/engine_manager/dnnengine_manager.cc"
    "${GE_CODE_DIR}/ge/plugin/engine/engine_manage.cc"
    "${GE_CODE_DIR}/ge/graph/build/logical_stream_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/build/stream_allocator.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/block_mem_assigner.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/binary_block_mem_assigner.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/hybrid_mem_assigner.cc"
    "${GE_CODE_DIR}/ge/graph/build/memory/max_block_mem_assigner.cc"
    "${GE_CODE_DIR}/ge/model/ge_model.cc"
    "${GE_CODE_DIR}/ge/common/helper/om_file_helper.cc"
    "${GE_CODE_DIR}/ge/common/tbe_kernel_store.cc"
    "${GE_CODE_DIR}/ge/common/thread_pool.cc"
    "${GE_CODE_DIR}/ge/common/model_parser/model_parser.cc"
    "${GE_CODE_DIR}/ge/graph/build/run_context.cc"
    "${GE_CODE_DIR}/ge/graph/common/local_context.cc"
)

set(GRAPH_PASS_COMMON_SRC_FILES
    "${GE_CODE_DIR}/ge/graph/passes/pass_manager.cc"
    "${GE_CODE_DIR}/ge/graph/passes/base_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/variable_prepare_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/variable_ref_delete_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/atomic_addr_clean_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/constant_folding_pass.cc"
    "${GE_CODE_DIR}/parser/parser/tensorflow/iterator_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/iterator_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/net_output_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/print_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/dimension_compute_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/dimension_adjust_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/get_original_format_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/stop_gradient_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/guarantee_const_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/prevent_gradient_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/identity_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/placeholder_with_default_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/snapshot_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/shape_operate_op_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/permute_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/var_is_initialized_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/cast_translate_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/prune_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/enter_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/next_iteration_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/pass_utils.cc"
    "${GE_CODE_DIR}/ge/graph/passes/addn_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/save_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/merge_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/switch_logic_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/assert_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/dropout_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/unused_const_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/reshape_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/resource_pair_add_control_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/resource_pair_remove_control_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/remove_same_const_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/useless_control_out_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transop_breadth_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transop_without_reshape_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transop_depth_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transop_nearby_allreduce_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/same_transdata_breadth_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/compile_nodes_pass.cc"
    "${GE_CODE_DIR}/ge/graph/common/transop_util.cc"
    "${GE_CODE_DIR}/ge/graph/passes/flow_ctrl_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/parallel_group_pass.cc"
    #"${GE_CODE_DIR}/ge/graph/optimize/optimizer/allreduce_fusion_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/folding_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/variable_op_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/transpose_transdata_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/hccl_memcpy_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/no_use_reshape_remove_pass.cc"
    "${GE_CODE_DIR}/ge/graph/passes/infershape_pass.cc"
    "${GE_CODE_DIR}/ge/ge_local_engine/engine/host_cpu_engine.cc"
    "${GE_CODE_DIR}/ge/analyzer/analyzer.cc"
    "${GE_CODE_DIR}/ge/graph/passes/net_output_pass.cc"
    "${GE_CODE_DIR}/ge/graph/common/local_context.cc"
)

set(KERNEL_SRC_FILES
    "${GE_CODE_DIR}/ge/host_kernels/broadcast_gradient_args_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/broadcast_args_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/greater_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/gather_v2_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/maximum_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/floormod_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/floordiv_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/range_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/mul_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/shape_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/shape_n_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/size_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/rank_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/fill_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/empty_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/cast_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/transdata_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/permute_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/squeeze_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/reshape_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/reformat_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/kernel_utils.cc"
    "${GE_CODE_DIR}/ge/host_kernels/expanddims_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/ssd_prior_box_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/pack_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/strided_slice_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/concat_v2_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/add_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/sub_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/reduce_prod_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/rsqrt_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/concat_offset_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/slice_kernel.cc"
    "${GE_CODE_DIR}/ge/host_kernels/dynamic_stitch_kernel.cc"
)

set(SINGLE_OP_SRC_FILES
    "${GE_CODE_DIR}/ge/single_op/task/build_task_utils.cc"
    "${GE_CODE_DIR}/ge/single_op/task/op_task.cc"
    "${GE_CODE_DIR}/ge/single_op/task/tbe_task_builder.cc"
    "${GE_CODE_DIR}/ge/single_op/single_op.cc"
    "${GE_CODE_DIR}/ge/single_op/single_op_model.cc"
    "${GE_CODE_DIR}/ge/single_op/stream_resource.cc"
    "${GE_CODE_DIR}/ge/single_op/single_op_manager.cc"
    "${GE_CODE_DIR}/ge/single_op/task/aicpu_task_builder.cc"
    "${GE_CODE_DIR}/ge/single_op/task/aicpu_kernel_task_builder.cc"
    "${GE_CODE_DIR}/ge/single_op/task/rts_kernel_task_builder.cc"
    "${GE_CODE_DIR}/ge/hybrid/common/tensor_value.cc"
    "${GE_CODE_DIR}/ge/hybrid/common/npu_memory_allocator.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/rt_callback_manager.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/node_state.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/node_done_manager.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/hybrid_profiler.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/hybrid_model_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/hybrid_model_async_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/hybrid_execution_context.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/hybrid_model_pipeline_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/subgraph_context.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/subgraph_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/worker/task_compile_engine.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/worker/shape_inference_engine.cc"
    "${GE_CODE_DIR}/ge/hybrid/executor/worker/execution_engine.cc"
    "${GE_CODE_DIR}/ge/hybrid/model/hybrid_model.cc"
    "${GE_CODE_DIR}/ge/hybrid/model/hybrid_model_builder.cc"
    "${GE_CODE_DIR}/ge/hybrid/model/node_item.cc"
    "${GE_CODE_DIR}/ge/hybrid/model/graph_item.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/aicore/aicore_node_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/aicore/aicore_op_task.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/aicore/aicore_task_builder.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/aicore/aicore_task_compiler.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/aicpu/aicpu_ext_info.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/aicpu/aicpu_node_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/compiledsubgraph/known_node_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/ge_local/ge_local_node_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/host_cpu/host_cpu_node_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/controlop/control_op_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/partitioned_call/partitioned_call_node_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/hccl/hccl_node_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/rts/rts_node_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/rts/rts_node_task.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/rts/rts_task_factory.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/node_executor.cc"
    "${GE_CODE_DIR}/ge/hybrid/node_executor/task_context.cc"
    "${GE_CODE_DIR}/ge/hybrid/hybrid_davinci_model.cc"
)

# test files
set(COMMON_TEST_FILES
    "graph/passes/graph_builder_utils.cc"
    "graph/utils/buffer_pool_graph_builder.cc"
    "test.cc"
)

set(DISTINCT_GRAPH_LOAD_TEST_FILES
    "graph/load/data_dumper_unittest.cc"
    #"graph/load/new_model_manager_data_inputer_unittest.cc"
    #"graph/load/new_model_manager_davinci_model_unittest.cc"
    "graph/load/model_manager_unittest.cc"
    #"graph/load/new_model_manager_task_build_unittest.cc"
    "graph/load/new_model_manager_model_manager_aicpu_unittest.cc"
    "graph/load/end_graph_task_unittest.cc"
    "graph/load/new_model_manager_event_manager_unittest.cc"
    #"graph/load/output_net_output_unittest.cc"
    "graph/load/davinci_model_unittest.cc"
    "graph/load/tbe_handle_store_unittest.cc"
    "graph/load/hccl_task_info_unittest.cc"
    "graph/load/kernel_ex_task_info_unittest.cc"
    "graph/load/kernel_task_info_unittest.cc"
    "graph/load/ffts_task_info_unittest.cc"
    "graph/load/memcpy_addr_async_task_info_unittest.cc"
    "graph/load/memcpy_async_task_info_unittest.cc"
    "graph/load/cpu_queue_schedule_unittest.cc"
    #"graph/graph_load_unittest.cc"
    "graph/ge_executor_unittest.cc"
    "graph/load/model_helper_unittest.cc"
    "graph/load/model_utils_unittest.cc"
)

set(PASS_TEST_FILES
    "graph/passes/prune_pass_unittest.cc"
    "graph/passes/enter_pass_unittest.cc"
    "graph/passes/switch_op_pass_unittest.cc"
    "graph/passes/get_original_format_pass_unittest.cc"
    "graph/passes/pass_manager_unittest.cc"
    "graph/passes/permute_pass_unittest.cc"
    "graph/passes/print_op_pass_unittest.cc"
    "graph/passes/shape_operate_op_remove_pass_unittest.cc"
    "graph/passes/variable_op_pass_unittest.cc"
    "graph/passes/base_pass_unittest.cc"
    "graph/passes/addn_pass_unittest.cc"
    "graph/passes/save_pass_unittest.cc"
    "graph/passes/merge_pass_unittest.cc"
    "graph/passes/switch_logic_remove_pass_unittest.cc"
    "graph/passes/cond_branch_v1_unittest.cc"
    "graph/passes/loop_branch_v1_unittest.cc"
    "graph/passes/switch_dead_branch_elimination_unittest.cc"
    "graph/passes/subgraph_pass_unittest.cc"
    "graph/passes/assert_pass_unittest.cc"
    "graph/passes/dropout_pass_unittest.cc"
    "graph/passes/unused_const_pass_unittest.cc"
    "graph/passes/reshape_remove_pass_unittest.cc"
    "graph/passes/resource_pair_control_pass_unittest.cc"
    "graph/passes/trans_op_breadth_fusion_pass_unittest.cc"
    "graph/passes/trans_op_depth_fusion_pass_unittest.cc"
    "graph/passes/transop_nearby_allreduce_fusion_pass_unittest.cc"
    "graph/passes/constant_folding_pass_unittest.cc"
    "graph/passes/fuse_data_nodes_with_common_input_pass_unittest.cc"
    "graph/passes/stop_gradient_pass_unittest.cc"
    "graph/passes/prevent_gradient_pass_unittest.cc"
    "graph/passes/identity_pass_unittest.cc"
    "graph/passes/global_step_insert_pass_unittest.cc"
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
    "graph/passes/no_use_reshape_remove_pass_unittest.cc"
    "graph/passes/infershape_pass_unittest.cc"
    "graph/passes/mark_force_unknown_for_cond_pass_unittest.cc"
    "graph/passes/multi_batch_clone_pass_unittest.cc"
    "graph/passes/subgraph_const_migration_pass_unittest.cc"
    "graph/passes/replace_with_empty_const_pass_unittest.cc"
    "graph/passes/link_gen_mask_nodes_pass_unittest.cc"
    "graph/passes/transpose_transdata_pass_unittest.cc"
    "graph/passes/parallel_group_pass_unittest.cc"
    "graph/passes/buffer_pool_memory_pass_unittest.cc"
    "graph/passes/mark_node_unknown_shape_pass_unittest.cc"
    "graph/passes/reshape_recovery_pass_unittest.cc"
    "graph/passes/cast_remove_pass_unittest.cc"
    "graph/passes/memcpy_addr_async_unittest.cc"
    "graph/passes/hccl_continuous_pass_unittest.cc"
    "graph/passes/hccl_memcpy_pass_unittest.cc"
		
)

set(KERNEL_TEST_FILES
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
    "graph/passes/atomic_addr_clean_pass_unittest.cc"
)

set(MULTI_PARTS_TEST_FILES
    "graph_ir/ge_operator_factory_unittest.cc"
    "graph_ir/ge_ir_build_unittest.cc"
    "graph/transop_util_unittest.cc"
    "common/datatype_transfer_unittest.cc"
    "common/dump_manager_unittest.cc"
    "common/dump_op_unittest.cc"
    "common/dump_exception_unittest.cc"
    "common/opdebug_register_unittest.cc"
    "common/ge_opt_info_unittest.cc"
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
    "common/ge_auth_file_saver_unittest.cc"
    "graph/variable_accelerate_ctrl_unittest.cc"
    "graph/build/logical_stream_allocator_unittest.cc"
    "graph/build/stream_allocator_unittest.cc"
    "graph/build/model_builder_unittest.cc"
    "graph/build/mem_assigner_unittest.cc"
    "graph/build/task_generator_unittest.cc"
    "graph/build/buffer_pool_mem_assigner_unittest.cc"
    "graph/execute/graph_execute_unittest.cc"
    "graph/preprocess/graph_preprocess_unittest.cc"
    "graph/manager/hcom_util_unittest.cc"
    "graph/manager/graph_caching_allocator_unittest.cc"
    "graph/manager/host_mem_allocator_unittest.cc"
	"graph/manager/memory_api_unittest.cc"
    "graph/manager/session_scope_mem_allocator_unittest.cc"
    "graph/manager/run_graph_unittest.cc"
    "graph/partition/dynamic_shape_partition_unittest.cc"
    "graph/manager/graph_manager_unittest.cc"
    "graph/optimize/mem_rw_conflict_optimize_unittest.cc"
    "graph/optimize/graph_optimize_unittest.cc"
    "session/omg_omg_unittest.cc"
    "session/ge_api_unittest.cc"
    "session/inner_session_unittest.cc"
    "session/session_manager_unittest.cc"
    "common/host_cpu_engine_unittest.cc"
)

set(GENERATOR_TEST_FILES
    "generator/ge_generator_unittest.cc"
)

set(EXECUTOR_TEST_FILES
    "executor/ge_executor_unittest.cc"
)

set(SINGLE_OP_TEST_FILES
    "single_op/single_op_model_unittest.cc"
    "single_op/single_op_manager_unittest.cc"
    "single_op/stream_resource_unittest.cc"
    "single_op/single_op_task_unittest.cc"
    "single_op/single_op_unittest.cc"
)

set(PROFILING_MNG_TEST_FILES
    "profiling/ge_profiling_manager_unittest.cc"
)

set(HYBRID_TEST_FILES
    "hybrid/ge_hybrid_unittest.cc"
    "hybrid/known_node_executor_unittest.cc"
    "hybrid/executor/node_state_unittest.cc"
    "hybrid/executor/subgraph_executor_unittest.cc"
    "hybrid/executor/worker/execution_engine_unittest.cc"
    "hybrid/model/hybrid_model_builder_unittest.cc"
    "hybrid/node_executor/rts/rts_node_task_unittest.cc"
    "hybrid/node_executor/host_cpu/host_cpu_node_task_unittest.cc"
    "hybrid/node_executor/ge_local/ge_local_node_executor_unittest.cc"
    "hybrid/node_executor/hccl/hccl_node_executor_unittest.cc"
    "hybrid/executor/hybrid_model_async_executor_unittest.cc"
    "hybrid/executor/hybrid_model_pipeline_executor_unittest.cc"
    "hybrid/node_executor/aicore/aicore_task_compiler_unittest.cc"
)

set(OTHERS_TEST_FILES
    "plugin_manager/ge_util_unittest.cc"
)

list(APPEND COMMON_SHARED_LIBRARIES
    c_sec
    slog_stub
    cce_ge_stub
    runtime_stub
    profiler_stub
    mmpa_stub
    hccl_stub
    error_manager_stub
    opt_feature_stub
    ascend_protobuf
    json
)

# build graph
add_library(ge_ut_graph STATIC
    ${GRAPH_SRC_FILES} ${PARSER_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS}
)

target_compile_definitions(ge_ut_graph PRIVATE
    google=ascend_private
)

target_compile_options(ge_ut_graph PRIVATE
    -g
)

target_link_libraries(ge_ut_graph PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    ascend_protobuf
    json
)

# build common
add_library(ge_ut_common STATIC ${COMMON_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(ge_ut_common PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_ut_common PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ge_ut_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    ascend_protobuf
    json
    ge_ut_graph
    runtime_stub
)

# build common format
add_library(ge_ut_common_format STATIC ${COMMON_FORMAT_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(ge_ut_common_format PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_ut_common_format PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ge_ut_common_format PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    ascend_protobuf
    json
)

# build graph prepare common
add_library(ge_prepare_common STATIC ${GRAPH_PREPARE_COMMON_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(ge_prepare_common PRIVATE
    google=ascend_private
)

target_compile_options(ge_prepare_common PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ge_prepare_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    ascend_protobuf
    json
)

# build graph optimize common
add_library(ge_optimize_common STATIC ${GRAPH_OPTIMIZE_COMMON_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(ge_optimize_common PRIVATE
    google=ascend_private
)

target_compile_options(ge_optimize_common PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ge_optimize_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
    c_sec
    json
)

# build graph partition common
add_library(ge_partition_common STATIC ${GRAPH_PARTITION_COMMON_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(ge_partition_common PRIVATE
    google=ascend_private
)

target_compile_options(ge_partition_common PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ge_partition_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
    c_sec
    json
)

# build build graph load common
add_library(ge_load_common STATIC ${GRAPH_LOAD_COMMON_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(ge_load_common PRIVATE
    google=ascend_private
)

target_compile_options(ge_load_common PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ge_load_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    ascend_protobuf
    json
)

# build graph execute common
add_library(ge_execute_common STATIC ${GRAPH_EXECUTE_COMMON_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(ge_execute_common PRIVATE
    google=ascend_private
)

target_compile_options(ge_execute_common PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ge_execute_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    json
    ascend_protobuf
)

# build graph build common
add_library(ge_build_common STATIC ${GRAPH_BUILD_COMMON_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(ge_build_common PRIVATE
    google=ascend_private
)

target_compile_options(ge_build_common PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ge_build_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    ascend_protobuf
    json
)

# build graph pass common
add_library(ge_pass_common STATIC ${GRAPH_PASS_COMMON_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(ge_pass_common PRIVATE
    google=ascend_private
)

target_compile_options(ge_pass_common PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ge_pass_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
    c_sec
    json
)

# build single_op common
add_library(ge_single_op STATIC ${SINGLE_OP_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(ge_single_op PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_single_op PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ge_single_op PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
    json
    c_sec
)

# ut binary

# libge_mutiparts_utest
add_executable(ut_libge_multiparts_utest
    ${COMMON_TEST_FILES}
    ${MULTI_PARTS_TEST_FILES}
)

target_compile_options(ut_libge_multiparts_utest PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_compile_definitions(ut_libge_multiparts_utest PRIVATE
    google=ascend_private
)

target_link_libraries(ut_libge_multiparts_utest
    $<BUILD_INTERFACE:intf_pub>
    ge_build_common ge_load_common ge_execute_common ge_optimize_common ge_partition_common ge_prepare_common
    ge_single_op ge_ut_common_format ge_ut_common
    gtest gtest_main gmock gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl -lgcov
)

# libge_others_utest
add_executable(ut_libge_others_utest
    ${COMMON_TEST_FILES}
    ${PASS_TEST_FILES}
    ${EXECUTE_TEST_FILES}
    ${OTHERS_TEST_FILES}
)

target_compile_options(ut_libge_others_utest PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ut_libge_others_utest
    $<BUILD_INTERFACE:intf_pub>
    ge_load_common ge_execute_common ge_ut_common ge_ut_common_format
    gtest gtest_main gmock gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl -lgcov
)

# libge_kernel_utest
add_executable(ut_libge_kernel_utest
    ${COMMON_TEST_FILES}
    ${KERNEL_TEST_FILES}
    ${KERNEL_SRC_FILES}
)

target_compile_options(ut_libge_kernel_utest PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ut_libge_kernel_utest
    $<BUILD_INTERFACE:intf_pub>
    ge_load_common ge_ut_common ge_ut_common_format
    gtest gtest_main gmock gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl -lgcov
)

# libge_distinct_load_utest
add_executable(ut_libge_distinct_load_utest
        ${COMMON_TEST_FILES}
        ${GENERATOR_TEST_FILES}
        ${EXECUTOR_TEST_FILES}
        ${DISTINCT_GRAPH_LOAD_TEST_FILES}
        ${DISTINCT_GRAPH_LOAD_SRC_FILES}
        ${SINGLE_OP_TEST_FILES}
        ${PROFILING_MNG_TEST_FILES}
        ${HYBRID_TEST_FILES}
)

target_compile_options(ut_libge_distinct_load_utest PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_compile_definitions(ut_libge_distinct_load_utest PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_link_libraries(ut_libge_distinct_load_utest
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--whole-archive
    ge_single_op
    -Wl,--no-whole-archive
    ge_execute_common ge_load_common
    ge_prepare_common ge_optimize_common ge_build_common ge_partition_common ge_ut_common ge_ut_common_format
    gtest gtest_main gmock gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl -lpthread -lgcov
)
