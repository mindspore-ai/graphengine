if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
    add_subdirectory(common)
    add_subdirectory(plugin/engine)
    add_subdirectory(ge_local_engine)
    add_subdirectory(host_cpu_engine)
    add_subdirectory(executor)
    add_subdirectory(offline)
elseif (ENABLE_D)
    add_subdirectory(common)
    add_subdirectory(ge_runtime)
endif ()

set(PROTO_LIST
    "${METADEF_DIR}/proto/fusion_model.proto"
    "${GE_CODE_DIR}/ge/proto/optimizer_priority.proto"
)

set(PROTO_CLIENT_LIST
    "${METADEF_DIR}/proto/ge_api.proto"
)

set(PROTO_HEADER_LIST
    "${METADEF_DIR}/proto/om.proto"
    "${METADEF_DIR}/proto/task.proto"
    "${METADEF_DIR}/proto/insert_op.proto"
    "${METADEF_DIR}/proto/ge_ir.proto"
    "${METADEF_DIR}/proto/fwk_adapter.proto"
    "${METADEF_DIR}/proto/op_mapping.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})
protobuf_generate(ge PROTO_CLIENT_SRCS PROTO_CLIENT_HDRS ${PROTO_CLIENT_LIST})
protobuf_generate(ge PROTO_HEADER_SRCS PROTO_HEADER_HDRS ${PROTO_HEADER_LIST})
protobuf_generate(ge_client PROTO_CLIENT_HEADER_SRCS PROTO_CLIENT_HEADER_HDRS ${PROTO_HEADER_LIST})

if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
############ libge_proto_common.a ############
add_library(ge_proto_common STATIC
    ${PROTO_HEADER_HDRS}
    ${PROTO_SRCS}
)

target_compile_definitions(ge_proto_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_compile_options(ge_proto_common PRIVATE
    -O2
    -fno-common
)

target_link_libraries(ge_proto_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
)

############ libge_proto_client.a ############
add_library(ge_proto_client STATIC
    ${PROTO_CLIENT_HEADER_HDRS}
    ${PROTO_CLIENT_SRCS}
)

target_compile_definitions(ge_proto_client PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_include_directories(ge_proto_client PRIVATE
    ${CMAKE_BINARY_DIR}/proto/ge_client
    ${CMAKE_BINARY_DIR}/proto/ge_client/proto
)

target_compile_options(ge_proto_client PRIVATE
    -O2
    -fno-common
)

target_link_libraries(ge_proto_client PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
)
endif ()

##################################################################
set(TRAIN_SRC_LIST
    "common/formats/format_transfers/datatype_transfer.cc"
    "common/formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc"
    "common/formats/format_transfers/format_transfer_dhwcn_fracz3D.cc"
    "common/formats/format_transfers/format_transfer_dhwnc_fracz3D_transpose.cc"
    "common/formats/format_transfers/format_transfer_fractal_nz.cc"
    "common/formats/format_transfers/format_transfer_fractal_z.cc"
    "common/formats/format_transfers/format_transfer_fractal_zz.cc"
    "common/formats/format_transfers/format_transfer_fracz_hwcn.cc"
    "common/formats/format_transfers/format_transfer_fracz_nchw.cc"
    "common/formats/format_transfers/format_transfer_fracz_nhwc.cc"
    "common/formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc"
    "common/formats/format_transfers/format_transfer_nc1hwc0_nchw.cc"
    "common/formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc"
    "common/formats/format_transfers/format_transfer_nchw_nc1hwc0.cc"
    "common/formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc"
    "common/formats/format_transfers/format_transfer_transpose.cc"
    "common/formats/formats.cc"
    "common/formats/utils/formats_trans_utils.cc"
    "common/fp16_t.cc"
    "common/ge/plugin_manager.cc"
    "common/ge/op_tiling_manager.cc"
    "common/helper/model_cache_helper.cc"
    "common/profiling/profiling_manager.cc"
    "common/dump/dump_manager.cc"
    "common/dump/exception_dumper.cc"
    "common/dump/dump_properties.cc"
    "common/dump/opdebug_register.cc"
    "common/dump/dump_op.cc"
    "common/profiling/ge_profiling.cc"
    "common/profiling/ge_runner_profiling.cc"
    "engine_manager/dnnengine_manager.cc"
    "ge_local_engine/engine/host_cpu_engine.cc"
    "generator/ge_generator.cc"
    "generator/generator_api.cc"
    "graph/build/graph_builder.cc"
    "graph/build/label_allocator.cc"
    "graph/build/logical_stream_allocator.cc"
    "graph/build/model_builder.cc"
    "graph/build/run_context.cc"
    "graph/build/stream_allocator.cc"
    "graph/build/stream_graph_optimizer.cc"
    "graph/build/task_generator.cc"
    "graph/common/bcast.cc"
    "graph/common/local_context.cc"
    "graph/common/omg_util.cc"
    "graph/common/transop_util.cc"
    "graph/execute/graph_execute.cc"
    "graph/label/case_label_maker.cc"
    "graph/label/if_label_maker.cc"
    "graph/label/label_maker.cc"
    "graph/label/partitioned_call_label_maker.cc"
    "graph/label/while_label_maker.cc"
    "graph/load/graph_loader.cc"
    "graph/load/model_manager/cpu_queue_schedule.cc"
    "graph/load/model_manager/data_dumper.cc"
    "graph/load/model_manager/data_inputer.cc"
    "graph/load/model_manager/davinci_model.cc"
    "graph/load/model_manager/model_manager.cc"
    "graph/load/model_manager/model_utils.cc"
    "graph/load/model_manager/aipp_utils.cc"
    "graph/load/model_manager/task_info/end_graph_task_info.cc"
    "graph/load/model_manager/task_info/model_exit_task_info.cc"
    "graph/load/model_manager/task_info/event_record_task_info.cc"
    "graph/load/model_manager/task_info/event_wait_task_info.cc"
    "graph/load/model_manager/task_info/fusion_start_task_info.cc"
    "graph/load/model_manager/task_info/fusion_stop_task_info.cc"
    "graph/load/model_manager/task_info/hccl_task_info.cc"
    "graph/load/model_manager/task_info/kernel_ex_task_info.cc"
    "graph/load/model_manager/task_info/kernel_task_info.cc"
    "graph/load/model_manager/task_info/label_set_task_info.cc"
    "graph/load/model_manager/task_info/label_switch_by_index_task_info.cc"
    "graph/load/model_manager/task_info/label_goto_ex_task_info.cc"
    "graph/load/model_manager/task_info/memcpy_addr_async_task_info.cc"
    "graph/load/model_manager/task_info/memcpy_async_task_info.cc"
    "graph/load/model_manager/task_info/profiler_trace_task_info.cc"
    "graph/load/model_manager/task_info/stream_active_task_info.cc"
    "graph/load/model_manager/task_info/stream_switch_task_info.cc"
    "graph/load/model_manager/task_info/stream_switchn_task_info.cc"
    "graph/load/model_manager/task_info/super_kernel/super_kernel.cc"
    "graph/load/model_manager/task_info/super_kernel/super_kernel_factory.cc"
    "graph/load/model_manager/task_info/task_info.cc"
    "graph/load/model_manager/tbe_handle_store.cc"
    "graph/load/model_manager/zero_copy_task.cc"
    "graph/load/model_manager/zero_copy_offset.cc"
    "graph/manager/graph_context.cc"
    "graph/manager/graph_manager.cc"
    "graph/manager/graph_manager_utils.cc"
    "graph/manager/graph_mem_allocator.cc"
    "graph/manager/graph_caching_allocator.cc"
    "graph/manager/graph_var_manager.cc"
    "graph/manager/host_mem_manager.cc"
    "graph/manager/rdma_pool_allocator.cc"
    "graph/manager/host_mem_allocator.cc"
    "graph/manager/memory_api.cc"
    "graph/manager/model_manager/event_manager.cc"
    "graph/manager/trans_var_data_utils.cc"
    "graph/manager/util/debug.cc"
    "graph/manager/util/hcom_util.cc"
    "graph/manager/util/rt_context_util.cc"
    "graph/manager/util/variable_accelerate_ctrl.cc"
    "graph/optimize/graph_optimize.cc"
    "graph/optimize/mem_rw_conflict_optimize.cc"
    "graph/optimize/summary_optimize.cc"
    "graph/partition/engine_place.cc"
    "graph/partition/graph_partition.cc"
    "graph/passes/addn_pass.cc"
    "graph/passes/aicpu_constant_folding_pass.cc"
    "graph/passes/assert_pass.cc"
    "graph/passes/input_output_connection_identify_pass.cc"
    "graph/passes/atomic_addr_clean_pass.cc"
    "graph/passes/mark_same_addr_pass.cc"
    "graph/passes/mark_graph_unknown_status_pass.cc"
    "graph/passes/mark_node_unknown_shape_pass.cc"
    "graph/passes/mark_agnostic_pass.cc"
    "graph/partition/dynamic_shape_partition.cc"
    "graph/partition/stage_partition.cc"
    "graph/passes/base_pass.cc"
    "graph/passes/bitcast_pass.cc"
    "graph/passes/cast_remove_pass.cc"
    "graph/passes/cast_translate_pass.cc"
    "graph/passes/common_subexpression_elimination_pass.cc"
    "graph/passes/transop_symmetry_elimination_pass.cc"
    "graph/passes/compile_nodes_pass.cc"
    "graph/passes/constant_folding_pass.cc"
    "graph/passes/constant_fuse_same_pass.cc"
    "graph/passes/fuse_data_nodes_with_common_input_pass.cc"
    "graph/passes/remove_same_const_pass.cc"
    "graph/passes/useless_control_out_remove_pass.cc"
    "graph/passes/control_trigger_pass.cc"
    "graph/passes/dimension_adjust_pass.cc"
    "graph/passes/dimension_compute_pass.cc"
    "graph/passes/dropout_pass.cc"
    "graph/passes/hccl_group_pass.cc"
    "graph/passes/hccl_tailing_optimization_pass.cc"
    "graph/passes/enter_pass.cc"
    "graph/passes/assign_remove_pass.cc"
    "graph/passes/inplace_support_check_pass.cc"
    "graph/passes/flow_ctrl_pass.cc"
    "graph/passes/global_step_insert_pass.cc"
    "host_kernels/transpose_kernel.cc"
    "host_kernels/add_kernel.cc"
    "host_kernels/broadcast_args_kernel.cc"
    "host_kernels/broadcast_gradient_args_kernel.cc"
    "host_kernels/cast_kernel.cc"
    "host_kernels/concat_offset_kernel.cc"
    "host_kernels/concat_v2_kernel.cc"
    "host_kernels/dynamic_stitch_kernel.cc"
    "host_kernels/identity_kernel.cc"
    "host_kernels/empty_kernel.cc"
    "host_kernels/expanddims_kernel.cc"
    "host_kernels/fill_kernel.cc"
    "host_kernels/floordiv_kernel.cc"
    "host_kernels/floormod_kernel.cc"
    "host_kernels/gather_v2_kernel.cc"
    "host_kernels/greater_kernel.cc"
    "host_kernels/kernel_utils.cc"
    "host_kernels/maximum_kernel.cc"
    "host_kernels/mul_kernel.cc"
    "host_kernels/pack_kernel.cc"
    "host_kernels/permute_kernel.cc"
    "host_kernels/range_kernel.cc"
    "host_kernels/rank_kernel.cc"
    "host_kernels/reduce_prod_kernel.cc"
    "host_kernels/reshape_kernel.cc"
    "host_kernels/rsqrt_kernel.cc"
    "host_kernels/shape_kernel.cc"
    "host_kernels/shape_n_kernel.cc"
    "host_kernels/size_kernel.cc"
    "host_kernels/slice_d_kernel.cc"
    "host_kernels/slice_kernel.cc"
    "host_kernels/squeeze_kernel.cc"
    "host_kernels/unsqueeze_kernel.cc"
    "host_kernels/ssd_prior_box_kernel.cc"
    "host_kernels/strided_slice_kernel.cc"
    "host_kernels/sub_kernel.cc"
    "host_kernels/transdata_kernel.cc"
    "host_kernels/unpack_kernel.cc"
    "host_kernels/reformat_kernel.cc"
    "graph/passes/folding_pass.cc"
    "graph/passes/get_original_format_pass.cc"
    "graph/passes/guarantee_const_pass.cc"
    "graph/passes/hccl_memcpy_pass.cc"
    "graph/passes/hccl_continuous_memcpy_pass.cc"
    "graph/passes/identity_pass.cc"
    "graph/passes/ref_identity_delete_op_pass.cc"
    "graph/passes/infershape_pass.cc"
    "graph/passes/iterator_op_pass.cc"
    "graph/passes/link_gen_mask_nodes_pass.cc"
    "graph/passes/merge_pass.cc"
    "graph/passes/multi_batch_pass.cc"
    "graph/passes/multi_batch_clone_pass.cc"
    "graph/passes/subexpression_migration_pass.cc"
    "graph/passes/subgraph_const_migration_pass.cc"
    "graph/passes/unused_args_clean_pass.cc"
    "graph/passes/net_output_pass.cc"
    "graph/passes/next_iteration_pass.cc"
    "graph/passes/no_use_reshape_remove_pass.cc"
    "graph/passes/pass_manager.cc"
    "graph/passes/pass_utils.cc"
    "graph/passes/permute_pass.cc"
    "graph/passes/placeholder_with_default_pass.cc"
    "graph/passes/prevent_gradient_pass.cc"
    "graph/passes/print_op_pass.cc"
    "graph/passes/prune_pass.cc"
    "graph/passes/ctrl_edge_transfer_pass.cc"
    "graph/passes/replace_with_empty_const_pass.cc"
    "graph/passes/reshape_remove_pass.cc"
    "graph/passes/reshape_recovery_pass.cc"
    "graph/passes/resource_pair_add_control_pass.cc"
    "graph/passes/resource_pair_remove_control_pass.cc"
    "graph/passes/same_transdata_breadth_fusion_pass.cc"
    "graph/passes/save_pass.cc"
    "graph/passes/shape_operate_op_remove_pass.cc"
    "graph/passes/snapshot_pass.cc"
    "graph/passes/stop_gradient_pass.cc"
    "graph/passes/subgraph_pass.cc"
    "graph/passes/data_pass.cc"
    "graph/passes/switch_data_edges_bypass.cc"
    "graph/passes/switch_logic_remove_pass.cc"
    "graph/passes/merge_to_stream_merge_pass.cc"
    "graph/passes/merge_input_memcpy_pass.cc"
    "graph/passes/switch_to_stream_switch_pass.cc"
    "graph/passes/mark_force_unknown_for_cond_pass.cc"
    "graph/passes/attach_stream_label_pass.cc"
    "graph/passes/switch_dead_branch_elimination.cc"
    "graph/passes/replace_transshape_pass.cc"
    "graph/passes/transop_breadth_fusion_pass.cc"
    "graph/passes/transop_depth_fusion_pass.cc"
    "graph/passes/transop_nearby_allreduce_fusion_pass.cc"
    "graph/passes/transop_without_reshape_fusion_pass.cc"
    "graph/passes/transpose_transdata_pass.cc"
    "graph/passes/unused_const_pass.cc"
    "graph/passes/var_is_initialized_op_pass.cc"
    "graph/passes/parallel_concat_start_op_pass.cc"
    "graph/passes/cond_pass.cc"
    "graph/passes/cond_remove_pass.cc"
    "graph/passes/for_pass.cc"
    "graph/passes/variable_op_pass.cc"
    "graph/passes/variable_prepare_op_pass.cc"
    "graph/passes/variable_ref_delete_op_pass.cc"
    "graph/passes/variable_ref_useless_control_out_delete_pass.cc"
    "graph/passes/end_of_sequence_add_control_pass.cc"
    "graph/passes/memcpy_addr_async_pass.cc"
    "graph/passes/parallel_group_pass.cc"
    "graph/passes/set_input_output_offset_pass.cc"
    "graph/passes/buffer_pool_memory_pass.cc"
    "graph/preprocess/graph_preprocess.cc"
    "graph/preprocess/insert_op/ge_aipp_op.cc"
    "graph/preprocess/insert_op/util_insert_aipp_op.cc"
    "graph/preprocess/multi_batch_options.cc"
    "graph/preprocess/multi_batch_copy_graph.cc"
    "init/gelib.cc"
    "model/ge_model.cc"
    "model/ge_root_model.cc"
    "opskernel_manager/ops_kernel_manager.cc"
    "opskernel_manager/ops_kernel_builder_manager.cc"
    "session/inner_session.cc"
    "session/session_manager.cc"
    "single_op/single_op.cc"
    "single_op/single_op_manager.cc"
    "single_op/single_op_model.cc"
    "single_op/stream_resource.cc"
    "single_op/task/build_task_utils.cc"
    "single_op/task/op_task.cc"
    "single_op/task/tbe_task_builder.cc"
    "single_op/task/aicpu_task_builder.cc"
    "single_op/task/aicpu_kernel_task_builder.cc"
    "hybrid/common/tensor_value.cc"
    "hybrid/common/npu_memory_allocator.cc"
    "hybrid/executor/rt_callback_manager.cc"
    "hybrid/executor/node_state.cc"
    "hybrid/executor/node_done_manager.cc"
    "hybrid/executor/hybrid_profiler.cc"
    "hybrid/executor/hybrid_model_executor.cc"
    "hybrid/executor/hybrid_model_pipeline_executor.cc"
    "hybrid/executor/hybrid_model_async_executor.cc"
    "hybrid/executor/hybrid_execution_context.cc"
    "hybrid/executor/subgraph_context.cc"
    "hybrid/executor/subgraph_executor.cc"
    "hybrid/executor/worker/task_compile_engine.cc"
    "hybrid/executor/worker/shape_inference_engine.cc"
    "hybrid/executor/worker/execution_engine.cc"
    "hybrid/model/hybrid_model.cc"
    "hybrid/model/hybrid_model_builder.cc"
    "hybrid/model/node_item.cc"
    "hybrid/model/graph_item.cc"
    "hybrid/node_executor/aicore/aicore_node_executor.cc"
    "hybrid/node_executor/aicore/aicore_op_task.cc"
    "hybrid/node_executor/aicore/aicore_task_builder.cc"
    "hybrid/node_executor/aicore/aicore_task_compiler.cc"
    "hybrid/node_executor/aicpu/aicpu_ext_info.cc"
    "hybrid/node_executor/aicpu/aicpu_node_executor.cc"
    "hybrid/node_executor/compiledsubgraph/known_node_executor.cc"
    "hybrid/node_executor/ge_local/ge_local_node_executor.cc"
    "hybrid/node_executor/host_cpu/host_cpu_node_executor.cc"
    "hybrid/node_executor/host_cpu/kernel_factory.cc"
    "hybrid/node_executor/host_cpu/kernel/no_op_kernel.cc"
    "hybrid/node_executor/host_cpu/kernel/variable_kernel.cc"
    "hybrid/node_executor/host_cpu/kernel/assign_kernel.cc"
    "hybrid/node_executor/host_cpu/kernel/random_uniform_kernel.cc"
    "hybrid/node_executor/host_cpu/kernel/data_kernel.cc"
    "hybrid/node_executor/controlop/control_op_executor.cc"
    "hybrid/node_executor/partitioned_call/partitioned_call_node_executor.cc"
    "hybrid/node_executor/hccl/hccl_node_executor.cc"
    "hybrid/node_executor/rts/rts_node_executor.cc"
    "hybrid/node_executor/rts/rts_node_task.cc"
    "hybrid/node_executor/rts/rts_task_factory.cc"
    "hybrid/node_executor/node_executor.cc"
    "hybrid/node_executor/task_context.cc"
    "hybrid/hybrid_davinci_model.cc"
    "executor/ge_executor.cc"
    "client/ge_api.cc"
    "analyzer/analyzer.cc"
    "ir_build/ge_ir_build.cc"
    "ir_build/attr_options/utils.cc"
    "ir_build/attr_options/keep_dtype_option.cc"
    "ir_build/attr_options/weight_compress_option.cc"
    "ir_build/option_utils.cc"
    "graph/build/memory/memory_assigner.cc"
    "graph/build/memory/graph_mem_assigner.cc"
    "graph/build/memory/binary_block_mem_assigner.cc"
    "graph/build/memory/block_mem_assigner.cc"
    "graph/build/memory/hybrid_mem_assigner.cc"
    "graph/build/memory/max_block_mem_assigner.cc"
    "graph/build/memory/var_mem_assign_util.cc"
    "graph/build/memory/buffer_pool_mem_assigner.cc"
)

set(INFER_SRC_LIST
    "graph/manager/trans_var_data_utils.cc"
    "common/fp16_t.cc"
    "common/formats/utils/formats_trans_utils.cc"
    "common/formats/format_transfers/datatype_transfer.cc"
    "common/formats/format_transfers/format_transfer_transpose.cc"
    "common/formats/format_transfers/format_transfer_nchw_nc1hwc0.cc"
    "common/formats/format_transfers/format_transfer_fractal_z.cc"
    "common/formats/format_transfers/format_transfer_fractal_nz.cc"
    "common/formats/format_transfers/format_transfer_fractal_zz.cc"
    "common/formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc"
    "common/formats/format_transfers/format_transfer_nc1hwc0_nchw.cc"
    "common/formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc"
    "common/formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc"
    "common/formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc"
    "common/formats/format_transfers/format_transfer_fracz_nchw.cc"
    "common/formats/format_transfers/format_transfer_fracz_nhwc.cc"
    "common/formats/format_transfers/format_transfer_fracz_hwcn.cc"
    "common/formats/format_transfers/format_transfer_dhwcn_fracz3D.cc"
    "common/formats/format_transfers/format_transfer_dhwnc_fracz3D_transpose.cc"
    "common/formats/format_transfers/format_transfer_nchw_fz_c04.cc"
    "common/formats/formats.cc"
    "common/profiling/profiling_manager.cc"
    "common/dump/dump_properties.cc"
    "common/dump/exception_dumper.cc"
    "common/dump/dump_manager.cc"
    "common/dump/dump_op.cc"
    "common/dump/opdebug_register.cc"
    "common/dump/dump_server.cc"
    "common/helper/model_cache_helper.cc"
    "ge_local_engine/engine/host_cpu_engine.cc"
    "common/ge/plugin_manager.cc"
    "common/ge/op_tiling_manager.cc"
    "init/gelib.cc"
    "session/inner_session.cc"
    "session/session_manager.cc"
    "engine_manager/dnnengine_manager.cc"
    "opskernel_manager/ops_kernel_manager.cc"
    "opskernel_manager/ops_kernel_builder_manager.cc"
    "graph/manager/graph_manager.cc"
    "graph/manager/graph_manager_utils.cc"
    "graph/manager/graph_context.cc"
    "graph/preprocess/graph_preprocess.cc"
    "graph/preprocess/multi_batch_options.cc"
    "graph/preprocess/multi_batch_copy_graph.cc"
    "graph/execute/graph_execute.cc"
    "graph/load/graph_loader.cc"
    "graph/optimize/graph_optimize.cc"
    "graph/optimize/mem_rw_conflict_optimize.cc"
    "graph/optimize/summary_optimize.cc"
    "graph/build/graph_builder.cc"
    "graph/partition/engine_place.cc"
    "graph/partition/graph_partition.cc"
    "graph/partition/dynamic_shape_partition.cc"
    "graph/partition/stage_partition.cc"
    "generator/ge_generator.cc"
    "generator/generator_api.cc"
    "graph/manager/graph_var_manager.cc"
    "graph/manager/host_mem_manager.cc"
    "graph/manager/rdma_pool_allocator.cc"
    "graph/manager/host_mem_allocator.cc"
    "graph/manager/graph_mem_allocator.cc"
    "graph/manager/graph_caching_allocator.cc"
    "model/ge_model.cc"
    "model/ge_root_model.cc"
    "graph/common/transop_util.cc"
    "graph/passes/pass_manager.cc"
    "graph/passes/resource_pair_add_control_pass.cc"
    "graph/passes/resource_pair_remove_control_pass.cc"
    "graph/passes/pass_utils.cc"
    "graph/passes/base_pass.cc"
    "graph/passes/bitcast_pass.cc"
    "graph/passes/constant_folding_pass.cc"
    "graph/passes/aicpu_constant_folding_pass.cc"
    "graph/passes/reshape_remove_pass.cc"
    "graph/passes/reshape_recovery_pass.cc"
    "graph/passes/transop_breadth_fusion_pass.cc"
    "graph/passes/transop_depth_fusion_pass.cc"
    "graph/passes/transop_nearby_allreduce_fusion_pass.cc"
    "graph/passes/same_transdata_breadth_fusion_pass.cc"
    "graph/passes/transop_without_reshape_fusion_pass.cc"
    "graph/passes/compile_nodes_pass.cc"
    "graph/passes/variable_prepare_op_pass.cc"
    "graph/passes/variable_ref_delete_op_pass.cc"
    "graph/passes/variable_ref_useless_control_out_delete_pass.cc"
    "graph/passes/subgraph_pass.cc"
    "graph/passes/data_pass.cc"
    "graph/passes/net_output_pass.cc"
    "graph/passes/replace_transshape_pass.cc"
    "graph/passes/constant_fuse_same_pass.cc"
    "graph/passes/fuse_data_nodes_with_common_input_pass.cc"
    "graph/passes/print_op_pass.cc"
    "graph/passes/no_use_reshape_remove_pass.cc"
    "graph/passes/iterator_op_pass.cc"
    "graph/passes/input_output_connection_identify_pass.cc"
    "graph/passes/atomic_addr_clean_pass.cc"
    "graph/passes/mark_same_addr_pass.cc"
    "graph/passes/mark_graph_unknown_status_pass.cc"
    "graph/passes/mark_node_unknown_shape_pass.cc"
    "graph/passes/mark_agnostic_pass.cc"
    "graph/common/omg_util.cc"
    "graph/common/bcast.cc"
    "graph/common/local_context.cc"
    "graph/passes/dimension_compute_pass.cc"
    "graph/passes/dimension_adjust_pass.cc"
    "graph/passes/get_original_format_pass.cc"
    "graph/passes/shape_operate_op_remove_pass.cc"
    "graph/passes/assert_pass.cc"
    "graph/passes/dropout_pass.cc"
    "graph/passes/infershape_pass.cc"
    "graph/passes/unused_const_pass.cc"
    "graph/passes/permute_pass.cc"
    "graph/passes/ctrl_edge_transfer_pass.cc"
    "graph/passes/end_of_sequence_add_control_pass.cc"
    "host_kernels/broadcast_gradient_args_kernel.cc"
    "host_kernels/greater_kernel.cc"
    "host_kernels/gather_v2_kernel.cc"
    "host_kernels/maximum_kernel.cc"
    "host_kernels/floormod_kernel.cc"
    "host_kernels/floordiv_kernel.cc"
    "host_kernels/range_kernel.cc"
    "host_kernels/shape_kernel.cc"
    "host_kernels/size_kernel.cc"
    "host_kernels/shape_n_kernel.cc"
    "host_kernels/rank_kernel.cc"
    "host_kernels/broadcast_args_kernel.cc"
    "host_kernels/fill_kernel.cc"
    "host_kernels/empty_kernel.cc"
    "host_kernels/expanddims_kernel.cc"
    "host_kernels/reshape_kernel.cc"
    "host_kernels/squeeze_kernel.cc"
    "host_kernels/unsqueeze_kernel.cc"
    "host_kernels/kernel_utils.cc"
    "host_kernels/cast_kernel.cc"
    "host_kernels/transdata_kernel.cc"
    "host_kernels/unpack_kernel.cc"
    "host_kernels/transpose_kernel.cc"
    "host_kernels/permute_kernel.cc"
    "host_kernels/pack_kernel.cc"
    "host_kernels/concat_v2_kernel.cc"
    "host_kernels/concat_offset_kernel.cc"
    "host_kernels/strided_slice_kernel.cc"
    "host_kernels/ssd_prior_box_kernel.cc"
    "host_kernels/add_kernel.cc"
    "host_kernels/sub_kernel.cc"
    "host_kernels/mul_kernel.cc"
    "host_kernels/reduce_prod_kernel.cc"
    "host_kernels/rsqrt_kernel.cc"
    "host_kernels/slice_kernel.cc"
    "host_kernels/slice_d_kernel.cc"
    "host_kernels/dynamic_stitch_kernel.cc"
    "host_kernels/identity_kernel.cc"
    "host_kernels/reformat_kernel.cc"
    "graph/passes/stop_gradient_pass.cc"
    "graph/passes/prevent_gradient_pass.cc"
    "graph/passes/identity_pass.cc"
    "graph/passes/ref_identity_delete_op_pass.cc"
    "graph/passes/placeholder_with_default_pass.cc"
    "graph/passes/snapshot_pass.cc"
    "graph/passes/guarantee_const_pass.cc"
    "graph/passes/var_is_initialized_op_pass.cc"
    "graph/passes/parallel_concat_start_op_pass.cc"
    "graph/passes/folding_pass.cc"
    "graph/passes/cast_translate_pass.cc"
    "graph/passes/prune_pass.cc"
    "graph/passes/merge_to_stream_merge_pass.cc"
    "graph/passes/merge_input_memcpy_pass.cc"
    "graph/passes/switch_to_stream_switch_pass.cc"
    "graph/passes/mark_force_unknown_for_cond_pass.cc"
    "graph/passes/attach_stream_label_pass.cc"
    "graph/passes/multi_batch_pass.cc"
    "graph/passes/multi_batch_clone_pass.cc"
    "graph/passes/subexpression_migration_pass.cc"
    "graph/passes/subgraph_const_migration_pass.cc"
    "graph/passes/unused_args_clean_pass.cc"
    "graph/passes/next_iteration_pass.cc"
    "graph/passes/control_trigger_pass.cc"
    "graph/passes/cond_pass.cc"
    "graph/passes/cond_remove_pass.cc"
    "graph/passes/for_pass.cc"
    "graph/passes/enter_pass.cc"
    "graph/passes/assign_remove_pass.cc"
    "graph/passes/inplace_support_check_pass.cc"
    "graph/passes/addn_pass.cc"
    "graph/passes/common_subexpression_elimination_pass.cc"
    "graph/passes/remove_same_const_pass.cc"
    "graph/passes/useless_control_out_remove_pass.cc"
    "graph/passes/transop_symmetry_elimination_pass.cc"
    "graph/passes/save_pass.cc"
    "graph/passes/switch_dead_branch_elimination.cc"
    "graph/passes/switch_logic_remove_pass.cc"
    "graph/passes/switch_data_edges_bypass.cc"
    "graph/passes/merge_pass.cc"
    "graph/passes/variable_op_pass.cc"
    "graph/passes/cast_remove_pass.cc"
    "graph/passes/transpose_transdata_pass.cc"
    "graph/passes/hccl_memcpy_pass.cc"
    "graph/passes/hccl_continuous_memcpy_pass.cc"
    "graph/passes/flow_ctrl_pass.cc"
    "graph/passes/global_step_insert_pass.cc"
    "graph/passes/link_gen_mask_nodes_pass.cc"
    "graph/passes/replace_with_empty_const_pass.cc"
    "graph/passes/hccl_group_pass.cc"
    "graph/passes/hccl_tailing_optimization_pass.cc"
    "graph/passes/memcpy_addr_async_pass.cc"
    "graph/passes/set_input_output_offset_pass.cc"
    "graph/passes/parallel_group_pass.cc"
    "graph/passes/buffer_pool_memory_pass.cc"
    "graph/manager/model_manager/event_manager.cc"
    "graph/manager/util/rt_context_util.cc"
    "graph/manager/util/variable_accelerate_ctrl.cc"
    "graph/manager/util/debug.cc"
    "graph/load/model_manager/model_manager.cc"
    "graph/load/model_manager/data_inputer.cc"
    "graph/load/model_manager/davinci_model.cc"
    "graph/load/model_manager/model_utils.cc"
    "graph/load/model_manager/aipp_utils.cc"
    "graph/load/model_manager/tbe_handle_store.cc"
    "graph/load/model_manager/cpu_queue_schedule.cc"
    "graph/load/model_manager/zero_copy_task.cc"
    "graph/load/model_manager/zero_copy_offset.cc"
    "graph/load/model_manager/data_dumper.cc"
    "graph/load/model_manager/task_info/task_info.cc"
    "graph/load/model_manager/task_info/event_record_task_info.cc"
    "graph/load/model_manager/task_info/event_wait_task_info.cc"
    "graph/load/model_manager/task_info/fusion_start_task_info.cc"
    "graph/load/model_manager/task_info/fusion_stop_task_info.cc"
    "graph/load/model_manager/task_info/kernel_ex_task_info.cc"
    "graph/load/model_manager/task_info/kernel_task_info.cc"
    "graph/load/model_manager/task_info/label_set_task_info.cc"
    "graph/load/model_manager/task_info/label_switch_by_index_task_info.cc"
    "graph/load/model_manager/task_info/label_goto_ex_task_info.cc"
    "graph/load/model_manager/task_info/memcpy_async_task_info.cc"
    "graph/load/model_manager/task_info/memcpy_addr_async_task_info.cc"
    "graph/load/model_manager/task_info/profiler_trace_task_info.cc"
    "graph/load/model_manager/task_info/stream_active_task_info.cc"
    "graph/load/model_manager/task_info/stream_switch_task_info.cc"
    "graph/load/model_manager/task_info/stream_switchn_task_info.cc"
    "graph/load/model_manager/task_info/end_graph_task_info.cc"
    "graph/load/model_manager/task_info/model_exit_task_info.cc"
    "graph/load/model_manager/task_info/super_kernel/super_kernel_factory.cc"
    "graph/load/model_manager/task_info/super_kernel/super_kernel.cc"
    "hybrid/hybrid_davinci_model_stub.cc"
    "ir_build/ge_ir_build.cc"
    "ir_build/attr_options/utils.cc"
    "ir_build/attr_options/keep_dtype_option.cc"
    "ir_build/attr_options/weight_compress_option.cc"
    "ir_build/option_utils.cc"
    "graph/preprocess/insert_op/ge_aipp_op.cc"
    "graph/preprocess/insert_op/util_insert_aipp_op.cc"
    "hybrid/node_executor/aicpu/aicpu_ext_info.cc"
    "graph/build/model_builder.cc"
    "graph/build/task_generator.cc"
    "graph/build/stream_allocator.cc"
    "graph/build/logical_stream_allocator.cc"
    "graph/build/stream_graph_optimizer.cc"
    "graph/build/run_context.cc"
    "graph/build/label_allocator.cc"
    "graph/label/label_maker.cc"
    "graph/label/if_label_maker.cc"
    "graph/label/case_label_maker.cc"
    "graph/label/while_label_maker.cc"
    "graph/label/partitioned_call_label_maker.cc"
    "analyzer/analyzer.cc"
    "graph/build/memory/memory_assigner.cc"
    "graph/build/memory/graph_mem_assigner.cc"
    "graph/build/memory/binary_block_mem_assigner.cc"
    "graph/build/memory/block_mem_assigner.cc"
    "graph/build/memory/hybrid_mem_assigner.cc"
    "graph/build/memory/max_block_mem_assigner.cc"
    "graph/build/memory/var_mem_assign_util.cc"
    "graph/build/memory/buffer_pool_mem_assigner.cc"
)

if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
message("CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
############ libge_runner.so ############
add_library(ge_runner SHARED
    ${TRAIN_SRC_LIST}
    $<TARGET_OBJECTS:$<IF:$<TARGET_EXISTS:msprofiler_fwk>,msprofiler_fwk,msprofiler_fwk_object>>
)

add_library(msprofiler_fwk_object OBJECT IMPORTED GLOBAL)

if (msprofiler_fwk_ext_LIBRARY_DIR)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_fwk_object)
    execute_process(
        COMMAND ar x ${msprofiler_fwk_ext_LIBRARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_fwk_object
    )
    file(GLOB MSPROFILER_FWK_OBJECT_LIST ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_fwk_object/*.o)
    set_property(TARGET msprofiler_fwk_object PROPERTY IMPORTED_OBJECTS ${MSPROFILER_FWK_OBJECT_LIST})
endif()

target_compile_definitions(ge_runner PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    DAVINCI_SUPPORT_PROFILING
    REUSE_MEMORY=1
    FMK_SUPPORT_DUMP
    DAVINCI_CLOUD
    google=ascend_private
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_runner PRIVATE
    -O2
    -fno-common
    -fvisibility=hidden
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_VERSION},7.3.0>:-Werror=unused-variable>
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_VERSION},7.3.0>:-Werror=unused-const-variable -Werror=format>
)

target_include_directories(ge_runner SYSTEM PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/ge/analyzer
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${GE_CODE_DIR}/inc/framework/common
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    ${GE_CODE_DIR}/../inc/external
    ${GE_CODE_DIR}/../inc/cce
    ${GE_CODE_DIR}/../toolchain/ide/ide-daemon/external
    ${GE_CODE_DIR}/../abl/adump/external
    #### blue zone
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
    ${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain
)

target_link_options(ge_runner PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_runner PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    adump_server
    static_mmpa
    ge_proto_common
    ge_proto_client
    -Wl,--no-as-needed
    graph
    ge_common
    ascend_protobuf
    register
    c_sec
    slog
    runtime
    error_manager
    ascend_hal_stub
    -Wl,--as-needed
    json
    -lrt
    -ldl
)

############ libge_compiler.so ############
add_library(ge_compiler SHARED
    ${INFER_SRC_LIST}
)

target_compile_definitions(ge_compiler PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    REUSE_MEMORY=1
    FMK_SUPPORT_DUMP
    FMK_HOST_INFER
    COMPILE_OMG_PACKAGE
    google=ascend_private
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_compiler PRIVATE
    -O2
    -fno-common
    -fvisibility=hidden
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_VERSION},7.3.0>:-Werror=unused-variable>
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_VERSION},7.3.0>:-Werror=unused-const-variable -Werror=format>
)

target_include_directories(ge_compiler SYSTEM PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/ge/analyzer
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${GE_CODE_DIR}/inc/framework/common
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    ${GE_CODE_DIR}/../inc/external
    ${GE_CODE_DIR}/../inc/cce
    ${GE_CODE_DIR}/../toolchain/ide/ide-daemon/external
    ${GE_CODE_DIR}/../abl/adump/external
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
    ${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain
)

target_link_options(ge_compiler PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_compiler PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    static_mmpa
    ge_proto_common
    -Wl,--no-as-needed
    graph
    ge_common
    ascend_protobuf
    register
    c_sec
    error_manager
    slog
    runtime_compile
    -Wl,--as-needed
    json
    -lrt
    -ldl
)

############ libascendcl.so ############
file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/dummy.c CONTENT "")
#add_library(dummy_obj OBJECT ${CMAKE_BINARY_DIR}/dummy.c)
#set(DUMMY_OBJ $<TARGET_OBJECTS:dummy_obj>)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ascendcl_object)

if(EXISTS ${STATIC_ACL_LIB}/libascendcl.a)
    execute_process(
        COMMAND ar x ${STATIC_ACL_LIB}/libascendcl.a
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ascendcl_object
    )
    file(GLOB OBJECT_LIST ${CMAKE_CURRENT_BINARY_DIR}/ascendcl_object/*.o)
else()
    set(OBJECT_LIST ${CMAKE_BINARY_DIR}/dummy.c)
endif()

add_library(opensrc_ascendcl SHARED
    ${OBJECT_LIST}
    $<TARGET_OBJECTS:$<IF:$<TARGET_EXISTS:msprofiler>,msprofiler,msprofiler_object>>
)

add_library(msprofiler_object OBJECT IMPORTED GLOBAL)

if (msprofiler_ext_LIBRARY_DIR)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_object)
    execute_process(
        COMMAND ar x ${msprofiler_ext_LIBRARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_object
    )
    file(GLOB MSPROFILER_OBJECT_LIST ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_object/*.o)
    set_property(TARGET msprofiler_object PROPERTY IMPORTED_OBJECTS ${MSPROFILER_OBJECT_LIST})
endif()

target_compile_definitions(opensrc_ascendcl PRIVATE
    google=ascend_private
)

target_compile_options(opensrc_ascendcl PRIVATE
    -O2
    -fvisibility=hidden
)

target_link_options(opensrc_ascendcl PRIVATE
    -rdynamic
    -Wl,--allow-multiple-definition
    -Wl,-z,muldefs
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_libraries(opensrc_ascendcl PRIVATE
                     -Wl,--whole-archive
                     ge_executor
                     ge_common_static
                     graph_static
                     static_mmpa
                     ascend_protobuf_static
                     register_static
                     error_manager_static
                     adump_server
                     -Wl,--no-whole-archive
                     -Wl,--no-as-needed
                     c_sec
                     runtime
                     slog
                     ascend_hal_stub
                     -Wl,--as-needed
                     -lrt
                     -ldl
                     json
)

set_target_properties(opensrc_ascendcl PROPERTIES
    OUTPUT_NAME ascendcl
)

##################################################################
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_ir_build.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_api.cc
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/stub/gen_stubapi.py ${GE_CODE_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
            && mv ge_ir_build.cc stub_ge_ir_build.cc
            && mv ge_api.cc stub_ge_api.cc
            &&  echo "Generating stub files end."
    #WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    #DEPENDS stub/gen_stubapi.py ${TOP_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(ge_stub
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_ir_build.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_api.cc
)

##################################################################
############ stub/libge_compiler.so ############
add_library(atc_stub_ge_compiler SHARED
    stub_ge_ir_build.cc
)

add_dependencies(atc_stub_ge_compiler ge_stub)

target_compile_options(atc_stub_ge_compiler PRIVATE
    -fno-common
)

target_link_libraries(atc_stub_ge_compiler PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)

set_target_properties(atc_stub_ge_compiler PROPERTIES
    OUTPUT_NAME ge_compiler
    LIBRARY_OUTPUT_DIRECTORY atc_stub
)

target_include_directories(atc_stub_ge_compiler PRIVATE
    ${GE_CODE_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/ge/analyzer
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/framework
    ${GE_CODE_DIR}/inc/framework/common
    ${GE_CODE_DIR}/inc/external
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc/cce
    ${GE_CODE_DIR}/../toolchain/ide/ide-daemon/external
    ${GE_CODE_DIR}/../abl/adump/external
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
)

############ stub/libge_runner.so ############
add_library(fwk_stub_ge_runner SHARED
    stub_ge_api.cc
    stub_ge_ir_build.cc
)

add_dependencies(fwk_stub_ge_runner ge_stub)

target_compile_options(fwk_stub_ge_runner PRIVATE
    -fno-common
)

target_link_libraries(fwk_stub_ge_runner PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)

set_target_properties(fwk_stub_ge_runner PROPERTIES
    OUTPUT_NAME ge_runner
    LIBRARY_OUTPUT_DIRECTORY fwk_stub
)

target_include_directories(fwk_stub_ge_runner PRIVATE
    ${GE_CODE_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/ge/analyzer
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${GE_CODE_DIR}/inc/framework/common
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc/cce
    ${GE_CODE_DIR}/../toolchain/ide/ide-daemon/external
    ${GE_CODE_DIR}/../abl/adump/external
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
)

###############################################################
add_custom_target(
    engine_conf.json ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/engine_conf.json
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/engine_conf.json
    COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/engine_manager/engine_conf.json ${CMAKE_CURRENT_BINARY_DIR}/
)

###############################################################
add_custom_target(
    optimizer_priority.pbtxt ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/optimizer_priority.pbtxt
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/optimizer_priority.pbtxt
    COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/opskernel_manager/optimizer_priority.pbtxt ${CMAKE_CURRENT_BINARY_DIR}/
)

###############################################################

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS ge_runner ge_compiler opensrc_ascendcl OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS atc_stub_ge_compiler fwk_stub_ge_runner OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/engine_conf.json
    ${CMAKE_CURRENT_BINARY_DIR}/optimizer_priority.pbtxt OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)
endif()
