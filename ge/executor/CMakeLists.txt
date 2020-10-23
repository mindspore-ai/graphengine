set(PROTO_LIST
    "${METADEF_DIR}/proto/om.proto"
    "${METADEF_DIR}/proto/ge_ir.proto"
    "${METADEF_DIR}/proto/insert_op.proto"
    "${METADEF_DIR}/proto/task.proto"
    "${METADEF_DIR}/proto/op_mapping_info.proto"
    "${METADEF_DIR}/proto/dump_task.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

set(SRC_LIST
    "ge_executor.cc"
    "../common/profiling/profiling_manager.cc"
    "../common/ge/plugin_manager.cc"
    "../common/ge/op_tiling_manager.cc"
    "../common/dump/dump_properties.cc"
    "../common/dump/dump_manager.cc"
    "../common/dump/dump_op.cc"
    "../graph/load/graph_loader.cc"
    "../graph/execute/graph_execute.cc"
    "../omm/csa_interact.cc"
    "../graph/manager/graph_manager_utils.cc"
    "../graph/manager/graph_var_manager.cc"
    "../graph/manager/graph_mem_allocator.cc"
    "../graph/manager/graph_caching_allocator.cc"
    "../graph/manager/trans_var_data_utils.cc"
    "../graph/manager/util/debug.cc"
    "../graph/manager/rdma_pool_allocator.cc"
    "../hybrid/node_executor/aicpu/aicpu_ext_info.cc"
    "../model/ge_model.cc"
    "../model/ge_root_model.cc"
    "../graph/load/new_model_manager/davinci_model.cc"
    "../graph/load/new_model_manager/davinci_model_parser.cc"
    "../graph/load/new_model_manager/model_manager.cc"
    "../graph/load/new_model_manager/tbe_handle_store.cc"
    "../graph/load/new_model_manager/cpu_queue_schedule.cc"
    "../graph/load/new_model_manager/model_utils.cc"
    "../graph/load/new_model_manager/aipp_utils.cc"
    "../graph/load/new_model_manager/data_inputer.cc"
    "../graph/load/new_model_manager/data_dumper.cc"
    "../graph/load/new_model_manager/zero_copy_task.cc"
    "../graph/load/new_model_manager/zero_copy_offset.cc"
    "../graph/load/new_model_manager/task_info/task_info.cc"
    "../graph/load/new_model_manager/task_info/event_record_task_info.cc"
    "../graph/load/new_model_manager/task_info/event_wait_task_info.cc"
    "../graph/load/new_model_manager/task_info/fusion_start_task_info.cc"
    "../graph/load/new_model_manager/task_info/fusion_stop_task_info.cc"
    "../graph/load/new_model_manager/task_info/kernel_ex_task_info.cc"
    "../graph/load/new_model_manager/task_info/kernel_task_info.cc"
    "../graph/load/new_model_manager/task_info/label_set_task_info.cc"
    "../graph/load/new_model_manager/task_info/label_switch_by_index_task_info.cc"
    "../graph/load/new_model_manager/task_info/label_goto_ex_task_info.cc"
    "../graph/load/new_model_manager/task_info/memcpy_async_task_info.cc"
    "../graph/load/new_model_manager/task_info/memcpy_addr_async_task_info.cc"
    "../graph/load/new_model_manager/task_info/profiler_trace_task_info.cc"
    "../graph/load/new_model_manager/task_info/stream_active_task_info.cc"
    "../graph/load/new_model_manager/task_info/stream_switch_task_info.cc"
    "../graph/load/new_model_manager/task_info/stream_switchn_task_info.cc"
    "../graph/load/new_model_manager/task_info/end_graph_task_info.cc"
    "../graph/load/new_model_manager/task_info/model_exit_task_info.cc"
    "../graph/load/new_model_manager/task_info/super_kernel/super_kernel_factory.cc"
    "../graph/load/new_model_manager/task_info/super_kernel/super_kernel.cc"
    "../opskernel_manager/ops_kernel_builder_manager.cc"
    "../single_op/single_op_manager.cc"
    "../single_op/single_op_model.cc"
    "../single_op/single_op.cc"
    "../single_op/stream_resource.cc"
    "../single_op/task/op_task.cc"
    "../single_op/task/build_task_utils.cc"
    "../single_op/task/tbe_task_builder.cc"
    "../single_op/task/aicpu_task_builder.cc"
    "../single_op/task/aicpu_kernel_task_builder.cc"
    "../hybrid/hybrid_davinci_model_stub.cc"
)

######## libge_executor.a ########
add_library(ge_executor STATIC ${SRC_LIST} ${PROTO_HDRS})

target_compile_options(ge_executor PRIVATE
    -Werror
    -O2
)

target_compile_definitions(ge_executor PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    DAVINCI_SUPPORT_PROFILING
)

target_include_directories(ge_executor PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    ${GE_CODE_DIR}/../inc/cce    
)

target_link_libraries(ge_executor PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    json
    protobuf
    c_sec
    -lrt
    -ldl
)
