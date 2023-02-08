add_subdirectory(slice)
if (BUILD_METADEF)
    add_subdirectory(metadef)
endif()
if (BUILD_PARSER)
    add_subdirectory(parser)
endif()

##################################################################
set(SRC_FORMAT_TRANSFERS
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_dhwcn_fractal_z_3D.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_dhwnc_fractal_z_3D_transpose.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_nz.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_z.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_zz.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_z_tbe.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_hwcn.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_nchw.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_nhwc.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nc1hwc0_nchw.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nchw_to_fz_c04.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nchw_nc1hwc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_transpose.cc"
    "${AIR_CODE_DIR}/base/formats/formats.cc"
    "${AIR_CODE_DIR}/base/formats/ge_format_util.cc"
    "${AIR_CODE_DIR}/base/formats/register_format_transfer.cc"
    "${AIR_CODE_DIR}/base/formats/utils/formats_trans_utils.cc"
)

set(SRC_LIST
    "common/b_cast.cc"
    "common/plugin/runtime_plugin_loader.cc"
    "common/plugin/opp_so_manager.cc"
    "common/context/local_context.cc"
    "common/omg_util.cc"
    "common/context/properties_manager.cc"
    "common/op/transop_util.cc"
    "common/single_op_parser.cc"
    "common/data_dependent_interpreter.cc"
    "graph/partition/default_subgraph_builder.cc"
    "graph/partition/graph_partitioner.cc"
    "graph/partition/graph_partitioner_builder.cc"
    "graph/partition/graph_partitioner_cluster_dict.cc"
    "graph/partition/nodes_cluster.cc"
    "graph/partition/partition_utils.cc"
    "graph/partition/subgraph_builder_for_optimization.cc"
    "graph/partition/merge_adjacent_policy.cc"
    "graph/bin_cache/node_bin_selector_factory.cc"
    "graph/bin_cache/node_compile_cache_module.cc"
    "graph/bin_cache/op_binary_cache.cc"
    "local_engine/engine/host_cpu_engine.cc"
)

set(SRC_BASE_LIST
    "common/context/ctx.cc"
    "common/helper/file_saver.cc"
    "common/tbe_handle_store/cust_aicpu_kernel_store.cc"
    "common/debug/memory_dumper.cc"
    "common/dump/dump_manager.cc"
    "common/dump/dump_op.cc"
    "common/dump/exception_dumper.cc"
    "common/dump/dump_properties.cc"
    "common/fmk_error_codes.cc"
    "common/datatype_transfer.cc"
    "common/fp16_t.cc"
    "common/plugin/datatype_util.cc"
    "common/plugin/op_tiling_manager.cc"
    "common/ge_inner_error_codes.cc"
    "common/helper/model_helper.cc"
    "common/helper/om_file_helper.cc"
    "common/tbe_handle_store/kernel_store.cc"
    "common/model/ge_model.cc"
    "common/model/ge_root_model.cc"
    "common/global_variables/diagnose_switch.cc"
    "pne/model/flow_model.cc"
    "pne/model/flow_model_om_loader.cc"
    "pne/model/flow_model_om_saver.cc"
    "common/model/model_relation.cc"
    "common/helper/model_parser_base.cc"
    "common/helper/model_saver.cc"
    "common/op/ge_op_utils.cc"
    "common/tbe_handle_store/tbe_kernel_store.cc"
    "common/thread_pool.cc"
    "common/types.cc"
    "common/util.cc"
    "common/proto_util.cc"
    "common/file_constant_util.cc"
    "common/tbe_handle_store/tbe_handle_store.cc"
    "common/tbe_handle_store/bin_register_utils.cc"
    "graph/manager/graph_var_manager.cc"
    "graph/manager/graph_manager_utils.cc"
    "graph/build/memory/var_mem_assign_util.cc"
    "common/profiling/profiling_properties.cc"
    "exec_runtime/deploy/deploy_planner.cc"
    "exec_runtime/execution_runtime.cc"
    "common/profiling/profiling_definitions.cc"
    "common/ge_inner_attrs.cc"
    "pne/manager/process_node_engine_manager.cc"
    "pne/manager/heterogeneous_var_manager.cc"
    "common/op_so_store/op_so_store.cc"
)

if (NOT ENABLE_D AND NOT ENABLE_ACL)
############ libge_common.so ############
add_library(ge_common SHARED
    ${SRC_LIST}
)

add_dependencies(ge_common
    graphengine_protos
)

target_compile_definitions(ge_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_common PRIVATE
    -fvisibility=default
    -O2
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_include_directories(ge_common PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_options(ge_common PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common
    PRIVATE
        intf_pub
        msprof_headers
        runtime_headers
        datagw_headers
        static_mmpa
        -Wl,--no-as-needed
        graph
        ge_common_base
        ascend_protobuf
        c_sec
        error_manager
        slog
        platform
        -Wl,--as-needed
        json
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        air_headers
)

############ libge_common_base.so ############
add_library(ge_common_base SHARED
    ${SRC_BASE_LIST}
    ${SRC_FORMAT_TRANSFERS}
)

add_dependencies(ge_common_base
    graphengine_protos
)

target_compile_definitions(ge_common_base PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_common_base PRIVATE
    -fvisibility=default
    -O2
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_include_directories(ge_common_base PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_options(ge_common_base PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common_base
    PRIVATE
        intf_pub
        msprof_headers
        runtime_headers
        datagw_headers
        static_mmpa
        -Wl,--no-as-needed
        graph_base
        ascend_protobuf
        c_sec
        error_manager
        slog
        platform
        -Wl,--as-needed
        json
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        air_headers
)

############ libge_common.a ############
add_library(ge_common_static STATIC
    ${SRC_LIST}
    ${SRC_BASE_LIST}
    ${SRC_FORMAT_TRANSFERS}
)

add_dependencies(ge_common_static
    graphengine_protos
)

target_compile_definitions(ge_common_static PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    LOG_CPP
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_common_static PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-fvisibility=hidden -O2 -Werror -Wextra -Wfloat-equal -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_include_directories(ge_common_static PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_libraries(ge_common_static PRIVATE
    intf_pub
    slog_headers
    msprof_headers
    runtime_headers
    mmpa_headers
    metadef_headers
    datagw_headers
    ascend_protobuf_static
    json
    c_sec
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

else ()
############ libge_common.so w/static protobuf ############
add_library(ge_common SHARED
    ${SRC_LIST}
)
add_dependencies(ge_common
    graphengine_protos
)

target_compile_definitions(ge_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    LOG_CPP
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_common PRIVATE
    -fvisibility=default
    -O2
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_include_directories(ge_common PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_options(ge_common PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common
    PRIVATE
        intf_pub
        msprof_headers
        runtime_headers
        datagw_headers
        ascend_protobuf_static
        -Wl,--no-as-needed
        graph
        ge_common_base
        c_sec
        error_manager
        slog
        platform
        static_mmpa
        -Wl,--as-needed
        json
        -lrt
        -ldl
    PUBLIC
        air_headers
)

############ libge_common_base.so w/static protobuf ############
add_library(ge_common_base SHARED
    ${SRC_BASE_LIST}
    ${SRC_FORMAT_TRANSFERS}
)
add_dependencies(ge_common_base
    graphengine_protos
)

target_compile_definitions(ge_common_base PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    LOG_CPP
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_common_base PRIVATE
    -fvisibility=default
    -O2
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_include_directories(ge_common_base PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_options(ge_common_base PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common_base
    PRIVATE
        intf_pub
        msprof_headers
        runtime_headers
        datagw_headers
        ascend_protobuf_static
        -Wl,--no-as-needed
        graph_base
        c_sec
        error_manager
        slog
        platform
        static_mmpa
        -Wl,--as-needed
        json
        -lrt
        -ldl
    PUBLIC
        air_headers
)
endif ()

############ libge_common_stub.so w/static protobuf ############
add_library(ge_common_stub SHARED
    ${AIR_CODE_DIR}/base/stub/common_stub.cc
)

target_compile_definitions(ge_common_stub PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    LOG_CPP
    FUNC_VISIBILITY
)

target_compile_options(ge_common_stub PRIVATE
    -fvisibility=default
    -O2
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_include_directories(ge_common_stub PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_options(ge_common_stub PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common_stub
    PRIVATE
        intf_pub
        slog_headers
        msprof_headers
        mmpa_headers
        runtime_headers
        metadef_headers
        ascend_protobuf_static
        -Wl,--no-as-needed
        c_sec
        -Wl,--as-needed
        json
        -lrt
        -ldl
    PUBLIC
        air_headers
)

set_target_properties(ge_common_stub PROPERTIES
    OUTPUT_NAME ge_common
    LIBRARY_OUTPUT_DIRECTORY stub
)

install(TARGETS ge_common_stub OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub
)
