set(PROTO_LIST
    "${METADEF_DIR}/proto/tensorflow/attr_value.proto"
    "${METADEF_DIR}/proto/tensorflow/function.proto"
    "${METADEF_DIR}/proto/tensorflow/graph.proto"
    "${METADEF_DIR}/proto/tensorflow/node_def.proto"
    "${METADEF_DIR}/proto/tensorflow/op_def.proto"
    "${METADEF_DIR}/proto/tensorflow/resource_handle.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor_shape.proto"
    "${METADEF_DIR}/proto/tensorflow/types.proto"
    "${METADEF_DIR}/proto/tensorflow/versions.proto"
    "${METADEF_DIR}/proto/task.proto"
    "${METADEF_DIR}/proto/om.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

set(SRC_LIST
    "register.cpp"
    "ops_kernel_builder_registry.cc"
    "graph_optimizer/graph_fusion/graph_fusion_pass_base.cc"
    "graph_optimizer/graph_fusion/fusion_pass_registry.cc"
    "graph_optimizer/graph_fusion/fusion_pattern.cc"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass.cc"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.cc"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"
    "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.cc"
    "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.cc"
    "graph_optimizer/buffer_fusion/buffer_fusion_pattern.cc"
    "graph_optimizer/fusion_statistic/fusion_statistic_recorder.cc"
    "register_format_transfer.cc"
    "op_kernel_registry.cpp"
    "auto_mapping_util.cpp"
    "host_cpu_context.cc"
    "tensor_assign.cpp"
    "infer_data_slice_registry.cc"
    "scope/scope_graph.cc"
    "scope/scope_pass.cc"
    "scope/scope_pattern.cc"
    "scope/scope_util.cc"
    "scope/scope_pass_registry.cc"
)

############ libregister.so ############
add_library(register SHARED ${SRC_LIST} ${PROTO_SRCS})

target_compile_options(register PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations>
)

target_compile_definitions(register PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(register PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    ${METADEF_DIR}/..
    ${METADEF_DIR}/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${METADEF_DIR}/../inc
    #### temp independent ####
    ${METADEF_DIR}/../graphengine/inc
    ${METADEF_DIR}/../graphengine/inc/framework
    ${METADEF_DIR}/../graphengine/inc/external
    #### temp in ge ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../inc/framework
    ${METADEF_DIR}/../inc/external
    #### temp in parser ####
    ${METADEF_DIR}/../../graphengine/inc
    ${METADEF_DIR}/../../graphengine/inc/framework
    ${METADEF_DIR}/../../graphengine/inc/external
    ${METADEF_DIR}/../../inc
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
    ${METADEF_DIR}/../third_party/fwkacllib/inc
    #### blue independent compile #####
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/ge/inc
    ${METADEF_DIR}/third_party/graphengine/inc/framework
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
)

target_link_libraries(register PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--whole-archive
    op_tiling_o2
    -Wl,--no-whole-archive
    -Wl,--no-as-needed
    ascend_protobuf
    c_sec
    slog
    graph
    -Wl,--as-needed
    json
)

############ libregister.a ############
add_library(register_static STATIC ${SRC_LIST} ${PROTO_SRCS}
    "op_tiling.cpp"
    "op_tiling_registry.cpp"
)

target_compile_options(register_static PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:/utf-8>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_compile_definitions(register_static PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    LOG_CPP
)
target_include_directories(register_static PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    ${METADEF_DIR}/..
    ${METADEF_DIR}/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${METADEF_DIR}/../inc
    #### temp independent ####
    ${METADEF_DIR}/../graphengine/inc
    ${METADEF_DIR}/../graphengine/inc/framework
    ${METADEF_DIR}/../graphengine/inc/external
    #### temp in ge ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../inc/framework
    ${METADEF_DIR}/../inc/external
    #### temp in parser ####
    ${METADEF_DIR}/../../graphengine/inc
    ${METADEF_DIR}/../../graphengine/inc/framework
    ${METADEF_DIR}/../../graphengine/inc/external
    ${METADEF_DIR}/../../inc
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
    #### blue independent compile #####
    ${METADEF_DIR}/../third_party/fwkacllib/inc 
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/framework
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
)

target_link_libraries(register_static PRIVATE
    ascend_protobuf
    c_sec
    json
    $<BUILD_INTERFACE:intf_pub>
)

set_target_properties(register_static PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    OUTPUT_NAME $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,libregister,register>
)

############ libop_tiling_o2.a ############
add_library(op_tiling_o2 STATIC
    "op_tiling.cpp"
    "op_tiling_registry.cpp"
)

target_include_directories(op_tiling_o2 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    ${METADEF_DIR}/..
    ${METADEF_DIR}/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${METADEF_DIR}/../inc
    #### temp independent ####
    ${METADEF_DIR}/../graphengine/inc
    ${METADEF_DIR}/../graphengine/inc/framework
    ${METADEF_DIR}/../graphengine/inc/external
    #### temp in ge ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../inc/framework
    ${METADEF_DIR}/../inc/external
    #### temp in parser ####
    ${METADEF_DIR}/../../graphengine/inc
    ${METADEF_DIR}/../../graphengine/inc/framework
    ${METADEF_DIR}/../../graphengine/inc/external
    ${METADEF_DIR}/../../inc
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
    ${METADEF_DIR}/../third_party/fwkacllib/inc
    #### blue independent compile ####
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
)

target_compile_options(op_tiling_o2 PRIVATE
    -O2
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations>
)

target_compile_definitions(op_tiling_o2 PRIVATE
    LOG_CPP
)

target_link_libraries(op_tiling_o2 PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    json
    c_sec
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS register register_static OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}
)
