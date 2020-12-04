set(PROTO_LIST
    "${METADEF_DIR}/proto/task.proto"
)

set(SRC_LIST
    "engine/ge_local_engine.cc"
    "ops_kernel_store/ge_local_ops_kernel_info.cc"
    "ops_kernel_store/op/op_factory.cc"
    "ops_kernel_store/op/op.cc"
    "ops_kernel_store/op/ge_deleted_op.cc"
    "ops_kernel_store/op/no_op.cc"
)

set(OPS_KERNEL_SRC_LIST
    "ops_kernel_store/ge_local_ops_kernel_builder.cc"
    "ops_kernel_store/op/op_factory.cc"
    "ops_kernel_store/op/op.cc"
    "ops_kernel_store/op/ge_deleted_op.cc"
    "ops_kernel_store/op/no_op.cc"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

############ libge_local_engine.so ############
add_library(ge_local_engine SHARED ${SRC_LIST} ${PROTO_HDRS})

target_compile_options(ge_local_engine PRIVATE
    -Werror
)

target_compile_definitions(ge_local_engine PRIVATE
    google=ascend_private
)

target_include_directories(ge_local_engine PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
)

target_link_libraries(ge_local_engine PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    graph
    ascend_protobuf
    register
    c_sec
    slog
    runtime
    -Wl,--as-needed
)

######### atclib/libge_local_engine.so #############
add_library(atc_ge_local_engine SHARED ${SRC_LIST} ${PROTO_HDRS})

target_compile_options(atc_ge_local_engine PRIVATE
    -Werror
)

target_compile_definitions(atc_ge_local_engine PRIVATE
    COMPILE_OMG_PACKAGE
    google=ascend_private
)

target_include_directories(atc_ge_local_engine PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
)

target_link_libraries(atc_ge_local_engine PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    graph
    ascend_protobuf
    register
    c_sec
    slog
    runtime_compile
    -Wl,--as-needed
)

set_target_properties(atc_ge_local_engine PROPERTIES
    OUTPUT_NAME ge_local_engine
    LIBRARY_OUTPUT_DIRECTORY atclib
)

############ libge_local_opskernel_builder.so ############
add_library(ge_local_opskernel_builder SHARED ${OPS_KERNEL_SRC_LIST} ${PROTO_HDRS})

target_compile_options(ge_local_opskernel_builder PRIVATE
    -Werror
)

target_compile_definitions(ge_local_opskernel_builder PRIVATE
    google=ascend_private
)

target_include_directories(ge_local_opskernel_builder PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
)

target_link_libraries(ge_local_opskernel_builder PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    ascend_protobuf
    c_sec
    slog
    register
    graph
    -Wl,--as-needed
)

############ atclib/libge_local_opskernel_builder.so ############
add_library(atc_ge_local_opskernel_builder SHARED ${OPS_KERNEL_SRC_LIST} ${PROTO_HDRS})

target_compile_options(atc_ge_local_opskernel_builder PRIVATE
    -Werror
)

target_compile_definitions(atc_ge_local_opskernel_builder PRIVATE
    google=ascend_private
)

target_include_directories(atc_ge_local_opskernel_builder PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
)

target_link_libraries(atc_ge_local_opskernel_builder PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    ascend_protobuf
    c_sec
    slog
    register
    graph
    -Wl,--as-needed
)

set_target_properties(atc_ge_local_opskernel_builder PROPERTIES
    OUTPUT_NAME ge_local_opskernel_builder
    LIBRARY_OUTPUT_DIRECTORY atclib
)

############ libge_local_opskernel_builder.a ############
add_library(ge_local_opskernel_builder_static SHARED ${OPS_KERNEL_SRC_LIST} ${PROTO_HDRS})

target_compile_options(ge_local_opskernel_builder_static PRIVATE
    -Werror
)

target_compile_definitions(ge_local_opskernel_builder_static PRIVATE
    google=ascend_private
)

target_include_directories(ge_local_opskernel_builder_static PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
)

target_link_libraries(ge_local_opskernel_builder_static PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
    c_sec
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS ge_local_engine ge_local_opskernel_builder OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS atc_ge_local_engine atc_ge_local_opskernel_builder OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/atclib
)
