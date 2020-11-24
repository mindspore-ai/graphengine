set(PROTO_LIST
    "${METADEF_DIR}/proto/task.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

set(SRC_LIST
    "engine/host_cpu_engine.cc"
    "ops_kernel_store/host_cpu_ops_kernel_info.cc"
    "ops_kernel_store/op/op_factory.cc"
    "ops_kernel_store/op/host_op.cc" 
)

set(CPU_OPS_KERNEL_LIST
    "ops_kernel_store/host_cpu_ops_kernel_builder.cc"
)

############ libhost_cpu_engine.so ############
add_library(host_cpu_engine SHARED ${SRC_LIST} ${PROTO_HDRS})

target_compile_options(host_cpu_engine PRIVATE
    -Werror
)

target_include_directories(host_cpu_engine PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
)

target_link_libraries(host_cpu_engine PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    protobuf
    c_sec
    graph
    register
    slog
    runtime
    -Wl,--as-needed
)

############ atcstub/libhost_cpu_engine.so ############
add_library(atc_host_cpu_engine SHARED ${SRC_LIST} ${PROTO_HDRS})

target_compile_options(atc_host_cpu_engine PRIVATE
    -Werror
)

target_compile_definitions(atc_host_cpu_engine PRIVATE
    COMPILE_OMG_PACKAGE
)

target_include_directories(atc_host_cpu_engine PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
)

target_link_libraries(atc_host_cpu_engine PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    protobuf
    c_sec
    graph
    register
    slog
    runtime_compile
    -Wl,--as-needed
)

set_target_properties(atc_host_cpu_engine PROPERTIES
    OUTPUT_NAME host_cpu_engine
    LIBRARY_OUTPUT_DIRECTORY atclib 
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS host_cpu_engine OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS atc_host_cpu_engine  OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/atclib
)
