set(PROTO_LIST
    "${METADEF_DIR}/proto/om.proto"
    "${METADEF_DIR}/proto/ge_ir.proto"
    "${METADEF_DIR}/proto/insert_op.proto"
    "${METADEF_DIR}/proto/task.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

set(SRC_LIST
    "main.cc"
    "single_op_parser.cc"
    "../session/omg.cc"
    "../ir_build/atc_ir_common.cc" 
)

############ atc ############
add_executable(atc ${SRC_LIST} ${PROTO_HDRS})

target_compile_options(atc PRIVATE 
    -Werror
    -O2
)

target_compile_definitions(atc PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    COMPILE_OMG_PACKAGE
)

target_include_directories(atc PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/graphengine
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/common/inc/external
    ${GE_CODE_DIR}/common/inc/external/graph
    ${GE_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    ${GE_CODE_DIR}/../inc/common
)

target_link_libraries(atc PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    protobuf
    ge_common
    register
    c_sec
    graph
    error_manager
    ge_compiler
    parser_common
    gflags
    json
    runtime_compile
    slog
    mmpa
    -lrt
    -ldl
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS atc OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
