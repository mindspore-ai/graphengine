set(SRC_LIST
    "memory_assigner.cc"
    "graph_mem_assigner.cc"
    "binary_block_mem_assigner.cc"
    "block_mem_assigner.cc"
    "hybrid_mem_assigner.cc"
    "max_block_mem_assigner.cc"
    "var_mem_assign_util.cc"
)

############ libge_memory.a ############
add_library(ge_memory STATIC ${SRC_LIST})

target_compile_options(ge_memory PRIVATE
    -Werror
    -O2
)

target_compile_definitions(ge_memory PRIVATE
    google=ascend_private
)

target_link_libraries(ge_memory PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
    c_sec
)

target_include_directories(ge_memory PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${GE_CODE_DIR}/inc/framework
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
)
