############ libge_runtime.so ############
set(GE_SRC_LIST
    "model_runner.cc"
    "runtime_model.cc"
    "output.cc"
    "task/aicpu_task.cc"
    "task/cce_task.cc"
    "task/tbe_task.cc"
    "task/event_record_task.cc"
    "task/event_wait_task.cc"
    "task/stream_active_task.cc"
    "task/stream_switch_task.cc"
    "task/hccl_task.cc"
    "task/memcpy_async_task.cc"
    "task/profiler_task.cc"
)

add_library(ge_runtime SHARED ${GE_SRC_LIST})

target_compile_options(ge_runtime PRIVATE
    -Werror
    -O2
)

target_compile_definitions(ge_runtime PRIVATE 
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
)

target_include_directories(ge_runtime PRIVATE
    ${TOP_DIR}
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/graph
    ${TOP_DIR}/inc/external
    ${TOP_DIR}/inc/framework
    ${TOP_DIR}/inc/framework/common
    ${TOP_DIR}/inc/framework/ge_runtime
    ${TOP_DIR}/inc/cce
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
)

target_link_libraries(ge_runtime PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    slog
    runtime
    c_sec
    -Wl,--as-needed
    -lrt
    -ldl
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS ge_runtime OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
