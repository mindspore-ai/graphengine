include_guard(GLOBAL)

add_library(runtime_headers INTERFACE)
target_include_directories(runtime_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../external>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../external/runtime>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/runtime>
    $<INSTALL_INTERFACE:include/runtime/external>
    $<INSTALL_INTERFACE:include/runtime/external/runtime>
)
