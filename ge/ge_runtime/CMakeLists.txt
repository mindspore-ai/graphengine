# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
############ libge_runtime.so ############
set(GE_SRC_LIST
    "model_runner.cc"
    "runtime_model.cc"
    "output.cc"
    "task/*.cc"
)

add_library(ge_runtime SHARED ${GE_SRC_LIST})

target_compile_options(ge_runtime PRIVATE
    -Werror
    -O2
)

target_compile_definitions(ge_runtime PUBLIC 
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
    graph
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
