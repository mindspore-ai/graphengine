list(APPEND INCLUDE_DIRECTORIES
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${GE_CODE_DIR}"
        "${GE_CODE_DIR}/inc"
        "${GE_CODE_DIR}/metadef/inc"
        "${GE_CODE_DIR}/ge"
        "${GE_CODE_DIR}/ge/inc"
        "${GE_CODE_DIR}/ge/ir_build"
        "${GE_CODE_DIR}/metadef"
        "${GE_CODE_DIR}/metadef/graph"
        "${GE_CODE_DIR}/inc/external"
        "${GE_CODE_DIR}/inc/framework/common"
        "${GE_CODE_DIR}/metadef/inc/external"
        "${GE_CODE_DIR}/metadef/inc/external/graph"
        "${GE_CODE_DIR}/metadef/inc/graph"
        "${GE_CODE_DIR}/inc/framework"
        "${GE_CODE_DIR}/metadef/inc/common"
        "${GE_CODE_DIR}/metadef/third_party"
        "${GE_CODE_DIR}/metadef/third_party/transformer/inc"
        "${GE_CODE_DIR}/parser"
        "${GE_CODE_DIR}/parser/parser"
        "${GE_CODE_DIR}/third_party/fwkacllib/inc"
        "${GE_CODE_DIR}/third_party/fwkacllib/inc/cce"
        "${GE_CODE_DIR}/third_party/fwkacllib/inc/ops"
        "${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain"
        "${GE_CODE_DIR}/tests/ut/ge"
        "${GE_CODE_DIR}/tests/ut/common"
        "${CMAKE_BINARY_DIR}"
        "${CMAKE_BINARY_DIR}/proto/ge"
        "${CMAKE_BINARY_DIR}/proto/ge/proto"
        )

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cc" "*.CC" "*.cpp" "*.CPP" "*.c++")

# ---- Target : stub Host engine ----
add_library(fe SHARED ${SOURCES})

target_include_directories(fe
        PUBLIC
        ${INCLUDE_DIRECTORIES}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

target_compile_definitions(fe PRIVATE
        google=ascend_private
        FMK_SUPPORT_DUMP
        )

target_compile_options(fe PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -Werror=format
        )

target_link_libraries(fe PUBLIC
        $<BUILD_INTERFACE:intf_pub> ${STUB_LIBS} metadef_graph -lmmpa -L${GE_CODE_DIR}/third_party/prebuild/x86_64 -lrt -ldl -lpthread -lgcov
        )

set_target_properties(fe PROPERTIES CXX_STANDARD 11)
