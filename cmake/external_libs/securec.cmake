graphengine_add_pkg(securec
        VER 1.1.10
        URL https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.10.tar.gz
        MD5 193f0ca5246c1dd84920db34d2d8249f
        LIBS c_sec
        PATCHES ${GE_SOURCE_DIR}/third_party/patch/securec/securec.patch001
        CMAKE_OPTION "-DCMAKE_BUILD_TYPE=Release"
        )
include_directories(${securec_INC})
file(COPY ${securec_INC}/../lib/libc_sec.so DESTINATION ${CMAKE_SOURCE_DIR}/build/graphengine)
add_library(graphengine::securec ALIAS securec::c_sec)