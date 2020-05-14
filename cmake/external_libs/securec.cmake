graphengine_add_pkg(securec
        VER 1.1.10
        URL https://gitee.com/openeuler/bounds_checking_function/repository/archive/v1.1.10.tar.gz
        MD5 0782dd2351fde6920d31a599b23d8c91
        LIBS c_sec
        PATCHES ${GE_SOURCE_DIR}/third_party/patch/securec/securec.patch001
        CMAKE_OPTION " "
        )
include_directories(${securec_INC})
file(COPY ${securec_INC}/../lib/libc_sec.so DESTINATION ${CMAKE_SOURCE_DIR}/build/graphengine)
add_library(graphengine::securec ALIAS securec::c_sec)
