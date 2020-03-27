set(ge_gtest_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(ge_gtest_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

graphengine_add_pkg(ge_gtest
        VER 1.8.0
        LIBS gtest gtest_main
        URL https://github.com/google/googletest/archive/release-1.8.0.tar.gz
        MD5 16877098823401d1bf2ed7891d7dce36
        CMAKE_OPTION -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON
        -DCMAKE_MACOSX_RPATH=TRUE -Dgtest_disable_pthreads=ON)

add_library(graphengine::gtest ALIAS ge_gtest::gtest)
add_library(graphengine::gtest_main ALIAS ge_gtest::gtest_main)
include_directories(${ge_gtest_INC})
file(COPY ${ge_gtest_INC}/../lib/libgtest.so DESTINATION ${CMAKE_SOURCE_DIR}/build/graphengine)
file(COPY ${ge_gtest_INC}/../lib/libgtest_main.so DESTINATION ${CMAKE_SOURCE_DIR}/build/graphengine)
