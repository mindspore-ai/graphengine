set(Eigen3_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(Eigen3_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(Eigen3_NS "ge_")

if (ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/eigen-git-mirrorsource/repository/archive/3.3.7.tar.gz")
    set(MD5 "cf6552a5d90c1aca4b5e0b011f65ea93")
else()
    set(REQ_URL "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz")
    set(MD5 "9e30f67e8531477de4117506fe44669b")
endif ()

graphengine_add_pkg(Eigen3
        VER 3.3.7
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DBUILD_TESTING=OFF)

find_package(Eigen3 3.3.7 REQUIRED ${GE_FIND_NO_DEFAULT_PATH})
set_property(TARGET Eigen3::Eigen PROPERTY IMPORTED_GLOBAL TRUE)
add_library(graphengine::eigen ALIAS Eigen3::Eigen)
include_directories(${EIGEN3_INCLUDE_DIRS})
