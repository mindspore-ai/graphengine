if (HAVE_PROTOBUF)
    return()
endif()

include(ExternalProject)
include(GNUInstallDirs)

if ((${CMAKE_INSTALL_PREFIX} STREQUAL /usr/local) OR
    (${CMAKE_INSTALL_PREFIX} STREQUAL "C:/Program Files (x86)/ascend"))
    set(CMAKE_INSTALL_PREFIX ${METADEF_DIR}/output CACHE STRING "path for install()" FORCE)
    message(STATUS "No install prefix selected, default to ${CMAKE_INSTALL_PREFIX}.")
endif()

if (ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v3.8.0.tar.gz")
    set(MD5 "eba86ae9f07ba5cfbaf8af3bc4e84236")
else()
    set(REQ_URL "https://github.com/protocolbuffers/protobuf/archive/v3.8.0.tar.gz")
    set(MD5 "3d9e32700639618a4d2d342c99d4507a")
endif ()

set(protobuf_CXXFLAGS "-Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -Dgoogle=ascend_private")
set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
ExternalProject_Add(protobuf_build
                    URL https://github.com/protocolbuffers/protobuf/archive/v3.8.0.tar.gz
                    CONFIGURE_COMMAND ${CMAKE_COMMAND}
                    -Dprotobuf_WITH_ZLIB=OFF
                    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_LINKER=${CMAKE_LINKER}
                    -DCMAKE_AR=${CMAKE_AR}
                    -DCMAKE_RANLIB=${CMAKE_RANLIB}
                    -DLIB_PREFIX=ascend_
                    -Dprotobuf_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS=${protobuf_CXXFLAGS} -DCMAKE_CXX_LDFLAGS=${protobuf_LDFLAGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/protobuf <SOURCE_DIR>/cmake
                    BUILD_COMMAND $(MAKE)
                    INSTALL_COMMAND $(MAKE) install
                    EXCLUDE_FROM_ALL TRUE
)
include(GNUInstallDirs)

set(PROTOBUF_SHARED_PKG_DIR ${CMAKE_INSTALL_PREFIX}/protobuf)

add_library(ascend_protobuf SHARED IMPORTED)

file(MAKE_DIRECTORY ${PROTOBUF_SHARED_PKG_DIR}/include)

set_target_properties(ascend_protobuf PROPERTIES
                      IMPORTED_LOCATION ${PROTOBUF_SHARED_PKG_DIR}/${CMAKE_INSTALL_LIBDIR}/libascend_protobuf.so
)

target_include_directories(ascend_protobuf INTERFACE ${PROTOBUF_SHARED_PKG_DIR}/include)

set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(FILES ${PROTOBUF_SHARED_PKG_DIR}/${CMAKE_INSTALL_LIBDIR}/ascend_protobuf.so.3.8.0.0 OPTIONAL
        DESTINATION ${INSTALL_LIBRARY_DIR})
install(FILES ${PROTOBUF_SHARED_PKG_DIR}/${CMAKE_INSTALL_LIBDIR}/ascend_protobuf.so OPTIONAL
        DESTINATION ${INSTALL_LIBRARY_DIR})

add_dependencies(ascend_protobuf protobuf_build)

#set(HAVE_PROTOBUF TRUE CACHE BOOL "protobuf build add")
set(HAVE_PROTOBUF TRUE)
