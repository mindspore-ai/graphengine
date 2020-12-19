cmake_minimum_required(VERSION 3.14)
project (MetaDef[CXX])

set(METADEF_DIR ${CMAKE_CURRENT_LIST_DIR})

if (DEFINED ENV{D_PKG_SERVER})
    set(METADEF_PB_PKG $ENV{D_PKG_SERVER})
endif()

option(ENABLE_OPEN_SRC "Enable graphengine compile in opensource." FALSE)

if (ENABLE_OPEN_SRC)
    set(HI_PYTHON python3)

    include(cmake/external_libs/protobuf_shared.cmake)
    include(cmake/external_libs/protoc.cmake)
    include(cmake/external_libs/securec.cmake)
    include(cmake/external_libs/json.cmake)
    include(cmake/FindModule.cmake)
    include(cmake/intf_pub_linux.cmake)

    if(DEFINED ENV{D_LINK_PATH})
        # D_LINK_PATH is set
        set(GE_LIB_PATH $ENV{D_LINK_PATH})
        set(GE_SYS_ARCH "")
        if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
            # x86 ubuntu
            set(GE_SYS_ARCH "x86_64")
        elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
            # arm euleros
            set(GE_SYS_ARCH "aarch64")
        else()
            message(FATAL_ERROR "Running on a unsupported architecture: ${SYSTEM_TYPE}, build terminated")
        endif()
        set(GE_LIB_PATH ${GE_LIB_PATH}/${GE_SYS_ARCH})
        find_module(slog libslog.so ${GE_LIB_PATH})
        find_module(static_mmpa libmmpa.a ${GE_LIB_PATH})
        find_module(error_manager liberror_manager.so ${GE_LIB_PATH})
    else()
        if(DEFINED ENV{ASCEND_CUSTOM_PATH})
            set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
        else()
            set(ASCEND_DIR /usr/local/Ascend)
        endif()

        set(ASCEND_ATC_DIR ${ASCEND_DIR}/atc/lib64)

        find_module(slog libslog.so ${ASCEND_ATC_DIR})
        find_module(static_mmpa libmmpa.a ${ASCEND_ATC_DIR})
        find_module(error_manager liberror_manager.so ${ASCEND_ATC_DIR})
    endif()

endif()

add_subdirectory(graph)
if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
    add_subdirectory(register)
endif ()

