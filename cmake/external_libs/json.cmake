if (HAVE_JSON)
    return()
endif()

include(ExternalProject)

set(JSON_SRC_DIR ${CMAKE_BINARY_DIR}/opensrc/json/include)
ExternalProject_Add(json_build
                    URL https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip
                    #URL /home/txd/workspace/cloud_code/pkg/include.zip
                    SOURCE_DIR  ${JSON_SRC_DIR}
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    EXCLUDE_FROM_ALL TRUE 
)


add_library(json INTERFACE)
target_include_directories(json INTERFACE ${JSON_SRC_DIR})
add_dependencies(json json_build)

#set(HAVE_JSON TRUE CACHE BOOL "json build add")
set(HAVE_JSON TRUE)
