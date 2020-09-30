include(ExternalProject)

#set(ONNX_SRC_DIR /home/txd/workspace/cloud_code/graphengine/build/graphengine/open_source/onnx)
#set(ONNX_PROTO ${ONNX_SRC_DIR}/onnx/onnx.proto)
set(ONNX_PROTO_DIR ${CMAKE_BINARY_DIR}/onnx)
set(ONNX_PROTO_FILE ${ONNX_PROTO_DIR}/onnx.proto)
file(MAKE_DIRECTORY ${ONNX_PROTO_DIR})

ExternalProject_Add(onnx
                    #URL https://github.com/onnx/onnx/releases/download/v1.6.0/onnx-1.6.0.tar.gz
                    URL /home/txd/workspace/cloud_code/pkg/onnx-1.6.0.tar.gz
                    #URL_HASH SHA256=3b88c3fe521151651a0403c4d131cb2e0311bd28b753ef692020a432a81ce345
                    #SOURCE_DIR ${ONNX_SRC_DIR}
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    #INSTALL_COMMAND "" 
                    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/onnx/onnx.proto ${ONNX_PROTO_FILE}
                    #BUILD_ALWAYS TRUE
                    EXCLUDE_FROM_ALL TRUE
)

macro(onnx_protobuf_generate comp c_var h_var)
    add_custom_command(OUTPUT ${ONNX_PROTO_FILE}
        DEPENDS onnx
    )
    ge_protobuf_generate(${comp} ${c_var} ${h_var} ${ONNX_PROTO_FILE})
endmacro()


