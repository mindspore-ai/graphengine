
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := atc

LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0  -O2

LOCAL_SRC_FILES := \
    main.cc \
    single_op_parser.cc \
    ../session/omg.cc \
    ../ir_build/atc_ir_common.cc \

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/../ ./ \
    $(TOPDIR)inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)inc/external/graph \
    $(TOPDIR)inc/framework \
    $(TOPDIR)inc/framework/domi \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)inc/common/util \
    third_party/json/include \
    third_party/gflags/include \
    third_party/protobuf/include \
    proto/om.proto \
    proto/ge_ir.proto \
    proto/task.proto \
    proto/insert_op.proto \

LOCAL_SHARED_LIBRARIES := \
    libc_sec \
    libge_common \
    libprotobuf \
    libslog \
    libgraph \
    libregister \
    liberror_manager \
    libge_compiler \
    libruntime_compile \
    libparser_common \
    libfmk_tensorflow_parser \
    liberror_manager \

LOCAL_STATIC_LIBRARIES := libgflags

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_EXECUTABLE)

