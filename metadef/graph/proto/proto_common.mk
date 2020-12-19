LOCAL_PATH := $(call my-dir)

COMMON_LOCAL_SRC_FILES := \
    om.proto \
    ge_ir.proto\
    ge_onnx.proto\
    insert_op.proto     \
    task.proto          \
    fwk_adapter.proto    \
    op_mapping_info.proto \

COMMON_LOCAL_C_INCLUDES := \
    inc \
    inc/external \
    inc/external/graph \
    inc/common \
    inc/graph \
    common \
    common/graph \
    third_party/protobuf/include \
    libc_sec/include \
    ops/built-in/op_proto/inc \
    cann/ops/built-in/op_proto/inc \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libproto_common

LOCAL_CFLAGS += -DFMK_SUPPORT_DUMP -Dgoogle=ascend_private

LOCAL_CPPFLAGS += -fexceptions
LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libascend_protobuf   \
    libslog       \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_STATIC_LIBRARY)

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := libproto_common

LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libascend_protobuf   \
    libslog       \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_STATIC_LIBRARY)

# compile for ut/st
include $(CLEAR_VARS)
LOCAL_MODULE := libproto_common

LOCAL_CFLAGS += -Werror -Wno-unused-variable -Dgoogle=ascend_private
LOCAL_CFLAGS += -DDAVINCI_MINI

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libascend_protobuf   \
    libslog       \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_LLT_STATIC_LIBRARY)
