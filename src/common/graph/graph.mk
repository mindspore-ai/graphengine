LOCAL_PATH := $(call my-dir)
include $(LOCAL_PATH)/stub/Makefile
COMMON_LOCAL_SRC_FILES := \
    ./proto/om.proto \
    ./proto/ge_ir.proto \
    ./proto/ge_onnx.proto \
    ./proto/insert_op.proto \
    ./proto/task.proto \
    ./proto/fwk_adapter.proto \
    ./proto/op_mapping_info.proto \
    ./proto/dump_task.proto \
    ./anchor.cc \
    ./ge_attr_value.cc \
    ./attr_value.cc \
    ./buffer.cc \
    ./compute_graph.cc \
    ./graph.cc \
    ./inference_context.cc \
    ./shape_refiner.cc \
    ./format_refiner.cc \
    ./ref_relation.cc \
    ./model.cc \
    ./model_serialize.cc \
    ./node.cc \
    ./op_desc.cc \
    ./operator.cc \
    ./operator_factory.cc \
    ./operator_factory_impl.cc \
    ./ge_attr_define.cc \
    ./ge_tensor.cc \
    ./detail/attributes_holder.cc \
    ./utils/anchor_utils.cc \
    ./utils/tuning_utils.cc \
    ./utils/graph_utils.cc \
    ./utils/ge_ir_utils.cc \
    ./utils/op_desc_utils.cc \
    ./utils/type_utils.cc \
    ./utils/tensor_utils.cc \
    ./tensor.cc \
    ./debug/graph_debug.cc \
    ./opsproto/opsproto_manager.cc \
    ../ops/op_imp.cpp \
    option/ge_context.cc \
    option/ge_local_context.cc \
    ./runtime_inference_context.cc \
    ./utils/node_utils.cc \

COMMON_LOCAL_C_INCLUDES := \
    proto/om.proto \
    proto/ge_ir.proto \
    proto_inner/ge_onnx.proto \
    proto/insert_op.proto \
    proto/task.proto \
    proto/fwk_adapter.proto \
    proto/op_mapping_info.proto \
    proto/dump_task.proto \
    inc \
    inc/external \
    inc/external/graph \
    inc/graph \
    inc/common \
    common \
    common/graph \
    third_party/protobuf/include \
    libc_sec/include \
    ops/built-in/op_proto/inc \


#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libgraph

LOCAL_CFLAGS += -DFMK_SUPPORT_DUMP -O2
LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libprotobuf   \
    libslog       \
    liberror_manager \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_SHARED_LIBRARY)

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := stub/libgraph

LOCAL_CFLAGS += -DFMK_SUPPORT_DUMP -O2
LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := \
    ../../out/graph/lib64/stub/graph.cc \
    ../../out/graph/lib64/stub/operator.cc \
    ../../out/graph/lib64/stub/tensor.cc \
    ../../out/graph/lib64/stub/operator_factory.cc \


LOCAL_SHARED_LIBRARIES :=

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_SHARED_LIBRARY)

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := fwk_stub/libgraph

LOCAL_CFLAGS += -DFMK_SUPPORT_DUMP -O2
LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := \
    ../../out/graph/lib64/stub/attr_value.cc \
    ../../out/graph/lib64/stub/graph.cc \
    ../../out/graph/lib64/stub/operator.cc \
    ../../out/graph/lib64/stub/operator_factory.cc \
    ../../out/graph/lib64/stub/tensor.cc \
    ../../out/graph/lib64/stub/inference_context.cc \


LOCAL_SHARED_LIBRARIES :=

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_SHARED_LIBRARY)

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := libgraph

LOCAL_CFLAGS += -O2

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libprotobuf   \
    libslog       \
    liberror_manager \

LOCAL_LDFLAGS := -lrt -ldl

ifeq ($(device_os),android)
LOCAL_LDFLAGS := -ldl
endif

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_SHARED_LIBRARY)

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := stub/libgraph

LOCAL_CFLAGS += -O2

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := \
    ../../out/graph/lib64/stub/graph.cc \
    ../../out/graph/lib64/stub/operator.cc \
    ../../out/graph/lib64/stub/tensor.cc \
    ../../out/graph/lib64/stub/operator_factory.cc \


LOCAL_SHARED_LIBRARIES :=

LOCAL_LDFLAGS := -lrt -ldl

ifeq ($(device_os),android)
LOCAL_LDFLAGS := -ldl
endif

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_SHARED_LIBRARY)

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := fwk_stub/libgraph

LOCAL_CFLAGS += -O2

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := \
    ../../out/graph/lib64/stub/attr_value.cc \
    ../../out/graph/lib64/stub/graph.cc \
    ../../out/graph/lib64/stub/operator.cc \
    ../../out/graph/lib64/stub/operator_factory.cc \
    ../../out/graph/lib64/stub/tensor.cc \
    ../../out/graph/lib64/stub/inference_context.cc \


LOCAL_SHARED_LIBRARIES :=

LOCAL_LDFLAGS := -lrt -ldl

ifeq ($(device_os),android)
LOCAL_LDFLAGS := -ldl
endif

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_SHARED_LIBRARY)

# compile for ut/st
include $(CLEAR_VARS)
LOCAL_MODULE := libgraph

LOCAL_CFLAGS +=

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libprotobuf   \
    libslog       \
    liberror_manager \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_LLT_SHARED_LIBRARY)


#compiler for host static lib
include $(CLEAR_VARS)
LOCAL_MODULE := libgraph

LOCAL_CFLAGS += -DFMK_SUPPORT_DUMP -O2
LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_STATIC_LIBRARIES := \
    libprotobuf   \

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libslog       \
    liberror_manager \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_STATIC_LIBRARY)

#compiler for device static lib
include $(CLEAR_VARS)
LOCAL_MODULE := libgraph

LOCAL_CFLAGS += -O2

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_STATIC_LIBRARIES := \
    libprotobuf   \

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libslog       \
    liberror_manager \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_STATIC_LIBRARY)
