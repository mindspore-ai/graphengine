LOCAL_PATH := $(call my-dir)


local_lib_src_files :=  memory_assigner.cc \
                        graph_mem_assigner.cc \
                        binary_block_mem_assigner.cc \
                        block_mem_assigner.cc \
                        hybrid_mem_assigner.cc \
                        max_block_mem_assigner.cc \
                        var_mem_assign_util.cc \

local_lib_inc_path :=   ${LOCAL_PATH} \
                        ${TOPDIR}inc \
                        ${TOPDIR}inc/external \
                        ${TOPDIR}inc/external/graph \
                        $(TOPDIR)libc_sec/include \
                        ${TOPDIR}third_party/protobuf/include \
                        ${TOPDIR}inc/framework \
                        $(TOPDIR)framework/domi \
                        $(TOPDIR)graphengine/ge \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libge_memory

LOCAL_CFLAGS += -std=c++11
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=   libascend_protobuf \
                            libc_sec \
                            libslog \
                            libgraph \
                            libge_common \

LOCAL_SRC_FILES := $(local_lib_src_files)

generated_sources_dir := $(call local-generated-sources-dir)
LOCAL_EXPORT_C_INCLUDE_DIRS := $(generated_sources_dir)/proto/$(LOCAL_PATH)
LOCAL_C_INCLUDES := $(local_lib_inc_path)
LOCAL_C_INCLUDES += LOCAL_EXPORT_C_INCLUDE_DIRS

include ${BUILD_HOST_STATIC_LIBRARY}


#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := libge_memory

LOCAL_CFLAGS += -std=c++11
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DGOOGLE_PROTOBUF_NO_RTTI -DDEV_VISIBILITY
LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=   libascend_protobuf \
                            libc_sec \
                            libslog \
                            libgraph \
                            libge_common \

LOCAL_SRC_FILES := $(local_lib_src_files)

generated_sources_dir := $(call local-generated-sources-dir)
LOCAL_EXPORT_C_INCLUDE_DIRS := $(generated_sources_dir)/proto/$(LOCAL_PATH)
LOCAL_C_INCLUDES := $(local_lib_inc_path)
LOCAL_C_INCLUDES += LOCAL_EXPORT_C_INCLUDE_DIRS

include ${BUILD_STATIC_LIBRARY}

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := libge_memory

LOCAL_CFLAGS += -std=c++11 -Dgoogle=ascend_private
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=   libascend_protobuf \
                            libc_sec \
                            libslog \
                            libgraph \
                            libge_common \

LOCAL_SRC_FILES := $(local_lib_src_files)

generated_sources_dir := $(call local-generated-sources-dir)
LOCAL_EXPORT_C_INCLUDE_DIRS := $(generated_sources_dir)/proto/$(LOCAL_PATH)
LOCAL_C_INCLUDES := $(local_lib_inc_path)
LOCAL_C_INCLUDES += LOCAL_EXPORT_C_INCLUDE_DIRS

include ${BUILD_LLT_STATIC_LIBRARY}
