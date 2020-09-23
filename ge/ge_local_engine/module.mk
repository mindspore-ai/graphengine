LOCAL_PATH := $(call my-dir)


local_lib_src_files :=  engine/ge_local_engine.cc \
                        ops_kernel_store/ge_local_ops_kernel_info.cc \
                        ops_kernel_store/op/op_factory.cc \
                        ops_kernel_store/op/op.cc \
                        ops_kernel_store/op/ge_deleted_op.cc \
                        ops_kernel_store/op/no_op.cc \

local_lib_inc_path :=   proto/task.proto \
                        ${LOCAL_PATH} \
                        ${TOPDIR}inc \
                        ${TOPDIR}inc/external \
                        ${TOPDIR}inc/external/graph \
                        $(TOPDIR)libc_sec/include \
                        ${TOPDIR}third_party/protobuf/include \
                        ${TOPDIR}inc/framework \
                        $(TOPDIR)framework/domi \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libge_local_engine
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS := 

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=   libprotobuf \
                            libc_sec \
                            libslog \
                            libgraph \
                            libregister \
                            libruntime

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

#compiler for atc
include $(CLEAR_VARS)
LOCAL_MODULE := atclib/libge_local_engine
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=   libprotobuf \
                            libc_sec \
                            libslog \
                            libgraph \
                            libregister \
                            libruntime_compile

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}
