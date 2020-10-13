LOCAL_PATH := $(call my-dir)

local_ge_executor_src_files :=  \
    ge_executor.cc \
    ../common/profiling/profiling_manager.cc \
    ../common/dump/dump_properties.cc \
    ../common/dump/dump_manager.cc \
    ../common/dump/dump_op.cc \
    ../common/ge/plugin_manager.cc \
    ../common/ge/op_tiling_manager.cc \
    ../graph/load/graph_loader.cc \
    ../graph/execute/graph_execute.cc \
    ../omm/csa_interact.cc \
    ../graph/manager/graph_manager_utils.cc \
    ../graph/manager/graph_var_manager.cc \
    ../graph/manager/rdma_pool_allocator.cc \
    ../graph/manager/graph_mem_allocator.cc \
    ../graph/manager/graph_caching_allocator.cc \
    ../graph/manager/trans_var_data_utils.cc \
    ../graph/manager/util/debug.cc \
    ../model/ge_model.cc \
    ../model/ge_root_model.cc \
    ../graph/load/new_model_manager/davinci_model.cc \
    ../graph/load/new_model_manager/davinci_model_parser.cc \
    ../graph/load/new_model_manager/model_manager.cc \
    ../graph/load/new_model_manager/tbe_handle_store.cc \
    ../graph/load/new_model_manager/cpu_queue_schedule.cc \
    ../graph/load/new_model_manager/model_utils.cc \
    ../graph/load/new_model_manager/aipp_utils.cc \
    ../graph/load/new_model_manager/data_inputer.cc \
    ../graph/load/new_model_manager/data_dumper.cc \
    ../graph/load/new_model_manager/zero_copy_task.cc \
    ../graph/load/new_model_manager/zero_copy_offset.cc \
    ../graph/load/new_model_manager/task_info/task_info.cc                  \
    ../graph/load/new_model_manager/task_info/event_record_task_info.cc     \
    ../graph/load/new_model_manager/task_info/event_wait_task_info.cc       \
    ../graph/load/new_model_manager/task_info/fusion_start_task_info.cc     \
    ../graph/load/new_model_manager/task_info/fusion_stop_task_info.cc      \
    ../graph/load/new_model_manager/task_info/kernel_ex_task_info.cc        \
    ../graph/load/new_model_manager/task_info/kernel_task_info.cc           \
    ../graph/load/new_model_manager/task_info/label_set_task_info.cc        \
    ../graph/load/new_model_manager/task_info/label_switch_by_index_task_info.cc \
    ../graph/load/new_model_manager/task_info/label_goto_ex_task_info.cc    \
    ../graph/load/new_model_manager/task_info/memcpy_async_task_info.cc     \
    ../graph/load/new_model_manager/task_info/memcpy_addr_async_task_info.cc \
    ../graph/load/new_model_manager/task_info/profiler_trace_task_info.cc   \
    ../graph/load/new_model_manager/task_info/stream_active_task_info.cc    \
    ../graph/load/new_model_manager/task_info/stream_switch_task_info.cc    \
    ../graph/load/new_model_manager/task_info/stream_switchn_task_info.cc   \
    ../graph/load/new_model_manager/task_info/end_graph_task_info.cc        \
    ../graph/load/new_model_manager/task_info/super_kernel/super_kernel_factory.cc   \
    ../graph/load/new_model_manager/task_info/super_kernel/super_kernel.cc  \
    ../opskernel_manager/ops_kernel_builder_manager.cc \
    ../single_op/single_op_manager.cc \
    ../single_op/single_op_model.cc \
    ../single_op/single_op.cc \
    ../single_op/stream_resource.cc \
    ../single_op/task/op_task.cc \
    ../single_op/task/build_task_utils.cc \
    ../single_op/task/tbe_task_builder.cc \
    ../single_op/task/aicpu_task_builder.cc \
    ../single_op/task/aicpu_kernel_task_builder.cc \
    ../hybrid/hybrid_davinci_model_stub.cc\
    ../hybrid/node_executor/aicpu/aicpu_ext_info.cc \

local_ge_executor_c_include :=             \
    proto/insert_op.proto                  \
    proto/op_mapping_info.proto            \
    proto/dump_task.proto                  \
    proto/ge_ir.proto                      \
    proto/task.proto                       \
    proto/om.proto                         \
    $(TOPDIR)inc/external                  \
    $(TOPDIR)inc/external/graph            \
    $(TOPDIR)inc/framework                 \
    $(TOPDIR)inc                           \
    $(LOCAL_PATH)/../                      \
    $(TOPDIR)libc_sec/include              \
    third_party/protobuf/include           \
    third_party/json/include               \

local_ge_executor_shared_library :=        \
    libprotobuf                            \
    libc_sec                               \
    libge_common                           \
    libruntime                             \
    libslog                                \
    libmmpa                                \
    libgraph                               \
    libregister                            \
    libmsprof                              \
    liberror_manager                       \

local_ge_executor_ldflags := -lrt -ldl     \


#compile arm  device dynamic lib
include $(CLEAR_VARS)

LOCAL_MODULE := libge_executor
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -O2 -DDAVINCI_SUPPORT_PROFILING

LOCAL_SRC_FILES := $(local_ge_executor_src_files)
LOCAL_C_INCLUDES := $(local_ge_executor_c_include)

LOCAL_SHARED_LIBRARIES := $(local_ge_executor_shared_library)

LOCAL_SHARED_LIBRARIES += libascend_hal

LOCAL_STATIC_LIBRARIES := \
    libmsprofiler \

ifeq ($(device_os),android)
LOCAL_LDFLAGS += -ldl
LOCAL_LDLIBS += -L$(PWD)/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29 -llog
else
LOCAL_LDFLAGS += $(local_ge_executor_ldflags)
endif

include $(BUILD_SHARED_LIBRARY)

#compile x86 host dynamic lib
include $(CLEAR_VARS)

LOCAL_MODULE := libge_executor
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DDAVINCI_SUPPORT_PROFILING
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
else
LOCAL_CFLAGS += -O2
endif

LOCAL_SRC_FILES := $(local_ge_executor_src_files)

LOCAL_C_INCLUDES := $(local_ge_executor_c_include)

LOCAL_SHARED_LIBRARIES :=                  \
    libprotobuf                            \
    libc_sec                               \
    libge_common                           \
    libruntime                             \
    libslog                                \
    libmmpa                                \
    libgraph                               \
    libregister                            \
    libmsprof                              \
    liberror_manager                       \
    stub/libascend_hal                     \

LOCAL_STATIC_LIBRARIES := \
    libmsprofiler \

LOCAL_LDFLAGS += $(local_ge_executor_ldflags)

include $(BUILD_HOST_SHARED_LIBRARY)

#compile for host static lib
include $(CLEAR_VARS)

LOCAL_MODULE := libge_executor
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DDAVINCI_SUPPORT_PROFILING
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
else
LOCAL_CFLAGS += -O2
endif

LOCAL_SRC_FILES := $(local_ge_executor_src_files)

LOCAL_C_INCLUDES := $(local_ge_executor_c_include)

LOCAL_STATIC_LIBRARIES := \
    libge_common \
    libgraph     \
    libregister  \
    libprotobuf  \

LOCAL_SHARED_LIBRARIES :=                  \
    libc_sec                               \
    libruntime                             \
    libslog                                \
    libmmpa                                \
    libmsprof                              \

LOCAL_LDFLAGS += $(local_ge_executor_ldflags)

include $(BUILD_HOST_STATIC_LIBRARY)

#compile for device static lib
include $(CLEAR_VARS)

LOCAL_MODULE := libge_executor
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DDAVINCI_SUPPORT_PROFILING
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
else
LOCAL_CFLAGS += -O2
endif

LOCAL_SRC_FILES := $(local_ge_executor_src_files)
LOCAL_C_INCLUDES := $(local_ge_executor_c_include)

LOCAL_STATIC_LIBRARIES := \
    libge_common \
    libgraph     \
    libregister  \
    libprotobuf  \

LOCAL_SHARED_LIBRARIES :=                  \
    libc_sec                               \
    libruntime                             \
    libslog                                \
    libmmpa                                \
    libmsprof                              \

ifeq ($(device_os),android)
LOCAL_LDFLAGS += -ldl
LOCAL_LDLIBS += -L$(PWD)/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29 -llog
else
LOCAL_LDFLAGS += $(local_ge_executor_ldflags)
endif

include $(BUILD_STATIC_LIBRARY)
