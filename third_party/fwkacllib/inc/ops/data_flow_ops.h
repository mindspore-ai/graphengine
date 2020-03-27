/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GE_OP_DATA_FLOW_OPS_H_
#define GE_OP_DATA_FLOW_OPS_H_

#include <algorithm>
#include "graph/operator_reg.h"

namespace ge {

REG_OP(QueueIsClosed)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(is_closed, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(QueueIsClosed)

REG_OP(QueueSize)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(size, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(QueueSize)

REG_OP(FIFOQueue)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(component_types, ListType)
    .ATTR(shapes, ListListInt, {})
    .ATTR(capacity, Int, -1)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(FIFOQueue)

REG_OP(QueueEnqueue)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .DYNAMIC_INPUT(components, TensorType({DT_INT8, DT_UINT8, \
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_UINT32, \
        DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
    .ATTR(timeout_ms, Int, -1)
    .OP_END_FACTORY_REG(QueueEnqueue)

REG_OP(QueueEnqueueMany)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .DYNAMIC_INPUT(components, TensorType({DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, \
        DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
    .ATTR(timeout_ms, Int, -1)
    .OP_END_FACTORY_REG(QueueEnqueueMany)

REG_OP(QueueDequeue)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .DYNAMIC_OUTPUT(components, TensorType({DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, \
        DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
    .ATTR(timeout_ms, Int, -1)
    .REQUIRED_ATTR(component_types, ListType)
    .OP_END_FACTORY_REG(QueueDequeue)

REG_OP(QueueDequeueMany)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(n, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(components, TensorType({DT_INT8, DT_UINT8, \
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, \
        DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
    .ATTR(timeout_ms, Int, -1)
    .REQUIRED_ATTR(component_types, ListType)
    .OP_END_FACTORY_REG(QueueDequeueMany)

REG_OP(QueueDequeueUpTo)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(n, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(components, TensorType({DT_INT8, DT_UINT8, \
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_UINT32, \
        DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
    .ATTR(timeout_ms, Int, -1)
    .REQUIRED_ATTR(component_types, ListType)
    .OP_END_FACTORY_REG(QueueDequeueUpTo)

REG_OP(Stage)
    .DYNAMIC_INPUT(values, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
        DT_DOUBLE, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(Stage)

REG_OP(StageClear)
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(dtypes, ListType, {})
    .OP_END_FACTORY_REG(StageClear)

REG_OP(StagePeek)
    .INPUT(index, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
                     DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
                     DT_DOUBLE, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(dtypes, ListType, {})
    .OP_END_FACTORY_REG(StagePeek)

REG_OP(StageSize)
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(dtypes, ListType, {})
    .OP_END_FACTORY_REG(StageSize)

REG_OP(StackPop)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(element, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
                     DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
                     DT_DOUBLE, DT_UINT32, DT_UNIT64}))
    .REQUIRED_ATTR(elem_type, Type)
    .OP_END_FACTORY_REG(StackPop)

REG_OP(StackPush)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(element, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
                     DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
                     DT_DOUBLE, DT_UINT32, DT_UNIT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
                     DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
                     DT_DOUBLE, DT_UINT32, DT_UNIT64}))
    .ATTR(swap_memory, Bool, false)
    .OP_END_FACTORY_REG(StackPush)

REG_OP(StackClose)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(StackClose)

REG_OP(Stack)
    .INPUT(max_size, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(stack_name, String, "")
    .REQUIRED_ATTR(elem_type, Type)
    .OP_END_FACTORY_REG(Stack)

REG_OP(DynamicPartition)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(partitions, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(num_partitions, Int, 1)
    .OP_END_FACTORY_REG(DynamicPartition)

REG_OP(DynamicStitch)
    .DYNAMIC_INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
        DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(DynamicStitch)

REG_OP(ParallelDynamicStitch)
    .DYNAMIC_INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
        DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ParallelDynamicStitch)

REG_OP(MapClear)
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapClear)

REG_OP(MapIncompleteSize)
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapIncompleteSize)

REG_OP(Unstage)
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
            DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
            DT_DOUBLE, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .REQUIRED_ATTR(dtypes, ListType)
    .OP_END_FACTORY_REG(Unstage)

REG_OP(MapStage)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapStage)

REG_OP(MapUnstage)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapUnstage)

REG_OP(MapUnstageNoKey)
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(key, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapUnstageNoKey)

REG_OP(MapPeek)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapPeek)

REG_OP(MapSize)
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapSize)

REG_OP(TensorArray)
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(flow, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(element_shape, ListInt, ge::UNKNOWN_SHAPE)
    .ATTR(dynamic_size, Bool, false)
    .ATTR(clear_after_read, Bool, true)
    .ATTR(identical_element_shapes, Bool, false)
    .ATTR(tensor_array_name, String, "")
    .OP_END_FACTORY_REG(TensorArray)

REG_OP(TensorArrayClose)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(TensorArrayClose)

REG_OP(TensorArrayConcat)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(flow_in, TensorType({DT_FLOAT}))
    .OUTPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL}))
    .OUTPUT(lengths, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(element_shape_except0, ListInt, ge::UNKNOWN_SHAPE)
    .OP_END_FACTORY_REG(TensorArrayConcat)

REG_OP(TensorArrayGather)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(flow_in, TensorType({DT_FLOAT}))
    .OUTPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(element_shape, ListInt, ge::UNKNOWN_SHAPE)
    .OP_END_FACTORY_REG(TensorArrayGather)

REG_OP(TensorArrayGrad)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(flow_in, TensorType({DT_FLOAT}))
    .OUTPUT(grad_handle, TensorType({DT_RESOURCE}))
    .OUTPUT(flow_out, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(source, String)
    .OP_END_FACTORY_REG(TensorArrayGrad)

REG_OP(TensorArrayWrite)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL}))
    .INPUT(flow_in, TensorType({DT_FLOAT}))
    .OUTPUT(flow_out, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(TensorArrayWrite)

REG_OP(TensorArrayGradWithShape)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .INPUT(shape_to_prepend, TensorType({ DT_INT32 }))
    .OUTPUT(grad_handle, TensorType({ DT_RESOURCE }))
    .OUTPUT(flow_out, TensorType({ DT_FLOAT }))
    .ATTR(source, String, "")
    .OP_END_FACTORY_REG(TensorArrayGradWithShape)

REG_OP(TensorArrayRead)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(index, TensorType({ DT_INT32 }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE }))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(TensorArrayRead)

REG_OP(TensorArrayScatter)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(indices, TensorType({ DT_INT32 }))
    .INPUT(value, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .OUTPUT(flow_out, TensorType({ DT_FLOAT }))
    .OP_END_FACTORY_REG(TensorArrayScatter)

REG_OP(TensorArraySplit)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(value, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE }))
    .INPUT(lengths, TensorType({ DT_INT64 }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .OUTPUT(flow_out, TensorType({ DT_FLOAT }))
    .OP_END_FACTORY_REG(TensorArraySplit)

REG_OP(TensorArraySize)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .OUTPUT(size, TensorType({ DT_INT32 }))
    .OP_END_FACTORY_REG(TensorArraySize)

REG_OP(RandomShuffleQueue)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(component_types, ListType)
    .ATTR(shapes, ListListInt, {})
    .ATTR(capacity, Int, -1)
    .ATTR(min_after_dequeue, Int, 0)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(RandomShuffleQueue)

REG_OP(PaddingFIFOQueue)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(component_types, ListType)
    .ATTR(shapes, ListListInt, {})
    .ATTR(capacity, Int, -1)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(PaddingFIFOQueue)

REG_OP(PriorityQueue)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(component_types, ListType, {})
    .ATTR(shapes, ListListInt, {})
    .ATTR(capacity, Int, -1)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(PriorityQueue)

REG_OP(QueueClose)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(cancel_pending_enqueues, Bool, false)
    .OP_END_FACTORY_REG(QueueClose)

REG_OP(OrderedMapStage)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                                      DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16,
                                      DT_DOUBLE, DT_BOOL, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapStage)

REG_OP(OrderedMapSize)
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapSize)

REG_OP(OrderedMapClear)
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapClear)

REG_OP(OrderedMapIncompleteSize)
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapIncompleteSize)

REG_OP(OrderedMapPeek)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                                        DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16,
                                        DT_DOUBLE, DT_BOOL, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapPeek)

REG_OP(OrderedMapUnstageNoKey)
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(key, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                                        DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16,
                                        DT_DOUBLE, DT_BOOL, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapUnstageNoKey)

REG_OP(OrderedMapUnstage)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                                        DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16,
                                        DT_DOUBLE, DT_BOOL, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapUnstage)

REG_OP(Barrier)
    .OUTPUT(handle, TensorType({DT_STRING_REF}))
    .REQUIRED_ATTR(component_types, ListType)
    .ATTR(shapes, ListListInt, {})
    .ATTR(capacity, Int, -1)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(Barrier)

REG_OP(BarrierInsertMany)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(keys, TensorType({DT_STRING}))
    .INPUT(values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_UINT32, DT_UINT64}))
    .REQUIRED_ATTR(component_index, Int)
    .OP_END_FACTORY_REG(BarrierInsertMany)

REG_OP(BarrierTakeMany)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(num_elements, TensorType(DT_INT32))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(keys, TensorType({DT_STRING}))
    .DYNAMIC_OUTPUT(values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_UINT32, DT_UINT64}))
    .REQUIRED_ATTR(component_types, ListType)
    .ATTR(allow_small_batch, Bool, false)
    .ATTR(wait_for_incomplete, Bool, false)
    .ATTR(timeout_ms, Int, -1)
    .OP_END_FACTORY_REG(BarrierTakeMany)

REG_OP(BarrierClose)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .ATTR(cancel_pending_enqueues, Bool, false)
    .OP_END_FACTORY_REG(BarrierClose)

REG_OP(BarrierReadySize)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .OUTPUT(size, TensorType(DT_INT32))
    .OP_END_FACTORY_REG(BarrierReadySize)

REG_OP(BarrierIncompleteSize)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .OUTPUT(size, TensorType(DT_INT32))
    .OP_END_FACTORY_REG(BarrierIncompleteSize)

REG_OP(RecordInput)
    .OUTPUT(records, TensorType({DT_STRING}))
    .REQUIRED_ATTR(file_pattern, String)
    .ATTR(file_random_seed, Int, 301)
    .ATTR(file_shuffle_shift_ratio, Float, 0)
    .ATTR(file_buffer_size, Int, 10000)
    .ATTR(file_parallelism, Int, 16)
    .ATTR(batch_size, Int, 32)
    .ATTR(compression_type, String, "")
    .OP_END_FACTORY_REG(RecordInput)

REG_OP(ConditionalAccumulator)
    .OUTPUT(handle, TensorType({DT_STRING_REF}))
    .REQUIRED_ATTR(dtype, Type)
    .REQUIRED_ATTR(shape, ListInt)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(reduction_type, String, "MEAN")
    .OP_END_FACTORY_REG(ConditionalAccumulator)

REG_OP(AccumulatorApplyGradient)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(local_step, TensorType({DT_INT64}))
    .INPUT(gradient, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AccumulatorApplyGradient)

REG_OP(AccumulatorNumAccumulated)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(AccumulatorNumAccumulated)

REG_OP(AccumulatorSetGlobalStep)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(new_global_step, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(AccumulatorSetGlobalStep)

REG_OP(AccumulatorTakeGradient)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(num_required, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
        DT_INT64, DT_DOUBLE, DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AccumulatorTakeGradient)

REG_OP(SparseConditionalAccumulator)
    .OUTPUT(handle, TensorType({DT_STRING_REF}))
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(reduction_type, String, "MEAN")
    .OP_END_FACTORY_REG(SparseConditionalAccumulator)

REG_OP(SparseAccumulatorApplyGradient)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(local_step, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT}))
    .INPUT(shape, TensorType({DT_INT64}))
    .REQUIRED_ATTR(has_known_shape, Bool)
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(SparseAccumulatorApplyGradient)

REG_OP(SparseAccumulatorTakeGradient)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(num_required, TensorType({DT_INT32}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(SparseAccumulatorTakeGradient)
}  // namespace ge

#endif  // GE_OP_DATA_FLOW_OPS_H_
