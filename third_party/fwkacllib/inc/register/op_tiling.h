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

#ifndef INC_OP_TILING_H_
#define INC_OP_TILING_H_

#include "external/register/register_types.h"
#include "external/graph/tensor.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"

#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <functional>
#include <vector>
#include <map>
#include <string>
#include "graph/node.h"

#define REGISTER_OP_TILING_FUNC(optype, opfunc)                     \
    REGISTER_OP_TILING_FUNC_UNIQ_HELPER(optype, opfunc, __COUNTER__)

#define REGISTER_OP_TILING_FUNC_UNIQ_HELPER(optype, opfunc, counter)    \
    REGISTER_OP_TILING_FUNC_UNIQ(optype, opfunc, counter)

#define REGISTER_OP_TILING_FUNC_UNIQ(optype, opfunc, counter)    \
    static OpTilingInterf g_##optype##TilingInterf##counter(#optype, opfunc)

namespace optiling {

enum TensorArgType {
    TA_NONE,
    TA_SINGLE,
    TA_LIST,
};


using ByteBuffer = std::stringstream;

struct TeOpTensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> ori_shape;
    std::string format;
    std::string ori_format;
    std::string dtype;
    std::map<std::string, std::string> attrs;
};


struct TeOpTensorArg {
    TensorArgType arg_type;
    std::vector<TeOpTensor> tensor;
};

struct OpRunInfo {
    uint32_t block_dim;
    std::vector<int64_t> workspaces;
    ByteBuffer tiling_data;
    bool clear_atomic;
};


using TeOpAttrArgs = std::vector<std::string>;
using TeConstTensorData = std::tuple<const uint8_t*, size_t, ge::Tensor>;

struct TeOpParas {
    std::vector<TeOpTensorArg> inputs;
    std::vector<TeOpTensorArg> outputs;
    std::map<std::string, TeConstTensorData> const_inputs;
    TeOpAttrArgs attrs;
};


using OpTilingFunc = std::function<bool(const std::string&, const TeOpParas&,
                                        const nlohmann::json& , OpRunInfo&)>;

using OpTilingFuncPtr = bool(*)(const std::string&, const TeOpParas&, const nlohmann::json& , OpRunInfo&);

class FMK_FUNC_HOST_VISIBILITY OpTilingInterf
{
public:
    OpTilingInterf(std::string op_type, OpTilingFunc func);
    ~OpTilingInterf() = default;
    static std::map<std::string, OpTilingFunc> &RegisteredOpInterf();
    static std::string OpTilingUuid;
};


template <class T>
ByteBuffer& ByteBufferPut(ByteBuffer &buf, const T &value)
{
    buf.write(reinterpret_cast<const char*>(&value), sizeof(value));
    buf.flush();
    return buf;
}

template <class T>
ByteBuffer& ByteBufferGet(ByteBuffer &buf, T &value)
{
    buf.read(reinterpret_cast<char*>(&value), sizeof(value));
    return buf;
}

inline size_t ByteBufferGetAll(ByteBuffer &buf, char *dest, size_t dest_len)
{
    size_t nread = 0;
    size_t rn = 0;
    do {
        rn = buf.readsome(dest + nread, dest_len - nread);
        nread += rn;
    } while (rn > 0 && dest_len > nread);

    return nread;
}


extern "C" ge::graphStatus OpParaCalculate(const ge::Node &node, OpRunInfo &run_info);
extern "C" ge::graphStatus OpAtomicCalculate(const ge::Node &node, OpRunInfo &run_info);

}

#endif // INC_OP_TILING_H_
