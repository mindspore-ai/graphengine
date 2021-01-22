/**
* @file ExternalSoftDp.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef EXTERNALSOFTDP_H
#define EXTERNALSOFTDP_H

#include <stdint.h>

extern "C" {
struct SoftDpProcsessInfo {
    uint8_t* inputBuffer;
    uint32_t inputBufferSize;

    uint8_t* outputBuffer;
    uint32_t outputBufferSize;

    uint32_t outputWidth;
    uint32_t outputHeight;

    uint32_t reserved;
};

struct DpCropInfo {
    uint32_t left;
    uint32_t right;
    uint32_t up;
    uint32_t down;
};

/*
 * @brief decode and resize interface
 * @param [in] SoftDpProcsessInfo& softDpProcsessInfo : soft dp struct
 * @return success: return 0, fail: return error number
 */
uint32_t DecodeAndResizeJpeg(SoftDpProcsessInfo& softDpProcsessInfo);

/*
 * @brief decode crop and resize interface
 * @param [in] SoftDpProcsessInfo& softDpProcsessInfo : soft dp struct
 * @param [in] const DpCropInfo& cropInfo: crop struct
 * @return success: return 0, fail: return error number
 */
uint32_t DecodeAndCropAndResizeJpeg(SoftDpProcsessInfo& softDpProcsessInfo, const DpCropInfo& cropInfo);
}
#endif // EXTERNALSOFTDP_H