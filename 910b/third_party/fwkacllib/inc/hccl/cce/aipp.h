/**
 * @file cce/aipp.h
 *
 * Copyright(C), 2017 - 2017, Huawei Tech. Co., Ltd. ALL RIGHTS RESERVED.
 *
 * @brief library api header file.
 *
 * @version 1.0
 *
 */
#ifndef __AIPP_H__
#define __AIPP_H__

#include <stdint.h>


/**
* @ingroup dnn
* @brief struct define of dynamic aipp batch parameter.
*/
typedef struct tagAippDynamicBatchPara
{
    int8_t cropSwitch;              //crop switch
    int8_t scfSwitch;               //resize switch
    int8_t paddingSwitch;           //0: unable padding, 1: padding config value,sfr_filling_hblank_ch0 ~  sfr_filling_hblank_ch2
                                    //2: padding source picture data, single row/collumn copy
                                    //3: padding source picture data, block copy
                                    //4: padding source picture data, mirror copy
    int8_t rotateSwitch;            //rotate switch，0: non-ratate，1: ratate 90° clockwise，2: ratate 180° clockwise，3: ratate 270° clockwise
    int8_t reserve[4];

    int32_t cropStartPosW;          //the start horizontal position of cropping
    int32_t cropStartPosH;          //the start vertical position of cropping
    int32_t cropSizeW;              //crop width
    int32_t cropSizeH;              //crop height

    int32_t scfInputSizeW;          //input width of scf
    int32_t scfInputSizeH;          //input height of scf
    int32_t scfOutputSizeW;         //output width of scf
    int32_t scfOutputSizeH;         //output height of scf

    int32_t paddingSizeTop;         //top padding size
    int32_t paddingSizeBottom;      //bottom padding size
    int32_t paddingSizeLeft;        //left padding size
    int32_t paddingSizeRight;       //right padding size

    int16_t dtcPixelMeanChn0;       //mean value of channel 0
    int16_t dtcPixelMeanChn1;       //mean value of channel 1
    int16_t dtcPixelMeanChn2;       //mean value of channel 2
    int16_t dtcPixelMeanChn3;       //mean value of channel 3
#ifndef DAVINCI_TINY
    uint16_t dtcPixelMinChn0;       //min value of channel 0
    uint16_t dtcPixelMinChn1;       //min value of channel 1
    uint16_t dtcPixelMinChn2;       //min value of channel 2
    uint16_t dtcPixelMinChn3;       //min value of channel 3
    uint16_t dtcPixelVarReciChn0;   //sfr_dtc_pixel_variance_reci_ch0
    uint16_t dtcPixelVarReciChn1;   //sfr_dtc_pixel_variance_reci_ch1
    uint16_t dtcPixelVarReciChn2;   //sfr_dtc_pixel_variance_reci_ch2
    uint16_t dtcPixelVarReciChn3;   //sfr_dtc_pixel_variance_reci_ch3

    int8_t reserve1[16];            //32B assign, for ub copy
#endif
}kAippDynamicBatchPara;

/**
* @ingroup dnn
* @brief struct define of dynamic aipp parameter. lite:64+96*batchNum byte ; tiny:64+64*batchNum byte
*/
typedef struct tagAippDynamicPara
{
    uint8_t inputFormat;            //input format：YUV420SP_U8/XRGB8888_U8/RGB888_U8
    //uint8_t outDataType;          //output data type: CC_DATA_HALF,CC_DATA_INT8, CC_DATA_UINT8
    int8_t cscSwitch;               //csc switch
    int8_t rbuvSwapSwitch;          //rb/ub swap switch
    int8_t axSwapSwitch;            //RGBA->ARGB, YUVA->AYUV swap switch
    int8_t batchNum;                //batch parameter number
    int8_t reserve1[3];

    int32_t srcImageSizeW;          //source image width
    int32_t srcImageSizeH;          //source image height

    int16_t cscMatrixR0C0;          //csc_matrix_r0_c0
    int16_t cscMatrixR0C1;          //csc_matrix_r0_c1
    int16_t cscMatrixR0C2;          //csc_matrix_r0_c2
    int16_t cscMatrixR1C0;          //csc_matrix_r1_c0
    int16_t cscMatrixR1C1;          //csc_matrix_r1_c1
    int16_t cscMatrixR1C2;          //csc_matrix_r1_c2
    int16_t cscMatrixR2C0;          //csc_matrix_r2_c0
    int16_t cscMatrixR2C1;          //csc_matrix_r2_c1
    int16_t cscMatrixR2C2;          //csc_matrix_r2_c2
    int16_t reserve2[3];
    uint8_t cscOutputBiasR0;        //output Bias for RGB to YUV, element of row 0, unsigned number
    uint8_t cscOutputBiasR1;        //output Bias for RGB to YUV, element of row 1, unsigned number
    uint8_t cscOutputBiasR2;        //output Bias for RGB to YUV, element of row 2, unsigned number
    uint8_t cscInputBiasR0;         //input Bias for YUV to RGB, element of row 0, unsigned number
    uint8_t cscInputBiasR1;         //input Bias for YUV to RGB, element of row 1, unsigned number
    uint8_t cscInputBiasR2;         //input Bias for YUV to RGB, element of row 2, unsigned number
    uint8_t reserve3[2];

    int8_t reserve4[16];            //32B assign, for ub copy

    kAippDynamicBatchPara aippBatchPara;  //allow transfer several batch para.
} kAippDynamicPara;

#endif /* __AIPP_H__ */
