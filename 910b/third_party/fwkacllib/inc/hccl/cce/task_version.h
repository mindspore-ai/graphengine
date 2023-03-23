/**
 * @file cce/task_version.h
 *
 * Copyright(C), 2017 - 2017, Huawei Tech. Co., Ltd. ALL RIGHTS RESERVED.
 *
 * @brief header file of cce version info.
 *
 * @version 1.0
 *
 */
#ifndef __TASK_VERSION_H__
#define __TASK_VERSION_H__

// Add by one only when taks info is changed, once chaged elder version needs to recompile davinci models online.
#define CCE_TASK_VERSION_COUNTER (0)

namespace cce
{
/**
 * @ingroup cce
 * @brief get cce task info version counter's value.
 * @param [out] count , task info version count
 * @return ccStatus_t
*/
ccStatus_t ccGetRtVersion(uint32_t *count);

} // namespace cce

#endif