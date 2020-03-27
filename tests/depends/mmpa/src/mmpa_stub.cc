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

#include "mmpa/mmpa_api.h"

#include "common/types.h"
#include "common/util.h"

using namespace domi;

INT32 mmOpen(const CHAR *pathName, INT32 flags) {
  INT32 fd = HANDLE_INVALID_VALUE;

  if (NULL == pathName) {
    syslog(LOG_ERR, "The path name pointer is null.\r\n");
    return EN_INVALID_PARAM;
  }
  if (0 == (flags & (O_RDONLY | O_WRONLY | O_RDWR | O_CREAT))) {
    syslog(LOG_ERR, "The file open mode is error.\r\n");
    return EN_INVALID_PARAM;
  }

  fd = open(pathName, flags);
  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "Open file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return fd;
}

INT32 mmOpen2(const CHAR *pathName, INT32 flags, MODE mode) {
  INT32 fd = HANDLE_INVALID_VALUE;

  if (NULL == pathName) {
    syslog(LOG_ERR, "The path name pointer is null.\r\n");
    return EN_INVALID_PARAM;
  }
  if (MMPA_ZERO == (flags & (O_RDONLY | O_WRONLY | O_RDWR | O_CREAT))) {
    syslog(LOG_ERR, "The file open mode is error.\r\n");
    return EN_INVALID_PARAM;
  }
  if ((MMPA_ZERO == (mode & (S_IRUSR | S_IREAD))) && (MMPA_ZERO == (mode & (S_IWUSR | S_IWRITE)))) {
    syslog(LOG_ERR, "The permission mode of the file is error.\r\n");
    return EN_INVALID_PARAM;
  }

  fd = open(pathName, flags, mode);
  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "Open file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return fd;
}

INT32 mmClose(INT32 fd) {
  INT32 result = EN_OK;

  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "The file fd is invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = close(fd);
  if (EN_OK != result) {
    syslog(LOG_ERR, "Close the file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return EN_OK;
}

mmSsize_t mmWrite(INT32 fd, VOID *mmBuf, UINT32 mmCount) {
  mmSsize_t result = MMPA_ZERO;

  if ((fd < MMPA_ZERO) || (NULL == mmBuf)) {
    syslog(LOG_ERR, "Input parameter invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = write(fd, mmBuf, (size_t)mmCount);
  if (result < MMPA_ZERO) {
    syslog(LOG_ERR, "Write buf to file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return result;
}

mmSsize_t mmRead(INT32 fd, VOID *mmBuf, UINT32 mmCount) {
  mmSsize_t result = MMPA_ZERO;

  if ((fd < MMPA_ZERO) || (NULL == mmBuf)) {
    syslog(LOG_ERR, "Input parameter invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = read(fd, mmBuf, (size_t)mmCount);
  if (result < MMPA_ZERO) {
    syslog(LOG_ERR, "Read file to buf failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return result;
}

INT32 mmMkdir(const CHAR *lpPathName, mmMode_t mode) {
  INT32 tmode = mode;
  INT32 ret = EN_OK;

  if (NULL == lpPathName) {
    syslog(LOG_ERR, "The input path is null.\r\n");
    return EN_INVALID_PARAM;
  }

  if (tmode < MMPA_ZERO) {
    syslog(LOG_ERR, "The input mode is wrong.\r\n");
    return EN_INVALID_PARAM;
  }

  ret = mkdir(lpPathName, mode);

  if (EN_OK != ret) {
    syslog(LOG_ERR, "Failed to create the directory, the ret is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return EN_OK;
}

void *memCpyS(void *Dest, const void *Src, UINT32 Count) {
  char *tmp = (char *)Dest;
  char *s = (char *)Src;

  if (MMPA_ZERO == Count) {
    return Dest;
  }

  while (Count--) {
    *tmp++ = *s++;
  }
  return Dest;
}

INT32 mmRmdir(const CHAR *lpPathName) { return rmdir(lpPathName); }

mmTimespec mmGetTickCount() {
  mmTimespec rts;
  struct timespec ts = {0};
  (void)clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  rts.tv_sec = ts.tv_sec;
  rts.tv_nsec = ts.tv_nsec;
  return rts;
}

INT32 mmGetTid() {
  INT32 ret = (INT32)syscall(SYS_gettid);

  if (ret < MMPA_ZERO) {
    return EN_ERROR;
  }

  return ret;
}

INT32 mmAccess(const CHAR *pathName) {
  if (pathName == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = access(pathName, F_OK);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmStatGet(const CHAR *path, mmStat_t *buffer) {
  if ((path == NULL) || (buffer == NULL)) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = stat(path, buffer);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmGetFileSize(const CHAR *fileName, ULONGLONG *length) {
  if ((fileName == NULL) || (length == NULL)) {
    return EN_INVALID_PARAM;
  }
  struct stat fileStat;
  (void)memset_s(&fileStat, sizeof(fileStat), 0, sizeof(fileStat));  // unsafe_function_ignore: memset
  INT32 ret = lstat(fileName, &fileStat);
  if (ret < MMPA_ZERO) {
    return EN_ERROR;
  }
  *length = (ULONGLONG)fileStat.st_size;
  return EN_OK;
}
