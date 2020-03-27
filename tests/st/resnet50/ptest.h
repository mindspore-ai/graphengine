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

#ifndef PTEST_H
#define PTEST_H

#include <stdarg.h>
#include <string.h>
#include <exception>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>

namespace ptest {
class assertion_error : public std::exception {
 public:
  const char *what() const throw() { return "Assertion Exception"; }
};

class TestFixture {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}
  void Run() { _func(); }
  void BindFunction(std::function<void(void)> function) { _func = function; }
  void SetName(const std::string &name) { _name = name; }
  std::string Name() const { return _name; }
  virtual ~TestFixture() {}

 private:
  std::function<void(void)> _func;
  std::string _name;
};

enum TestResult { SUCCESS, FAILED, UNAVAILABLE, UNKNOWN, NOCASEFOUND };

class TestManager {
 public:
  static TestManager &GetSingleton() {
    static TestManager instance;
    return instance;
  }
  void RegisterTest(const std::string &name, TestFixture *fixture) { _testfixtures[name] = fixture; }

  const std::string GetRunningTestcaseName() const { return _running_testcase_name; }

  const std::list<std::string> GetAllTestNames() const {
    std::list<std::string> result;
    for (auto &t : _testfixtures) {
      result.push_back(t.first);
    }
    return result;
  }

  TestResult RunTest(const std::string &name) {
    if (_testfixtures.find(name) == _testfixtures.end()) {
      return NOCASEFOUND;
    }

    _running_testcase_name = name;

    do {
      SetTestResult(name, UNKNOWN);
      _testfixtures[name]->SetUp();
      if (_testresults[name] == FAILED) {
        _testresults[name] = UNAVAILABLE;
        break;
      }
      SetTestResult(name, SUCCESS);
      try {
        _testfixtures[name]->Run();
      } catch (assertion_error &e) {
        // Do nothing as the error has been handled by the TestManager.
      }
      _testfixtures[name]->TearDown();
    } while (0);

    return _testresults[name];
  }
  void SetTestResult(const std::string &name, TestResult result) { _testresults[name] = result; }
  TestResult GetTestResult(const std::string &name) { return _testresults[name]; }

 private:
  std::map<std::string, TestFixture *> _testfixtures;
  std::map<std::string, TestResult> _testresults;
  std::string _running_testcase_name;
};

class TestFixtureRegister {
 public:
  TestFixtureRegister(const std::string &name, TestFixture *fixture, std::function<void(void)> function) {
    fixture->BindFunction(function);
    fixture->SetName(name);
    TestManager::GetSingleton().RegisterTest(name, fixture);
  }
};
}  // namespace ptest

#define _STR(x) #x
#define _EMPTY_NAMESPACE

#define _TEST(NAMESPACE, FIXTURECLASS, TESTNAME, CASENAME)                                              \
  void g_func_##TESTNAME##_##CASENAME(void);                                                            \
  NAMESPACE::FIXTURECLASS g_fixture_##TESTNAME##_##CASENAME;                                            \
  ptest::TestFixtureRegister g_register_##TESTNAME##_##CASENAME(                                        \
      _STR(TESTNAME##_##CASENAME), &g_fixture_##TESTNAME##_##CASENAME, g_func_##TESTNAME##_##CASENAME); \
  void g_func_##TESTNAME##_##CASENAME(void)

#define TEST(TESTNAME, CASENAME) _TEST(ptest, TestFixture, TESTNAME, CASENAME)

#define TEST_F(TESTFIXTURE, CASENAME) _TEST(_EMPTY_NAMESPACE, TESTFIXTURE, TESTFIXTURE, CASENAME)

#define EXPECT_TRUE(X)                                                                    \
  do {                                                                                    \
    if (!(X)) {                                                                           \
      std::string testname = ptest::TestManager::GetSingleton().GetRunningTestcaseName(); \
      ptest::TestManager::GetSingleton().SetTestResult(testname, ptest::FAILED);          \
      std::cerr << #X << "Expectation Failed\n"                                           \
                << "Testcase Name: " << testname << "\n"                                  \
                << "File: " __FILE__ << "\tLine:" << __LINE__ << std::endl;               \
    }                                                                                     \
  } while (0);

// With the macro definition ensures that the compiler can detect compiler warning.
#define Max_Log_Len 1024
#define PRINT_ERR(lpszFormat, ...)                              \
  do {                                                          \
    char szTmpBuf[Max_Log_Len + 1] = {0};                       \
    snprintf(szTmpBuf, Max_Log_Len, lpszFormat, ##__VA_ARGS__); \
    std::cerr << szTmpBuf << std::endl;                         \
  } while (0)

// Increase the content of print error messages and error to facilitate rapid analysis
#define EXPECT_TRUE_C(X, ERR_TYPE, format, ...)                                                             \
  do {                                                                                                      \
    if (!(X)) {                                                                                             \
      std::string testname = ptest::TestManager::GetSingleton().GetRunningTestcaseName();                   \
      ptest::TestManager::GetSingleton().SetTestResult(testname, ptest::FAILED);                            \
      std::cerr << #X << " Expectation Failed."                                                             \
                << "Testcase Name: " << testname << " File:" __FILE__ << " Line:" << __LINE__ << std::endl; \
      PRINT_ERR("[" ERR_TYPE "]" format, ##__VA_ARGS__);                                                    \
    }                                                                                                       \
  } while (0)

#define ASSERT_TRUE(X)                                                                    \
  do {                                                                                    \
    if (!(X)) {                                                                           \
      std::string testname = ptest::TestManager::GetSingleton().GetRunningTestcaseName(); \
      ptest::TestManager::GetSingleton().SetTestResult(testname, ptest::FAILED);          \
      std::cerr << #X << "Assertion Failed\n"                                             \
                << "Testcase Name: " << testname << "\n"                                  \
                << "File: " __FILE__ << "\tLine:" << __LINE__ << std::endl;               \
      throw ptest::assertion_error();                                                     \
    }                                                                                     \
  } while (0);

// Add printing error information and error line content for quick analysis
#define ASSERT_TRUE_C(X, ERR_TYPE, format, ...)                                                             \
  do {                                                                                                      \
    if (!(X)) {                                                                                             \
      std::string testname = ptest::TestManager::GetSingleton().GetRunningTestcaseName();                   \
      ptest::TestManager::GetSingleton().SetTestResult(testname, ptest::FAILED);                            \
      std::cerr << #X << " Assertion Failed."                                                               \
                << "Testcase Name: " << testname << " File:" __FILE__ << " Line:" << __LINE__ << std::endl; \
      PRINT_ERR("[" ERR_TYPE "]" format, ##__VA_ARGS__);                                                    \
      throw ptest::assertion_error();                                                                       \
    }                                                                                                       \
  } while (0);

#define CONFIG_ERR "CONFIG_ERR"
#define LOAD_MODEL_ERR "LOAD_MODEL_ERR"
#define FILE_READ_ERR "FILE_READ_ERR"
#define RUN_ERROR "RUN_ERROR"
#define MEM_ERROR "MEM_ERROR"
#define RESULT_ERR "RESULT_ERR"

#define EXPECT_FALSE(X) EXPECT_TRUE(!(X))
#define EXPECT_EQ(X, Y) EXPECT_TRUE(((X) == (Y)))
#define EXPECT_NE(X, Y) EXPECT_TRUE(((X) != (Y)))
#define EXPECT_GT(X, Y) EXPECT_TRUE(((X) > (Y)))
#define EXPECT_GE(X, Y) EXPECT_TRUE(((X) >= (Y)))
#define EXPECT_LT(X, Y) EXPECT_TRUE(((X) < (Y)))
#define EXPECT_LE(X, Y) EXPECT_TRUE(((X) <= (Y)))

#define EXPECT_FALSE_C(X, ERR_TYPE, format, ...) EXPECT_TRUE_C(!(X), ERR_TYPE, format, ##__VA_ARGS__)
#define EXPECT_EQ_C(X, Y, ERR_TYPE, format, ...) EXPECT_TRUE_C(((X) == (Y)), ERR_TYPE, format, ##__VA_ARGS__)
#define EXPECT_NE_C(X, Y, ERR_TYPE, format, ...) EXPECT_TRUE_C(((X) != (Y)), ERR_TYPE, format, ##__VA_ARGS__)
#define EXPECT_GT_C(X, Y, ERR_TYPE, format, ...) EXPECT_TRUE_C(((X) > (Y)), ERR_TYPE, format, ##__VA_ARGS__)
#define EXPECT_GE_C(X, Y, ERR_TYPE, format, ...) EXPECT_TRUE_C(((X) >= (Y)), ERR_TYPE, format, ##__VA_ARGS__)
#define EXPECT_LT_C(X, Y, ERR_TYPE, format, ...) EXPECT_TRUE_C(((X) < (Y)), ERR_TYPE, format, ##__VA_ARGS__)
#define EXPECT_LE_C(X, Y, ERR_TYPE, format, ...) EXPECT_TRUE_C(((X) <= (Y)), ERR_TYPE, format, ##__VA_ARGS__)

#define ASSERT_FALSE(X) ASSERT_TRUE(!(X))
#define ASSERT_EQ(X, Y) ASSERT_TRUE(((X) == (Y)))
#define ASSERT_NE(X, Y) ASSERT_TRUE(((X) != (Y)))
#define ASSERT_GT(X, Y) ASSERT_TRUE(((X) > (Y)))
#define ASSERT_GE(X, Y) ASSERT_TRUE(((X) >= (Y)))
#define ASSERT_LT(X, Y) ASSERT_TRUE(((X) < (Y)))
#define ASSERT_LE(X, Y) ASSERT_TRUE(((X) <= (Y)))

#define ASSERT_FALSE_C(X, ERR_TYPE, format, ...) ASSERT_TRUE_C(!(X), ERR_TYPE, format, ##__VA_ARGS__)
#define ASSERT_EQ_C(X, Y, ERR_TYPE, format, ...) ASSERT_TRUE_C(((X) == (Y)), ERR_TYPE, format, ##__VA_ARGS__)
#define ASSERT_NE_C(X, Y, ERR_TYPE, format, ...) ASSERT_TRUE_C(((X) != (Y)), ERR_TYPE, format, ##__VA_ARGS__)
#define ASSERT_GT_C(X, Y, ERR_TYPE, format, ...) ASSERT_TRUE_C(((X) > (Y)), ERR_TYPE, format, ##__VA_ARGS__)
#define ASSERT_GE_C(X, Y, ERR_TYPE, format, ...) ASSERT_TRUE_C(((X) >= (Y)), ERR_TYPE, format, ##__VA_ARGS__)
#define ASSERT_LT_C(X, Y, ERR_TYPE, format, ...) ASSERT_TRUE_C(((X) < (Y)), ERR_TYPE, format, ##__VA_ARGS__)
#define ASSERT_LE_C(X, Y, ERR_TYPE, format, ...) ASSERT_TRUE_C(((X) <= (Y)), ERR_TYPE, format, ##__VA_ARGS__)

#endif
