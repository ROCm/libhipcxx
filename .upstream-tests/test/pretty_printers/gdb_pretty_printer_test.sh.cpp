// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// -*- C++ -*-
// UNSUPPORTED: system-windows
// REQUIRES: libcxx_gdb
//
// RUN: %cxx %flags %s -o %t.exe %compile_flags -g %link_flags
// Ensure locale-independence for unicode tests.
// RUN: %libcxx_gdb -nx -batch -iex "set autoload off" -ex "source %libcxx_src_root/utils/gdb/libcxx/printers.py" -ex "python register_libcxx_printer_loader()" -ex "source %libcxx_src_root/test/pretty_printers/gdb_pretty_printer_test.py" %t.exe

#include <hip/std/bitset>
#include <hip/std/deque>
#include <hip/std/list>
#include <hip/std/map>
#include <hip/std/memory>
#include <hip/std/queue>
#include <hip/std/set>
#include <hip/std/sstream>
#include <hip/std/stack>
#include <hip/std/string>
#include <hip/std/tuple>
#include <hip/std/unordered_map>
#include <hip/std/unordered_set>

#include "test_macros.h"

// To write a pretty-printer test:
//
// 1. Declare a variable of the type you want to test
//
// 2. Set its value to something which will test the pretty printer in an
//    interesting way.
//
// 3. Call ComparePrettyPrintToChars with that variable, and a "const char*"
//    value to compare to the printer's output.
//
//    Or
//
//    Call ComparePrettyPrintToChars with that variable, and a "const char*"
//    *python* regular expression to match against the printer's output.
//    The set of special characters in a Python regular expression overlaps
//    with a lot of things the pretty printers print--brackets, for
//    example--so take care to escape appropriately.
//
// Alternatively, construct a string that gdb can parse as an expression,
// so that printing the value of the expression will test the pretty printer
// in an interesting way. Then, call CompareExpressionPrettyPrintToChars or
// CompareExpressionPrettyPrintToRegex to compare the printer's output.

// Avoids setting a breakpoint in every-single instantiation of
// ComparePrettyPrintTo*.  Also, make sure neither it, nor the
// variables we need present in the Compare functions are optimized
// away.
#ifdef TEST_COMPILER_GCC
#define OPT_NONE __attribute__((noinline))
#else
#define OPT_NONE __attribute__((optnone))
#endif
void StopForDebugger(void *value, void *check) OPT_NONE;
void StopForDebugger(void *value, void *check)  {}


// Prevents the compiler optimizing away the parameter in the caller function.
template <typename Type>
void MarkAsLive(Type &&t) OPT_NONE;
template <typename Type>
void MarkAsLive(Type &&t) {}

// In all of the Compare(Expression)PrettyPrintTo(Regex/Chars) functions below,
// the python script sets a breakpoint just before the call to StopForDebugger,
// compares the result to the expectation.
//
// The expectation is a literal string to be matched exactly in
// *PrettyPrintToChars functions, and is a python regular expression in
// *PrettyPrintToRegex functions.
//
// In ComparePrettyPrint* functions, the value is a variable of any type. In
// CompareExpressionPrettyPrint functions, the value is a string expression that
// gdb will parse and print the result.
//
// The python script will print either "PASS", or a detailed failure explanation
// along with the line that has invoke the function. The testing will continue
// in either case.

template <typename TypeToPrint> void ComparePrettyPrintToChars(
    TypeToPrint value,
    const char *expectation) {
  StopForDebugger(&value, &expectation);
}

template <typename TypeToPrint> void ComparePrettyPrintToRegex(
    TypeToPrint value,
    const char *expectation) {
  StopForDebugger(&value, &expectation);
}

void CompareExpressionPrettyPrintToChars(
    hip::std::string value,
    const char *expectation) {
  StopForDebugger(&value, &expectation);
}

void CompareExpressionPrettyPrintToRegex(
    hip::std::string value,
    const char *expectation) {
  StopForDebugger(&value, &expectation);
}

namespace example {
  struct example_struct {
    int a = 0;
    int arr[1000];
  };
}

// If enabled, the self test will "fail"--because we want to be sure it properly
// diagnoses tests that *should* fail. Evaluate the output by hand.
void framework_self_test() {
#ifdef FRAMEWORK_SELF_TEST
  // Use the most simple data structure we can.
  const char a = 'a';

  // Tests that should pass
  ComparePrettyPrintToChars(a, "97 'a'");
  ComparePrettyPrintToRegex(a, ".*");

  // Tests that should fail.
  ComparePrettyPrintToChars(a, "b");
  ComparePrettyPrintToRegex(a, "b");
#endif
}

// A simple pass-through allocator to check that we handle CompressedPair
// correctly.
template <typename T> class UncompressibleAllocator : public hip::std::allocator<T> {
 public:
  char X;
};

void string_test() {
  hip::std::string short_string("kdjflskdjf");
  // The display_hint "string" adds quotes the printed result.
  ComparePrettyPrintToChars(short_string, "\"kdjflskdjf\"");

  hip::std::basic_string<char, hip::std::char_traits<char>, UncompressibleAllocator<char>>
      long_string("mehmet bizim dostumuz agzi kirik testimiz");
  ComparePrettyPrintToChars(long_string,
                            "\"mehmet bizim dostumuz agzi kirik testimiz\"");
}

void u16string_test() {
  hip::std::u16string test0 = u"Hello World";
  ComparePrettyPrintToChars(test0, "u\"Hello World\"");
  hip::std::u16string test1 = u"\U00010196\u20AC\u00A3\u0024";
  ComparePrettyPrintToChars(test1, "u\"\U00010196\u20AC\u00A3\u0024\"");
  hip::std::u16string test2 = u"\u0024\u0025\u0026\u0027";
  ComparePrettyPrintToChars(test2, "u\"\u0024\u0025\u0026\u0027\"");
  hip::std::u16string test3 = u"mehmet bizim dostumuz agzi kirik testimiz";
  ComparePrettyPrintToChars(test3,
                            ("u\"mehmet bizim dostumuz agzi kirik testimiz\""));
}

void u32string_test() {
  hip::std::u32string test0 = U"Hello World";
  ComparePrettyPrintToChars(test0, "U\"Hello World\"");
  hip::std::u32string test1 =
      U"\U0001d552\U0001d553\U0001d554\U0001d555\U0001d556\U0001d557";
  ComparePrettyPrintToChars(
      test1,
      ("U\"\U0001d552\U0001d553\U0001d554\U0001d555\U0001d556\U0001d557\""));
  hip::std::u32string test2 = U"\U00004f60\U0000597d";
  ComparePrettyPrintToChars(test2, ("U\"\U00004f60\U0000597d\""));
  hip::std::u32string test3 = U"mehmet bizim dostumuz agzi kirik testimiz";
  ComparePrettyPrintToChars(test3, ("U\"mehmet bizim dostumuz agzi kirik testimiz\""));
}

void tuple_test() {
  hip::std::tuple<int, int, int> test0(2, 3, 4);
  ComparePrettyPrintToChars(
      test0,
      "hip::std::tuple containing = {[1] = 2, [2] = 3, [3] = 4}");

  hip::std::tuple<> test1;
  ComparePrettyPrintToChars(
      test1,
      "empty hip::std::tuple");
}

void unique_ptr_test() {
  hip::std::unique_ptr<hip::std::string> matilda(new hip::std::string("Matilda"));
  ComparePrettyPrintToRegex(
      hip::std::move(matilda),
      R"(hip::std::unique_ptr<hip::std::string> containing = {__ptr_ = 0x[a-f0-9]+})");
  hip::std::unique_ptr<int> forty_two(new int(42));
  ComparePrettyPrintToRegex(hip::std::move(forty_two),
      R"(hip::std::unique_ptr<int> containing = {__ptr_ = 0x[a-f0-9]+})");

  hip::std::unique_ptr<int> this_is_null;
  ComparePrettyPrintToChars(hip::std::move(this_is_null),
      R"(hip::std::unique_ptr is nullptr)");
}

void bitset_test() {
  hip::std::bitset<258> i_am_empty(0);
  ComparePrettyPrintToChars(i_am_empty, "hip::std::bitset<258>");

  hip::std::bitset<0> very_empty;
  ComparePrettyPrintToChars(very_empty, "hip::std::bitset<0>");

  hip::std::bitset<15> b_000001111111100(1020);
  ComparePrettyPrintToChars(b_000001111111100,
      "hip::std::bitset<15> = {[2] = 1, [3] = 1, [4] = 1, [5] = 1, [6] = 1, "
      "[7] = 1, [8] = 1, [9] = 1}");

  hip::std::bitset<258> b_0_129_132(0);
  b_0_129_132[0] = true;
  b_0_129_132[129] = true;
  b_0_129_132[132] = true;
  ComparePrettyPrintToChars(b_0_129_132,
      "hip::std::bitset<258> = {[0] = 1, [129] = 1, [132] = 1}");
}

void list_test() {
  hip::std::list<int> i_am_empty{};
  ComparePrettyPrintToChars(i_am_empty, "hip::std::list is empty");

  hip::std::list<int> one_two_three {1, 2, 3};
  ComparePrettyPrintToChars(one_two_three,
      "hip::std::list with 3 elements = {1, 2, 3}");

  hip::std::list<hip::std::string> colors {"red", "blue", "green"};
  ComparePrettyPrintToChars(colors,
      R"(hip::std::list with 3 elements = {"red", "blue", "green"})");
}

void deque_test() {
  hip::std::deque<int> i_am_empty{};
  ComparePrettyPrintToChars(i_am_empty, "hip::std::deque is empty");

  hip::std::deque<int> one_two_three {1, 2, 3};
  ComparePrettyPrintToChars(one_two_three,
      "hip::std::deque with 3 elements = {1, 2, 3}");

  hip::std::deque<example::example_struct> bfg;
  for (int i = 0; i < 10; ++i) {
    example::example_struct current;
    current.a = i;
    bfg.push_back(current);
  }
  for (int i = 0; i < 3; ++i) {
    bfg.pop_front();
  }
  for (int i = 0; i < 3; ++i) {
    bfg.pop_back();
  }
  ComparePrettyPrintToRegex(bfg,
      "hip::std::deque with 4 elements = {"
      "{a = 3, arr = {[^}]+}}, "
      "{a = 4, arr = {[^}]+}}, "
      "{a = 5, arr = {[^}]+}}, "
      "{a = 6, arr = {[^}]+}}}");
}

void map_test() {
  hip::std::map<int, int> i_am_empty{};
  ComparePrettyPrintToChars(i_am_empty, "hip::std::map is empty");

  hip::std::map<int, hip::std::string> one_two_three;
  one_two_three.insert({1, "one"});
  one_two_three.insert({2, "two"});
  one_two_three.insert({3, "three"});
  ComparePrettyPrintToChars(one_two_three,
      "hip::std::map with 3 elements = "
      R"({[1] = "one", [2] = "two", [3] = "three"})");

  hip::std::map<int, example::example_struct> bfg;
  for (int i = 0; i < 4; ++i) {
    example::example_struct current;
    current.a = 17 * i;
    bfg.insert({i, current});
  }
  ComparePrettyPrintToRegex(bfg,
      R"(hip::std::map with 4 elements = {)"
      R"(\[0\] = {a = 0, arr = {[^}]+}}, )"
      R"(\[1\] = {a = 17, arr = {[^}]+}}, )"
      R"(\[2\] = {a = 34, arr = {[^}]+}}, )"
      R"(\[3\] = {a = 51, arr = {[^}]+}}})");
}

void multimap_test() {
  hip::std::multimap<int, int> i_am_empty{};
  ComparePrettyPrintToChars(i_am_empty, "hip::std::multimap is empty");

  hip::std::multimap<int, hip::std::string> one_two_three;
  one_two_three.insert({1, "one"});
  one_two_three.insert({3, "three"});
  one_two_three.insert({1, "ein"});
  one_two_three.insert({2, "two"});
  one_two_three.insert({2, "zwei"});
  one_two_three.insert({1, "bir"});

  ComparePrettyPrintToChars(one_two_three,
      "hip::std::multimap with 6 elements = "
      R"({[1] = "one", [1] = "ein", [1] = "bir", )"
      R"([2] = "two", [2] = "zwei", [3] = "three"})");
}

void queue_test() {
  hip::std::queue<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty,
      "hip::std::queue wrapping = {hip::std::deque is empty}");

  hip::std::queue<int> one_two_three(hip::std::deque<int>{1, 2, 3});
    ComparePrettyPrintToChars(one_two_three,
        "hip::std::queue wrapping = {"
        "hip::std::deque with 3 elements = {1, 2, 3}}");
}

void priority_queue_test() {
  hip::std::priority_queue<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty,
      "hip::std::priority_queue wrapping = {hip::std::vector of length 0, capacity 0}");

  hip::std::priority_queue<int> one_two_three;
  one_two_three.push(11111);
  one_two_three.push(22222);
  one_two_three.push(33333);

  ComparePrettyPrintToRegex(one_two_three,
      R"(hip::std::priority_queue wrapping = )"
      R"({hip::std::vector of length 3, capacity 3 = {33333)");

  ComparePrettyPrintToRegex(one_two_three, ".*11111.*");
  ComparePrettyPrintToRegex(one_two_three, ".*22222.*");
}

void set_test() {
  hip::std::set<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "hip::std::set is empty");

  hip::std::set<int> one_two_three {3, 1, 2};
  ComparePrettyPrintToChars(one_two_three,
      "hip::std::set with 3 elements = {1, 2, 3}");

  hip::std::set<hip::std::pair<int, int>> prime_pairs {
      hip::std::make_pair(3, 5), hip::std::make_pair(5, 7), hip::std::make_pair(3, 5)};

  ComparePrettyPrintToChars(prime_pairs,
      "hip::std::set with 2 elements = {"
      "{first = 3, second = 5}, {first = 5, second = 7}}");
}

void stack_test() {
  hip::std::stack<int> test0;
  ComparePrettyPrintToChars(test0,
                            "hip::std::stack wrapping = {hip::std::deque is empty}");
  test0.push(5);
  test0.push(6);
  ComparePrettyPrintToChars(
      test0, "hip::std::stack wrapping = {hip::std::deque with 2 elements = {5, 6}}");
  hip::std::stack<bool> test1;
  test1.push(true);
  test1.push(false);
  ComparePrettyPrintToChars(
      test1,
      "hip::std::stack wrapping = {hip::std::deque with 2 elements = {true, false}}");

  hip::std::stack<hip::std::string> test2;
  test2.push("Hello");
  test2.push("World");
  ComparePrettyPrintToChars(test2,
                            "hip::std::stack wrapping = {hip::std::deque with 2 elements "
                            "= {\"Hello\", \"World\"}}");
}

void multiset_test() {
  hip::std::multiset<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "hip::std::multiset is empty");

  hip::std::multiset<hip::std::string> one_two_three {"1:one", "2:two", "3:three", "1:one"};
  ComparePrettyPrintToChars(one_two_three,
      "hip::std::multiset with 4 elements = {"
      R"("1:one", "1:one", "2:two", "3:three"})");
}

void vector_test() {
  hip::std::vector<bool> test0 = {true, false};
  ComparePrettyPrintToChars(test0,
                            "hip::std::vector<bool> of "
                            "length 2, capacity 64 = {1, 0}");
  for (int i = 0; i < 31; ++i) {
    test0.push_back(true);
    test0.push_back(false);
  }
  ComparePrettyPrintToRegex(
      test0,
      "hip::std::vector<bool> of length 64, "
      "capacity 64 = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, "
      "0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, "
      "0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}");
  test0.push_back(true);
  ComparePrettyPrintToRegex(
      test0,
      "hip::std::vector<bool> of length 65, "
      "capacity 128 = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, "
      "1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, "
      "1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}");

  hip::std::vector<int> test1;
  ComparePrettyPrintToChars(test1, "hip::std::vector of length 0, capacity 0");

  hip::std::vector<int> test2 = {5, 6, 7};
  ComparePrettyPrintToChars(test2,
                            "hip::std::vector of length "
                            "3, capacity 3 = {5, 6, 7}");

  hip::std::vector<int, UncompressibleAllocator<int>> test3({7, 8});
  ComparePrettyPrintToChars(hip::std::move(test3),
                            "hip::std::vector of length "
                            "2, capacity 2 = {7, 8}");
}

void set_iterator_test() {
  hip::std::set<int> one_two_three {1111, 2222, 3333};
  auto it = one_two_three.find(2222);
  MarkAsLive(it);
  CompareExpressionPrettyPrintToRegex("it",
      R"(hip::std::__tree_const_iterator  = {\[0x[a-f0-9]+\] = 2222})");

  auto not_found = one_two_three.find(1234);
  MarkAsLive(not_found);
  // Because the end_node is not easily detected, just be sure it doesn't crash.
  CompareExpressionPrettyPrintToRegex("not_found",
      R"(hip::std::__tree_const_iterator ( = {\[0x[a-f0-9]+\] = .*}|<error reading variable:.*>))");
}

void map_iterator_test() {
  hip::std::map<int, hip::std::string> one_two_three;
  one_two_three.insert({1, "one"});
  one_two_three.insert({2, "two"});
  one_two_three.insert({3, "three"});
  auto it = one_two_three.begin();
  MarkAsLive(it);
  CompareExpressionPrettyPrintToRegex("it",
      R"(hip::std::__map_iterator  = )"
      R"({\[0x[a-f0-9]+\] = {first = 1, second = "one"}})");

  auto not_found = one_two_three.find(7);
  MarkAsLive(not_found);
  CompareExpressionPrettyPrintToRegex("not_found",
      R"(hip::std::__map_iterator  = {\[0x[a-f0-9]+\] =  end\(\)})");
}

void unordered_set_test() {
  hip::std::unordered_set<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "hip::std::unordered_set is empty");

  hip::std::unordered_set<int> numbers {12345, 67890, 222333, 12345};
  numbers.erase(numbers.find(222333));
  ComparePrettyPrintToRegex(numbers, "hip::std::unordered_set with 2 elements = ");
  ComparePrettyPrintToRegex(numbers, ".*12345.*");
  ComparePrettyPrintToRegex(numbers, ".*67890.*");

  hip::std::unordered_set<hip::std::string> colors {"red", "blue", "green"};
  ComparePrettyPrintToRegex(colors, "hip::std::unordered_set with 3 elements = ");
  ComparePrettyPrintToRegex(colors, R"(.*"red".*)");
  ComparePrettyPrintToRegex(colors, R"(.*"blue".*)");
  ComparePrettyPrintToRegex(colors, R"(.*"green".*)");
}

void unordered_multiset_test() {
  hip::std::unordered_multiset<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "hip::std::unordered_multiset is empty");

  hip::std::unordered_multiset<int> numbers {12345, 67890, 222333, 12345};
  ComparePrettyPrintToRegex(numbers,
                            "hip::std::unordered_multiset with 4 elements = ");
  ComparePrettyPrintToRegex(numbers, ".*12345.*12345.*");
  ComparePrettyPrintToRegex(numbers, ".*67890.*");
  ComparePrettyPrintToRegex(numbers, ".*222333.*");

  hip::std::unordered_multiset<hip::std::string> colors {"red", "blue", "green", "red"};
  ComparePrettyPrintToRegex(colors,
                            "hip::std::unordered_multiset with 4 elements = ");
  ComparePrettyPrintToRegex(colors, R"(.*"red".*"red".*)");
  ComparePrettyPrintToRegex(colors, R"(.*"blue".*)");
  ComparePrettyPrintToRegex(colors, R"(.*"green".*)");
}

void unordered_map_test() {
  hip::std::unordered_map<int, int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "hip::std::unordered_map is empty");

  hip::std::unordered_map<int, hip::std::string> one_two_three;
  one_two_three.insert({1, "one"});
  one_two_three.insert({2, "two"});
  one_two_three.insert({3, "three"});
  ComparePrettyPrintToRegex(one_two_three,
                            "hip::std::unordered_map with 3 elements = ");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[1\] = "one".*)");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[2\] = "two".*)");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[3\] = "three".*)");
}

void unordered_multimap_test() {
  hip::std::unordered_multimap<int, int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "hip::std::unordered_multimap is empty");

  hip::std::unordered_multimap<int, hip::std::string> one_two_three;
  one_two_three.insert({1, "one"});
  one_two_three.insert({2, "two"});
  one_two_three.insert({3, "three"});
  one_two_three.insert({2, "two"});
  ComparePrettyPrintToRegex(one_two_three,
                            "hip::std::unordered_multimap with 4 elements = ");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[1\] = "one".*)");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[2\] = "two".*\[2\] = "two")");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[3\] = "three".*)");
}

void unordered_map_iterator_test() {
  hip::std::unordered_map<int, int> ones_to_eights;
  ones_to_eights.insert({1, 8});
  ones_to_eights.insert({11, 88});
  ones_to_eights.insert({111, 888});

  auto ones_to_eights_begin = ones_to_eights.begin();
  MarkAsLive(ones_to_eights_begin);
  CompareExpressionPrettyPrintToRegex("ones_to_eights_begin",
      R"(hip::std::__hash_map_iterator  = {\[1+\] = 8+})");

  auto not_found = ones_to_eights.find(5);
  MarkAsLive(not_found);
  CompareExpressionPrettyPrintToRegex("not_found",
      R"(hip::std::__hash_map_iterator = end\(\))");
}

void unordered_set_iterator_test() {
  hip::std::unordered_set<int> ones;
  ones.insert(111);
  ones.insert(1111);
  ones.insert(11111);

  auto ones_begin = ones.begin();
  MarkAsLive(ones_begin);
  CompareExpressionPrettyPrintToRegex("ones_begin",
      R"(hip::std::__hash_const_iterator  = {1+})");

  auto not_found = ones.find(5);
  MarkAsLive(not_found);
  CompareExpressionPrettyPrintToRegex("not_found",
      R"(hip::std::__hash_const_iterator = end\(\))");
}

// Check that libc++ pretty printers do not handle pointers.
void pointer_negative_test() {
  int abc = 123;
  int *int_ptr = &abc;
  // Check that the result is equivalent to "p/r int_ptr" command.
  ComparePrettyPrintToRegex(int_ptr, R"(\(int \*\) 0x[a-f0-9]+)");
}

void shared_ptr_test() {
  // Shared ptr tests while using test framework call another function
  // due to which there is one more count for the pointer. Hence, all the
  // following tests are testing with expected count plus 1.
  hip::std::shared_ptr<const int> test0 = hip::std::make_shared<const int>(5);
  ComparePrettyPrintToRegex(
      test0,
      R"(hip::std::shared_ptr<int> count 2, weak 0 containing = {__ptr_ = 0x[a-f0-9]+})");

  hip::std::shared_ptr<const int> test1(test0);
  ComparePrettyPrintToRegex(
      test1,
      R"(hip::std::shared_ptr<int> count 3, weak 0 containing = {__ptr_ = 0x[a-f0-9]+})");

  {
    hip::std::weak_ptr<const int> test2 = test1;
    ComparePrettyPrintToRegex(
        test0,
        R"(hip::std::shared_ptr<int> count 3, weak 1 containing = {__ptr_ = 0x[a-f0-9]+})");
  }

  ComparePrettyPrintToRegex(
      test0,
      R"(hip::std::shared_ptr<int> count 3, weak 0 containing = {__ptr_ = 0x[a-f0-9]+})");

  hip::std::shared_ptr<const int> test3;
  ComparePrettyPrintToChars(test3, "hip::std::shared_ptr is nullptr");
}

void streampos_test() {
  hip::std::streampos test0 = 67;
  ComparePrettyPrintToChars(
      test0, "hip::std::fpos with stream offset:67 with state: {count:0 value:0}");
  hip::std::istringstream input("testing the input stream here");
  hip::std::streampos test1 = input.tellg();
  ComparePrettyPrintToChars(
      test1, "hip::std::fpos with stream offset:0 with state: {count:0 value:0}");
  hip::std::unique_ptr<char[]> buffer(new char[5]);
  input.read(buffer.get(), 5);
  test1 = input.tellg();
  ComparePrettyPrintToChars(
      test1, "hip::std::fpos with stream offset:5 with state: {count:0 value:0}");
}

int main(int argc, char* argv[]) {
  framework_self_test();

  string_test();

  u32string_test();
  tuple_test();
  unique_ptr_test();
  shared_ptr_test();
  bitset_test();
  list_test();
  deque_test();
  map_test();
  multimap_test();
  queue_test();
  priority_queue_test();
  stack_test();
  set_test();
  multiset_test();
  vector_test();
  set_iterator_test();
  map_iterator_test();
  unordered_set_test();
  unordered_multiset_test();
  unordered_map_test();
  unordered_multimap_test();
  unordered_map_iterator_test();
  unordered_set_iterator_test();
  pointer_negative_test();
  streampos_test();
  return 0;
}
