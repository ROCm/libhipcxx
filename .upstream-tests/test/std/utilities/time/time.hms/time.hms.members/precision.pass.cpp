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
// UNSUPPORTED: c++98, c++03, c++11
// <chrono>

// template <class Duration>
// class hh_mm_ss
// {
// public:
//     static unsigned constexpr fractional_width = see below;
//     using precision                            = see below;
//
//   precision is duration<common_type_t<Duration::rep, seconds::rep>, 
//                                 ratio<1, 10^^fractional_width>>
   
#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"

constexpr unsigned long long powers[] = {
	1ULL,
	10ULL,
	100ULL,
	1000ULL,
	10000ULL,
	100000ULL,
	1000000ULL,
	10000000ULL,
	100000000ULL,
	1000000000ULL,
	10000000000ULL,
	100000000000ULL,
	1000000000000ULL,
	10000000000000ULL,
	100000000000000ULL,
	1000000000000000ULL,
	10000000000000000ULL,
	100000000000000000ULL,
	1000000000000000000ULL,
	10000000000000000000ULL
	};
	
template <typename Duration, unsigned width>
__host__ __device__
constexpr bool check_precision()
{
	using HMS = hip::std::chrono::hh_mm_ss<Duration>;
	using CT  = hip::std::common_type_t<typename Duration::rep, hip::std::chrono::seconds::rep>;
	using Pre = hip::std::chrono::duration<CT, hip::std::ratio<1, powers[width]>>;
	return hip::std::is_same_v<typename HMS::precision, Pre>;
}

int main(int, char**)
{
	using microfortnights = hip::std::chrono::duration<int, hip::std::ratio<756, 625>>;

	static_assert( check_precision<hip::std::chrono::hours,                               0>(), "");
	static_assert( check_precision<hip::std::chrono::minutes,                             0>(), "");
	static_assert( check_precision<hip::std::chrono::seconds,                             0>(), "");
	static_assert( check_precision<hip::std::chrono::milliseconds,                        3>(), "");
	static_assert( check_precision<hip::std::chrono::microseconds,                        6>(), "");
	static_assert( check_precision<hip::std::chrono::nanoseconds,                         9>(), "");
	static_assert( check_precision<hip::std::chrono::duration<int, hip::std::ratio<  1,   2>>, 1>(), "");
	static_assert( check_precision<hip::std::chrono::duration<int, hip::std::ratio<  1,   3>>, 6>(), "");
	static_assert( check_precision<hip::std::chrono::duration<int, hip::std::ratio<  1,   4>>, 2>(), "");
	static_assert( check_precision<hip::std::chrono::duration<int, hip::std::ratio<  1,   5>>, 1>(), "");
	static_assert( check_precision<hip::std::chrono::duration<int, hip::std::ratio<  1,   6>>, 6>(), "");
	static_assert( check_precision<hip::std::chrono::duration<int, hip::std::ratio<  1,   7>>, 6>(), "");
	static_assert( check_precision<hip::std::chrono::duration<int, hip::std::ratio<  1,   8>>, 3>(), "");
	static_assert( check_precision<hip::std::chrono::duration<int, hip::std::ratio<  1,   9>>, 6>(), "");
	static_assert( check_precision<hip::std::chrono::duration<int, hip::std::ratio<  1,  10>>, 1>(), "");
	static_assert( check_precision<microfortnights,                                  4>(), "");

	return 0;
}
