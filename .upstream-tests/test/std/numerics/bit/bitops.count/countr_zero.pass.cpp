//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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

// UNSUPPORTED: c++98, c++03

// template <class T>
//   constexpr int countr_zero(T x) noexcept;

// Returns: The number of consecutive 0 bits, starting from the most significant bit.
//   [ Note: Returns N if x == 0. ]
//
// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <hip/std/bit>
#include <hip/std/cstdint>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

class A{};
enum       E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

template <typename T>
__host__ __device__ constexpr bool constexpr_test()
{
	return hip::std::countr_zero(T(0)) == hip::std::numeric_limits<T>::digits
	   &&  hip::std::countr_zero(T(1)) == 0
	   &&  hip::std::countr_zero(T(2)) == 1
	   &&  hip::std::countr_zero(T(3)) == 0
	   &&  hip::std::countr_zero(T(4)) == 2
	   &&  hip::std::countr_zero(T(5)) == 0
	   &&  hip::std::countr_zero(T(6)) == 1
	   &&  hip::std::countr_zero(T(7)) == 0
	   &&  hip::std::countr_zero(T(8)) == 3
	   &&  hip::std::countr_zero(T(9)) == 0
	   &&  hip::std::countr_zero(hip::std::numeric_limits<T>::max()) == 0
	  ;
}


template <typename T>
__host__ __device__ void runtime_test()
{
	ASSERT_SAME_TYPE(int, decltype(hip::std::countr_zero(T(0))));
	ASSERT_NOEXCEPT(               hip::std::countr_zero(T(0)));

	assert( hip::std::countr_zero(T(121)) == 0);
	assert( hip::std::countr_zero(T(122)) == 1);
	assert( hip::std::countr_zero(T(123)) == 0);
	assert( hip::std::countr_zero(T(124)) == 2);
	assert( hip::std::countr_zero(T(125)) == 0);
	assert( hip::std::countr_zero(T(126)) == 1);
	assert( hip::std::countr_zero(T(127)) == 0);
	assert( hip::std::countr_zero(T(128)) == 7);
	assert( hip::std::countr_zero(T(129)) == 0);
	assert( hip::std::countr_zero(T(130)) == 1);
}

int main(int, char **)
{
	static_assert(constexpr_test<unsigned char>(),      "");
	static_assert(constexpr_test<unsigned short>(),     "");
	static_assert(constexpr_test<unsigned>(),           "");
	static_assert(constexpr_test<unsigned long>(),      "");
	static_assert(constexpr_test<unsigned long long>(), "");

	static_assert(constexpr_test<uint8_t>(),   "");
	static_assert(constexpr_test<uint16_t>(),  "");
	static_assert(constexpr_test<uint32_t>(),  "");
	static_assert(constexpr_test<uint64_t>(),  "");
	static_assert(constexpr_test<size_t>(),    "");
	static_assert(constexpr_test<uintmax_t>(), "");
	static_assert(constexpr_test<uintptr_t>(), "");

#ifndef _LIBCUDACXX_HAS_NO_INT128
	static_assert(constexpr_test<__uint128_t>(),        "");
#endif

	runtime_test<unsigned char>();
	runtime_test<unsigned>();
	runtime_test<unsigned short>();
	runtime_test<unsigned long>();
	runtime_test<unsigned long long>();

	runtime_test<uint8_t>();
	runtime_test<uint16_t>();
	runtime_test<uint32_t>();
	runtime_test<uint64_t>();
	runtime_test<size_t>();
	runtime_test<uintmax_t>();
	runtime_test<uintptr_t>();

#ifndef _LIBCUDACXX_HAS_NO_INT128
	runtime_test<__uint128_t>();

	{
	__uint128_t val = 128;

	val <<= 32;
	assert( hip::std::countr_zero(val-1) ==  0);
	assert( hip::std::countr_zero(val)   == 39);
	assert( hip::std::countr_zero(val+1) ==  0);
	val <<= 2;
	assert( hip::std::countr_zero(val-1) ==  0);
	assert( hip::std::countr_zero(val)   == 41);
	assert( hip::std::countr_zero(val+1) ==  0);
	val <<= 3;
	assert( hip::std::countr_zero(val-1) ==  0);
	assert( hip::std::countr_zero(val)   == 44);
	assert( hip::std::countr_zero(val+1) ==  0);
	}
#endif

	return 0;
}
