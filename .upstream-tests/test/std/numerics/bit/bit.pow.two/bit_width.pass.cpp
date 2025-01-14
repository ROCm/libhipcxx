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
//   constexpr T log2p1(T x) noexcept;

// If x == 0, 0; otherwise one plus the base-2 logarithm of x, with any fractional part discarded.

// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <hip/std/bit>
#include <hip/std/cstdint>
#include <hip/std/cassert>

#include "test_macros.h"

class A{};
enum       E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

template <typename T>
__host__ __device__ constexpr bool constexpr_test()
{
	return hip::std::bit_width(T(0)) == T(0)
	   &&  hip::std::bit_width(T(1)) == T(1)
	   &&  hip::std::bit_width(T(2)) == T(2)
	   &&  hip::std::bit_width(T(3)) == T(2)
	   &&  hip::std::bit_width(T(4)) == T(3)
	   &&  hip::std::bit_width(T(5)) == T(3)
	   &&  hip::std::bit_width(T(6)) == T(3)
	   &&  hip::std::bit_width(T(7)) == T(3)
	   &&  hip::std::bit_width(T(8)) == T(4)
	   &&  hip::std::bit_width(T(9)) == T(4)
	   ;
}


template <typename T>
__host__ __device__ void runtime_test()
{
	ASSERT_SAME_TYPE(T, decltype(hip::std::bit_width(T(0))));
	ASSERT_NOEXCEPT(             hip::std::bit_width(T(0)));

	assert( hip::std::bit_width(T(0)) == T(0));
	assert( hip::std::bit_width(T(1)) == T(1));
	assert( hip::std::bit_width(T(2)) == T(2));
	assert( hip::std::bit_width(T(3)) == T(2));
	assert( hip::std::bit_width(T(4)) == T(3));
	assert( hip::std::bit_width(T(5)) == T(3));
	assert( hip::std::bit_width(T(6)) == T(3));
	assert( hip::std::bit_width(T(7)) == T(3));
	assert( hip::std::bit_width(T(8)) == T(4));
	assert( hip::std::bit_width(T(9)) == T(4));


	assert( hip::std::bit_width(T(121)) == T(7));
	assert( hip::std::bit_width(T(122)) == T(7));
	assert( hip::std::bit_width(T(123)) == T(7));
	assert( hip::std::bit_width(T(124)) == T(7));
	assert( hip::std::bit_width(T(125)) == T(7));
	assert( hip::std::bit_width(T(126)) == T(7));
	assert( hip::std::bit_width(T(127)) == T(7));
	assert( hip::std::bit_width(T(128)) == T(8));
	assert( hip::std::bit_width(T(129)) == T(8));
	assert( hip::std::bit_width(T(130)) == T(8));
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
	assert( hip::std::bit_width(val-1) == 39);
	assert( hip::std::bit_width(val)   == 40);
	assert( hip::std::bit_width(val+1) == 40);
	val <<= 2;
	assert( hip::std::bit_width(val-1) == 41);
	assert( hip::std::bit_width(val)   == 42);
	assert( hip::std::bit_width(val+1) == 42);
	val <<= 3;
	assert( hip::std::bit_width(val-1) == 44);
	assert( hip::std::bit_width(val)   == 45);
	assert( hip::std::bit_width(val+1) == 45);
	}
#endif

	return 0;
}
