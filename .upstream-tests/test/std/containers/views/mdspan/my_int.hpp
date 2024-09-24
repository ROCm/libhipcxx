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

#ifndef _MY_INT_HPP
#define _MY_INT_HPP

struct my_int_non_convertible;

struct my_int
{
    int _val;

    __host__ __device__ my_int( my_int_non_convertible ) noexcept;
    __host__ __device__ constexpr my_int( int val ) : _val( val ){};
    __host__ __device__ constexpr operator int() const noexcept { return _val; }
};

template <> struct hip::std::is_integral<my_int> : hip::std::true_type {};

// Wrapper type that's not implicitly convertible

struct my_int_non_convertible
{
    my_int _val;

    my_int_non_convertible();
    __host__ __device__ my_int_non_convertible( my_int val ) : _val( val ){};
    __host__ __device__ operator my_int() const noexcept { return _val; }
};

__host__ __device__ my_int::my_int( my_int_non_convertible ) noexcept {}

template <> struct hip::std::is_integral<my_int_non_convertible> : hip::std::true_type {};

// Wrapper type that's not nothrow-constructible

struct my_int_non_nothrow_constructible
{
    int _val;

    my_int_non_nothrow_constructible();
    __host__ __device__ my_int_non_nothrow_constructible( int val ) : _val( val ){};
    __host__ __device__ operator int() const { return _val; }
};

template <> struct hip::std::is_integral<my_int_non_nothrow_constructible> : hip::std::true_type {};

#endif
