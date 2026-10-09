..
    MIT License

    Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

.. meta::
  :description: API reference for cuda::mul_overflow, which performs multiplication with overflow checking in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, mul_overflow, overflow, integer arithmetic, numeric, overflow_result

.. _libcudacxx-extended-api-numeric-mul_overflow:

``cuda::mul_overflow``
======================

This page documents ``cuda::mul_overflow``, which performs multiplication with overflow checking.

Defined in ``<cuda/numeric>`` header.

.. code:: cpp

   namespace cuda {

   template <class T>
   struct overflow_result;

   template <class Result = /*unspecified*/, class Lhs, class Rhs>
   [[nodiscard]] __host__ __device__ constexpr
   overflow_result</*see-below*/> mul_overflow(Lhs lhs, Rhs rhs) noexcept; // (1)

   template <class Result, class Lhs, class Rhs>
   [[nodiscard]] __host__ __device__ constexpr
   bool mul_overflow(Result& result, Lhs lhs, Rhs rhs) noexcept; // (2)

   } // namespace cuda

The function ``cuda::mul_overflow`` performs multiplication of two values ``lhs`` and ``rhs`` with overflow checking. The result is the same as if the operands were first promoted to an infinite precision signed type, multiplied together and the result truncated to the type of the return value.

**Parameters**

- ``result``: The result of the multiplication (2).
- ``lhs``: The left-hand side operand (1, 2).
- ``rhs``: The right-hand side operand (1, 2).

**Return value**

1. Returns an :ref:`overflow_result <libcudacxx-extended-api-numeric-overflow_result>` object  containing the result of the multiplication and a boolean indicating whether an overflow or underflow occurred. If the ``Result`` type is specified, it will be used as the type of the result, otherwise the common type of ``Lhs`` and ``Rhs`` is used.
2. Returns ``true`` if an overflow or underflow occurred, ``false`` otherwise.

**Constraints**

- ``Result``, ``Lhs``, and ``Rhs`` must be `integer types <https://eel.is/c++draft/basic.fundamental#1>`_.

**Performance considerations**

- No overflow checking is required if ``Lhs +  Rhs`` is always representable with the ``Result`` type.
- Computation is generally faster when ``Lhs``, ``Rhs``, and ``Result`` have the same signedness.
- Unsigned computations are generally faster than signed computations.

Example
-------

.. code:: cpp

    #include <cuda/numeric>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void kernel()
    {
        constexpr auto int_max = cuda::std::numeric_limits<int>::max();
        constexpr auto int_min = cuda::std::numeric_limits<int>::min();

        // cuda::mul_overflow(lhs, rhs) returning common type of lhs and rhs
        // 'result' is evaluated to true if an overflow occurred, false otherwise
        if (auto result1 = cuda::mul_overflow(-1, int_min))
        {
            assert(result1.value == int_min);
        }

        // cuda::mul_overflow<Result>(lhs, rhs) with explicit return type
        auto [result2, overflow2] = cuda::mul_overflow<long long>(-1, int_min);
        assert(!overflow2); // no overflow
        assert(result2 == static_cast<long long>(int_min) * (-1ll));

        unsigned result3{};
        // cuda::mul_overflow(result, lhs, rhs) with bool return type
        if (!cuda::mul_overflow(result3, 2, int_max))
        {
            assert(result3 == static_cast<unsigned>(int_max) * 2u);
        }
    }

    int main()
    {
        kernel<<<1, 1>>>();
        hipDeviceSynchronize();
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/hTME8v6rK>`_
