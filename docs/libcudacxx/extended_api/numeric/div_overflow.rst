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
  :description: API reference for cuda::div_overflow, which performs integer division with overflow checking in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, div_overflow, overflow, integer arithmetic, numeric, overflow_result

.. _libcudacxx-extended-api-numeric-div_overflow:

``cuda::div_overflow``
======================

This page documents ``cuda::div_overflow``, which performs integer division with overflow checking.

Defined in ``<cuda/numeric>`` header.

.. code:: cpp

   namespace cuda {

   template <class T>
   struct overflow_result;

   template <class Result = /*unspecified*/, class Lhs, class Rhs>
   [[nodiscard]] __host__ __device__ constexpr
   overflow_result</*see-below*/> div_overflow(Lhs lhs, Rhs rhs) noexcept; // (1)

   template <class Result, class Lhs, class Rhs>
   [[nodiscard]] __host__ __device__ constexpr
   bool div_overflow(Result& result, Lhs lhs, Rhs rhs) noexcept; // (2)

   } // namespace cuda

The function ``cuda::div_overflow`` performs integer division of ``lhs`` by ``rhs`` with overflow and error detection. The result is the same as if the operands were first promoted to an infinite precision signed type, divided, and the result truncated to the type of the return value.

**Parameters**

- ``result``: Receives the quotient when no overflow is detected (2).
- ``lhs``: The dividend (1, 2).
- ``rhs``: The divisor (1, 2).

**Return value**

1. Returns an :ref:`overflow_result <libcudacxx-extended-api-numeric-overflow_result>` object containing the computed quotient and a boolean flag indicating whether an overflow or underflow occurred. If the ``Result`` type is specified, it will be used as the type of the result, otherwise the common type of ``Lhs`` and ``Rhs`` is used.
2. Returns ``true`` if an overflow or underflow occurred. When ``false`` is returned, the computed quotient is stored in ``result``.

**Preconditions**

- ``rhs != 0``

**Constraints**

- ``Result``, ``Lhs``, and ``Rhs`` must be `integer types <https://eel.is/c++draft/basic.fundamental#1>`_.

**Notes**

- For signed types, ``numeric_limits<Lhs>::min()`` divided by ``-1`` triggers overflow when ``Result`` matches the operand types.

**Performance considerations**

- No overflow checking if ``Lhs / Rhs`` is always  representable with the ``Result`` type.
- The computation with unsigned types is faster than signed types.

Example
-------

.. code:: cpp

    #include <cuda/numeric>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void kernel()
    {
        constexpr auto int_min = cuda::std::numeric_limits<int>::min();

        // cuda::div_overflow(lhs, rhs) returning the common type of the operands
        if (auto result = cuda::div_overflow(-1, int_min))
        {
            assert(result.overflow);
        }

        // cuda::div_overflow<Result>(lhs, rhs) with an explicitly wider result type
        auto wide = cuda::div_overflow<long long>(-1, int_min);
        assert(!wide.overflow);
        assert(wide.value == 0);

        unsigned quotient{};
        // cuda::div_overflow(result, lhs, rhs) with bool return type
        bool overflow = cuda::div_overflow(quotient, 10u, 2u);
        assert(!overflow);
        assert(quotient == 5u);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        hipDeviceSynchronize();
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/dYG3dWss5>`_
