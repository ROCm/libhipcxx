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
  :description: API reference for cuda::saturating_sub_overflow, which performs saturating subtraction with overflow checking in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, saturating_sub_overflow, saturation, overflow, integer arithmetic, numeric

.. _libcudacxx-extended-api-numeric-saturating_sub_overflow:

``cuda::saturating_sub_overflow``
=================================

This page documents ``cuda::saturating_sub_overflow``, which performs saturating subtraction with overflow checking.

Defined in ``<cuda/numeric>`` header.

.. code:: cpp

   namespace cuda {

   template <class T>
   struct overflow_result;

   template <class T>
   [[nodiscard]] __host__ __device__ constexpr
   overflow_result<T> saturating_sub_overflow(T lhs, T rhs) noexcept; // (1)

   template <class T>
   [[nodiscard]] __host__ __device__ constexpr
   bool saturating_sub_overflow(T& result, T lhs, T rhs) noexcept; // (2)

   } // namespace cuda

The function ``cuda::saturating_sub_overflow`` performs saturating subtraction of two values ``lhs`` and ``rhs`` with overflow detection.

**Parameters**

- ``result``: The result of the saturating subtraction. (2)
- ``lhs``: The left-hand side operand. (1, 2)
- ``rhs``: The right-hand side operand. (1, 2)

**Return value**

1. Returns an :ref:`overflow_result <libcudacxx-extended-api-numeric-overflow_result>` object containing the result of the saturating subtraction and a boolean flag indicating whether an overflow or underflow occurred.
2. Returns ``true`` if an overflow or underflow occurred, ``false`` otherwise.

**Constraints**

- ``T`` must be an `integer type <https://eel.is/c++draft/basic.fundamental#1>`_.

**Performance considerations**

- Functionality is implemented by correcting the ``cuda::sub_overflow`` result in case of overflow/underflow.
- Unsigned computations are generally faster than signed computations.

Example
-------

.. code:: cpp

    #include <cuda/numeric>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void kernel()
    {
        constexpr auto uint_max = cuda::std::numeric_limits<unsigned>::max();

        const auto result = cuda::saturating_sub_overflow(42, uint_max); // saturated
        assert(result.value == 0);
        assert(result.overflow);

        int value;
        if (cuda::saturating_sub_overflow(value, 42, 1024))
        {
            assert(false); // shouldn't be reached
        }
        assert(value == 982);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        hipDeviceSynchronize();
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/4cnTKPjbG>`_
