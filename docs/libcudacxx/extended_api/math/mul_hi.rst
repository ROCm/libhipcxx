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
  :description: API reference for cuda::mul_hi, which computes the most significant half of the product of two integers in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, mul_hi, multiplication, high bits, integer math

.. _libcudacxx-extended-api-math-mul-hi:

``cuda::mul_hi``
================

This page documents ``cuda::mul_hi``, which computes the most significant half of the product of two integers.

Defined in ``<cuda/cmath>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T mul_hi(T lhs, T rhs) noexcept;

   } // namespace cuda

Computes the most significant half of the bits of the product of two non-negative integers ``lhs`` and ``rhs``.

**Parameters**

- ``lhs``: First multiplicand.
- ``rhs``: Second multiplicand.

**Return value**

- The most significant half of ``lhs * rhs`` returned as ``T``.

**Constraints**

- ``T`` is an integer type.

**Remarks**

- Uses ``__mulhi``, ``__umulhi``, ``__mul64hi``, ``__umul64hi`` intrinsics on device when available.
- Uses ``__mulh``, ``__umulh`` intrinsics on Windows host code when available.
- Uses a double-width intermediate type when possible.
- Relies on a manual decomposition fallback when 128-bit intermediates are unavailable for 64-bit integers.

Example
-------

.. code:: cpp

   #include <cuda/cmath>
   #include <cuda/std/cassert>
   #include <cuda/std/cstdint>

   __global__ void mul_hi_kernel()
   {
       uint32_t lhs       = 0xABCD1234;
       uint32_t rhs       = 1 << 16; // 2^16
       uint32_t high_half = cuda::mul_hi(lhs, rhs);
       assert(high_half == 0xAB);
   }

   int main()
   {
       mul_hi_kernel<<<1, 1>>>();
       hipDeviceSynchronize();
       return 0;
   }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/64r6zT9Wq>`__
