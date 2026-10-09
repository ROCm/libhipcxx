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
  :description: API reference for cuda::isqrt, which computes the integer square root of a value rounded down in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, isqrt, integer square root, integer math

.. _libcudacxx-extended-api-math-isqrt:

``cuda::isqrt``
====================================

This page documents ``cuda::isqrt``, which computes the integer square root of a value, rounded down.

Defined in the ``<cuda/cmath>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T isqrt(T value) noexcept;

   } // namespace cuda

The function computes the integer square root of the input value rounded down.

**Parameters**

- ``value``: The input value.

**Return value**

- The square root value of the input value rounded down.

**Constraints**

- ``T`` is an integer type.

**Preconditions**

- ``value`` is non-negative.

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void isqrt_kernel() {
        assert(cuda::isqrt(1) == 1);
        assert(cuda::isqrt(4) == 2);
        assert(cuda::isqrt(42) == 6);
        assert(cuda::isqrt(99) == 9);
        assert(cuda::isqrt(100) == 10);
    }

    int main() {
        isqrt_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/xPcj35dq6>`__
