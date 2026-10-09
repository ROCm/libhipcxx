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
  :description: API reference for cuda::ipow, which computes an integer raised to an integer power in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, ipow, integer power, exponentiation, integer math

.. _libcudacxx-extended-api-math-ipow:

``cuda::ipow``
====================================

This page documents ``cuda::ipow``, which computes an integer base raised to an integer exponent.

Defined in the ``<cuda/cmath>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T, typename E>
   [[nodiscard]] __host__ __device__ constexpr
   T ipow(T base, E exp) noexcept;

   } // namespace cuda

The function computes the integer ``base`` raised to the power of ``exp``.

**Parameters**

- ``base``: The base value.
- ``exp``: The exponent value.

**Return value**

- The result of raising ``base`` to the power of ``exp``. If ``exp`` is negative, the result is 0.

**Constraints**

- ``T`` is an integer type.
- ``E`` is an integer type.

**Preconditions**

- if ``base`` is 0, then ``exp`` must be non-negative.

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void ipow_kernel() {
        assert(cuda::ipow(0, 0) == 1);
        assert(cuda::ipow(2, 2) == 4);
        assert(cuda::ipow(99, 1) == 99);
        assert(cuda::ipow(4, 7) == 16384);
        assert(cuda::ipow(-1, 3) == -1);
        assert(cuda::ipow(23, -1) == 0);
    }

    int main() {
        ipow_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/TMacWvz8v>`__
