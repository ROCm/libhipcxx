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
  :description: API reference for cuda::neg, which negates signed and unsigned integer values without warnings in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, neg, negation, integer math, unsigned

.. _libcudacxx-extended-api-math-neg:

``cuda::neg``
====================================

This page documents ``cuda::neg``, which computes the negation of a signed or unsigned integer value.

Defined in the ``<cuda/cmath>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T neg(T value) noexcept;

   } // namespace cuda

The function computes the negation of the input value accepting both signed and unsigned integer types. It doesn't emit any warnings for signed integer overflow and applying ``-`` to unsigned integer types.

**Parameters**

- ``value``: The input value.

**Return value**

- The negated value of the input value.

**Constraints**

- ``T`` is an integer type.

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void neg_kernel() {
        using cuda::std::numeric_limits;

        assert(cuda::neg(1) == -1);
        assert(cuda::neg(20) == -20);
        assert(cuda::neg(127u) == 4294967169u);
        assert(cuda::neg(-127) == 127);
        assert(cuda::neg(cuda::std::numeric_limits<int>::min()) == cuda::std::numeric_limits<int>::min());
    }

    int main() {
        neg_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/K3zcE9zqn>`__
