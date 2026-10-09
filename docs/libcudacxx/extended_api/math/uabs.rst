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
  :description: API reference for cuda::uabs, which computes the absolute value of an integer as an unsigned type in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, uabs, absolute value, unsigned, integer math

.. _libcudacxx-extended-api-math-uabs:

``cuda::uabs``
====================================

This page documents ``cuda::uabs``, which computes the absolute value of an integer and returns it as an unsigned type.

Defined in the ``<cuda/cmath>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   cuda::std::make_unsigned_t<T> uabs(T value) noexcept;

   } // namespace cuda

The function computes the absolute value of the input value. The result is returned as an unsigned integer type of the same size as the input value. In comparison to the standard ``abs`` function, the ``uabs`` eliminates the undefined behaviour when a signed ``T_MIN`` is passed as an input.

**Parameters**

- ``value``: The input value.

**Return value**

- The unsigned absolute value of the input value.

**Constraints**

- ``T`` is an integer type.

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void uabs_kernel() {
        using cuda::std::numeric_limits;

        assert(cuda::uabs(20) == 20u);
        assert(cuda::uabs(-32) == 32u);
        assert(cuda::uabs(numeric_limits<int>::max()) == static_cast<unsigned>(numeric_limits<int>::max()));
        assert(cuda::uabs(numeric_limits<int>::min()) == static_cast<unsigned>(numeric_limits<int>::max()) + 1);
    }

    int main() {
        uabs_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/KEoYfq53G>`__
