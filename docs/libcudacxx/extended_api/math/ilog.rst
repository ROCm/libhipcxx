..
    MIT License

    Copyright (c) 2024-2026 Advanced Micro Devices, Inc.

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
  :description: API reference for cuda::ilog2 and cuda::ilog10, which compute integer logarithms to base 2 and base 10 respectively in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, ilog2, ilog10, integer logarithm, math, ceil_ilog2

.. _libcudacxx-extended-api-math-ilog:

``cuda::ilog2`` and ``cuda::ilog10``
====================================

This page documents ``cuda::ilog2`` and ``cuda::ilog10``, which compute the integer logarithm to base 2 or base 10 of an integer value.

Defined in the ``<cuda/cmath>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   int ilog2(T value) noexcept;

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   int ceil_ilog2(T value) noexcept;

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   int ilog10(T value) noexcept;

   } // namespace cuda

The functions compute the logarithm to the base 2 and 10 of an integer value.

**Parameters**

``value``: The input value.

**Return value**

- ``ilog2``, ``ceil_ilog2``: The logarithm to the base 2, rounded down and up to the nearest integer respectively.
-  ``ilog10``: The logarithm to the 10, rounded down to the nearest integer.

**Constraints**

``T`` is an integer type.

**Preconditions**

``value > 0``

..
   **Performance considerations**

   The function performs the following operations in device code:

   - ``ilog2``: ``FLO``
   - ``ceil_ilog2``: ``FLO``, ``POPC``, ``ADD``, comparison
   - ``ilog10``: ``FLO``, ``FMUL``, ``F2I``, constant memory lookup, ``SEL`` + ``IADD`` only if ``T == uint32_t`` or ``T == __uint128_t``

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void ilog_kernel() {
        assert(cuda::ilog2(20) == 4);
        assert(cuda::ceil_ilog2(20) == 5);
        assert(cuda::ilog2(32) == 5);
        assert(cuda::ceil_ilog2(32) == 5);
        assert(cuda::ilog10(100) == 2);
        assert(cuda::ilog10(2000) == 3);
    }

    int main() {
        ilog_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/7W3WaGd3c>`__
