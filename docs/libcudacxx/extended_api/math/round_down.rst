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
  :description: API reference for cuda::round_down, which rounds an integer value down to the largest multiple of a given base in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, round_down, floor rounding, integer math, multiple, enumerator

.. _libcudacxx-extended-api-math-round-down:

``cuda::round_down``
====================

This page documents ``cuda::round_down``, which rounds an integral value down to the largest multiple of a given base.

Defined in the ``<cuda/cmath>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T, typename U>
   [[nodiscard]] __host__ __device__ constexpr
   cuda::std::common_type_t<T, U> round_down(T value, U base_multiple) noexcept;

   } // namespace cuda

The function computes the round down to the largest multiple of an integral or enumerator value :math:`floor(\frac{value}{base\_multiple}) * base\_multiple`.

**Parameters**

- ``value``: The value to be rounded down.
- ``base_multiple``: The base multiple to which the value rounds down.

**Return value**

``value`` rounded down to the largest multiple of ``base_multiple`` less than or equal to ``value``. If ``value`` is already a multiple of ``base_multiple``, returns ``value``.

**Constraints**

``T`` and ``U`` are integer types or enumerators.

**Preconditions**

- ``value >= 0``
- ``base_multiple > 0``

**Performance considerations**

- The function performs a truncation division followed by a multiplication. It provides better performance than ``(value / base_multiple) * base_multiple`` when the common type is a signed integer.

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cstdio>

    __global__ void round_up_kernel() {
        int      value    = 7;
        unsigned multiple = 3;
        printf("%d\n", cuda::round_down(value, multiple)); // print "6"
    }

    int main() {
        round_up_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/cxGYfMGna>`__
