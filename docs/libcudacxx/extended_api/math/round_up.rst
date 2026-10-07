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
  :description: API reference for cuda::round_up, which rounds an integer value up to the smallest multiple of a given base in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, round_up, ceiling rounding, integer math, multiple, enumerator

.. _libcudacxx-extended-api-math-round-up:

``cuda::round_up``
==================

This page documents ``cuda::round_up``, which rounds an integral value up to the smallest multiple of a given base.

.. code:: cpp

   template <typename T, typename U>
   [[nodiscard]] __host__ __device__ inline constexpr
   cuda::std::common_type_t<T, U> round_up(T value, U base_multiple) noexcept;

The function computes the round up to the smallest multiple of an integral or enumerator value :math:`ceil(\frac{value}{base\_multiple}) * base\_multiple`.

**Parameters**

- ``value``: The value to be rounded up.
- ``base_multiple``: The base multiple to which the value rounds up.

**Return value**

``value`` rounded up to the smallest multiple of ``base_multiple`` greater than or equal to ``value``. If ``value`` is already a multiple of ``base_multiple``, return ``value``.

.. note::

   The result can overflow if ``ceil(value / base_multiple) * base_multiple`` exceeds the maximum value of the common type of ``value`` and ``base_multiple``. The condition is checked in debug mode.

**Constraints**

``T`` and ``U`` are integer types or enumerators.

**Preconditions**

- ``value >= 0``
- ``base_multiple > 0``

**Performance considerations**

- The function performs a ceiling division (``cuda::ceil_div()``) followed by a multiplication

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cstdio>

    __global__ void round_up_kernel() {
        int      value    = 7;
        unsigned multiple = 3;
        printf("%d\n", cuda::round_up(value, multiple)); // print "9"
    }

    int main() {
        round_up_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/9vcxo3d8j>`_
