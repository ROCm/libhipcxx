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
  :description: API reference for cuda::is_power_of_two, cuda::next_power_of_two, and cuda::prev_power_of_two in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, is_power_of_two, next_power_of_two, prev_power_of_two, power of two, integer math

.. _libcudacxx-extended-api-math-pow2:

Power of Two Utilities
======================

This page documents ``cuda::is_power_of_two``, ``cuda::next_power_of_two``, and ``cuda::prev_power_of_two``, which test for and compute powers of two.

Defined in the ``<cuda/cmath>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   bool is_power_of_two(T value) noexcept;

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T next_power_of_two(T value) noexcept;

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T prev_power_of_two(T value) noexcept;

   } // namespace cuda

The functions provide utilities to determine if an integer value is a power of two, and to compute the next and previous power of two.

**Parameters**

- ``value``: The input value.

**Return value**

- ``is_power_of_two``: Return ``true`` if ``value`` is a power of two, ``false`` otherwise.
- ``next_power_of_two``: Return the smallest power of two greater than or equal to ``value``.
- ``prev_power_of_two``: Return the largest power of two less than or equal to ``value``.

**Constraints**

- ``T`` is an integer types. Contrary to ``cuda::std::has_single_bit``, ``cuda::std::bit_floor``, and ``cuda::std::bit_ceil``, ``T`` can be both signed and unsigned.

**Preconditions**

- ``value > 0``

**Performance considerations**

See :ref:`\<cuda/std/bit\> performance considerations <libcudacxx-standard-api-numerics-bit>`

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void pow2_kernel() {
        assert(!cuda::is_power_of_two(20));
        assert(cuda::next_power_of_two(20) == 32);
        assert(cuda::prev_power_of_two(20) == 16);
    }

    int main() {
        pow2_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/896Yx3vf8>`__
