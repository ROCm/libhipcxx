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
  :description: API reference for cuda::bitmask, which generates a bitmask of a specified width starting at a given bit position in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, bitmask, bit manipulation, unsigned integer, BMSK

.. _libcudacxx-extended-api-bit-bitmask:

``cuda::bitmask``
=================

This page documents ``cuda::bitmask``, which generates an integer bitmask of a specified width starting at a given bit position.

.. code:: cpp

   template <typename T = uint32_t>
   [[nodiscard]] constexpr T
   bitmask(int start, int width) noexcept;

The function generates a bitmask of size ``width`` starting at position ``start``.

**Parameters**

- ``start``: Starting position of the bitmask.
- ``width``: Width of the bitmask.

**Return value**

Bitmask of size ``width`` starting at ``start``.

**Constraints**

``T`` is an unsigned integral type.

**Preconditions**

- ``start >= 0 && start <= num_bits(T)``
- ``width >= 0 && width <= num_bits(T)``
- ``start + width <= num_bits(T)``

**Performance considerations**

..
   The function performs the following operations in device code:

   - ``uint8_t``, ``uint16_t``, ``uint32_t``: ``BMSK``
   - ``uint64_t``: ``SHL`` x4, ``ADD`` x2

.. note::

   When the input values are run-time values that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations, using the function could not be optimal.

..
   .. note::

      GCC <= 8 uses a slow path with more instructions even in CUDA

Example
-------

.. code:: cpp

    #include <cuda/bit>
    #include <cuda/std/cassert>
    #include <cuda/std/cstdint>

    __global__ void bitmask_kernel() {
        assert(cuda::bitmask(2, 4) == 0b111100u);
        assert(cuda::bitmask<uint64_t>(1, 3) == uint64_t{0b1110});
    }

    int main() {
        bitmask_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/PPqP8rTPd>`_
