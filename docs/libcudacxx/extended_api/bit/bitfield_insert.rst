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
  :description: API reference for cuda::bitfield_insert, which inserts a bitfield from one unsigned integer value into another at a specified position in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, bitfield_insert, bit manipulation, unsigned integer, bit field

.. _libcudacxx-extended-api-bit-bitfield_insert:

``cuda::bitfield_insert``
=========================

This page documents ``cuda::bitfield_insert``, which inserts the lower bits of a source value into a destination value at a specified bit position and width.

Defined in the ``<cuda/bit>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T bitfield_insert(T dest, T source, int start, int width) noexcept;

   } // namespace cuda

The function extracts the lower bitfield of size ``width`` from ``source`` and inserts it into ``dest`` at position ``start``.

**Parameters**

- ``dest``: The value to insert the bitfield into.
- ``source``: The value from which to extract the bitfield.
- ``start``: Initial position of the bitfield.
- ``width``: Width of the bitfield.

**Return value**

``((source << start) & mask) | (dest & ~mask)``, where ``mask`` is a bitmask of width ``width`` at position ``start``.

**Constraints**

``T`` is an unsigned integer type.

**Preconditions**

- ``start >= 0 && start <= num_bits(T)``.
- ``width >= 0 && width <= num_bits(T)``.
- ``start + width <= num_bits(T)``.

**Performance considerations**

..
   The function performs the following operations in CUDA for ``uint8_t``, ``uint16_t``, ``uint32_t``:

   - ``SM < 70``: ``BFI``.
   - ``SM >= 70``: ``BMSK``, bitwise operation x5.

.. note::

   When the input values are run-time values that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations, using the function could not be optimal.

..
   .. note::

      GCC <= 8 uses a slow path with more instructions even in CUDA.

Example
-------

.. code:: cpp

    #include <cuda/bit>
    #include <cuda/std/cassert>

    __global__ void bitfield_insert_kernel() {
        assert(cuda::bitfield_insert(0u, 0xFFFFu, 0, 4) == 0b1111);
        assert(cuda::bitfield_insert(0u, 0xFFFFu, 3, 4) == 0b1111000);
        assert(cuda::bitfield_insert(1u, 0xFFFFu, 3, 4) == 0b1111001);
    }

    int main() {
        bitfield_insert_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/4Thzz516M>`__
