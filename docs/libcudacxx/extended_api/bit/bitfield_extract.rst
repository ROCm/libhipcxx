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
  :description: API reference for cuda::bitfield_extract, which extracts a bitfield from an unsigned integer value and returns it in the lower bits in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, bitfield_extract, bit manipulation, unsigned integer, BFE

.. _libcudacxx-extended-api-bit-bitfield_extract:

``cuda::bitfield_extract``
==========================

This page documents ``cuda::bitfield_extract``, which extracts a bitfield of a specified width from a value and returns it in the lower bits.

.. code:: cpp

   template <typename T>
   [[nodiscard]] constexpr T
   bitfield_extract(T value, int start, int width) noexcept;

The function extracts a bitfield from a value and returns it in the lower bits.
``bitfield_extract()`` computes ``(value >> start) & bitfield``, where ``bitfield`` is a sequence of bits of width ``width``.

**Parameters**

- ``value``: The value to apply the bitfield.
- ``start``: Initial position of the bitfield.
- ``width``: Width of the bitfield.

**Return value**

``(value >> start) & bitfield``.

**Constraints**

``T`` is an unsigned integer type.

**Preconditions**

- ``start >= 0 && start <= num_bits(T)``
- ``width >= 0 && width <= num_bits(T)``
- ``start + width <= num_bits(T)``

**Performance considerations**

..
   The function performs the following operations in CUDA for ``uint8_t``, ``uint16_t``, ``uint32_t``:

   - ``SM < 70``: ``BFE``
   - ``SM >= 70``: ``BMSK``, bitwise operation x2

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

    __global__ void bitfield_insert_kernel() {
        assert(cuda::bitfield_extract(~0u, 0, 4) == 0b1111);
        assert(cuda::bitfield_extract(0b1011000, 3, 4) == 0b1011);
    }

    int main() {
        bitfield_insert_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/WvqfG9nbP>`_
