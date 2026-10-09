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
  :description: API reference for cuda::bit_reverse, which reverses the order of bits in an unsigned integer value in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, bit_reverse, bit manipulation, unsigned integer, bit reversal

.. _libcudacxx-extended-api-bit-bit_reverse:

``cuda::bit_reverse``
=====================

This page documents ``cuda::bit_reverse``, which reverses the order of bits in an unsigned integer value.

Defined in the ``<cuda/bit>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T bit_reverse(T value) noexcept;

   } // namespace cuda

The function reverses the order of bits in a value.

**Parameters**

``value``: Input value.

**Return value**

Value with reversed bits.

**Constraints**

``T`` is an unsigned integer type.

**Performance considerations**

In host code the function uses ``__builtin_bitreverse<N>`` with clang.

..
   The function performs the following operations:

   - Device:

     - ``uint8_t`` ``uint16_t``: ``PRMT``, ``BREV``
     - ``uint32_t``: ``BREV``
     - ``uint64_t``: ``BREV`` x2, ``MOV`` x2
     - ``uint128_t``: ``BREV`` x4, ``MOV`` x4

   - Host: ``__builtin_bitreverse<N>`` with clang

.. note::

   Using the function could not be optimal when the input is a run-time value that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations.

..
   .. note::

      GCC <= 8 uses a slow path with more instructions even in device code.

Example
-------

.. code:: cpp

    #include <cuda/bit>
    #include <cuda/std/cassert>

    __global__ void bit_reverse_kernel() {
        assert(cuda::bit_reverse(0x0000FFFFu) == 0xFFFF0000u);
        assert(cuda::bit_reverse(uint8_t{0b00001011}) == uint8_t{0b11010000});
    }

    int main() {
        bit_reverse_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/nW6qe5fT4>`__
