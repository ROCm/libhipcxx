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
  :description: API reference for cuda::is_trivially_copyable, which tells whether a type, including extended floating-point vector types, can be copied bytewise in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, is_trivially_copyable, type traits, trivially copyable, extended floating-point

.. _libcudacxx-extended-api-type_traits-is_trivially_copyable:

``cuda::is_trivially_copyable``
=======================================

This page documents ``cuda::is_trivially_copyable``, which tells whether a type can be copied by copying its underlying bytes.

Defined in the ``<cuda/type_traits>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T>
   constexpr bool is_trivially_copyable_v = /* see below */;

   template <typename T>
   using is_trivially_copyable = cuda::std::bool_constant<cuda::std::is_trivially_copyable_v<T>>;

   } // namespace cuda

``cuda::is_trivially_copyable_v`` trait evaluates if a type can be copied by copying its underlying bytes.
It extends ``cuda::std::is_trivially_copyable`` to also recognize extended floating-point vector types as trivially copyable.

The trait is true when ``T`` is any of the following:

- A type for which ``cuda::std::is_trivially_copyable_v<T>`` is true.
- An extended floating-point vector type, for example ``__half2``, ``__nv_bfloat162``.

The trait also propagates through composite types:

- C-style arrays: ``T[N]`` and ``T[]`` are trivially copyable when ``T`` is.
- ``cuda::std::array<T, N>``: trivially copyable when ``T`` is.
- ``cuda::std::pair<T1, T2>``: trivially copyable when both ``T1`` and ``T2`` are.
- ``cuda::std::tuple<Ts...>``: trivially copyable when all ``Ts...`` are.
- ``cuda::std::complex<T>``: trivially copyable when ``T`` is.
- ``cuda::complex<T>``: trivially copyable when ``T`` is.
- `Aggregates <https://en.cppreference.com/cpp/language/aggregate_initialization>`__: trivially copyable when all their members are.

  - On MSVC, recursive data-member inspection is not supported beyond the first level.

``const`` qualification is handled transparently, while ``volatile`` is compiler dependent.

Examples
--------

.. code:: cpp

   #include <cuda/type_traits>
   #include <cuda/std/array>
   #include <cuda/std/tuple>
   #include <cuda/std/utility>

   #include <cuda_fp16.h>

   // Standard trivially copyable types
   static_assert(cuda::is_trivially_copyable_v<int>);
   static_assert(cuda::is_trivially_copyable_v<float>);

   // Extended floating-point types
   static_assert(cuda::is_trivially_copyable_v<__half>);
   static_assert(cuda::is_trivially_copyable_v<__nv_bfloat16>);
   static_assert(cuda::is_trivially_copyable_v<__half2>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::complex<__half>>);
   static_assert(cuda::is_trivially_copyable_v<cuda::complex<__half>>);

   // Composite types containing extended floating-point types
   static_assert(cuda::is_trivially_copyable_v<__half[4]>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::array<__half2, 4>>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::pair<__half2, __half>>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::tuple<__half, __half2>>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::pair<__half2, int>>);


..
   `See it on Godbolt 🔗 <https://godbolt.org/z/PqccjfEv6>`__
