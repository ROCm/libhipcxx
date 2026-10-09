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
  :description: API reference for cuda::is_floating_point, which tells whether a type is a standard or extended floating-point type in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, is_floating_point, type traits, floating point, extended floating-point types

.. _libcudacxx-extended-api-type_traits-is_floating_point:

``cuda::is_floating_point``
===========================

This page documents ``cuda::is_floating_point``, which tells whether a type is a floating-point type.

.. code:: cpp

   namespace cuda {

   template <class T>
   inline constexpr bool is_floating_point_v = __ implementation defined __;

   template <class T>
   using is_floating_point = cuda::std::bool_constant<is_floating_point_v<T>>;

   } // namespace cuda

Tells whether a type is a floating point type, including implementation defined extended floating point types.
Users are allowed to specialize the variable template for their own types, but libhipcxx does not provide support for any issues arising from that.
