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
  :description: API reference for cuda::narrow and cuda::narrow_cast, which perform checked and unchecked narrowing conversions in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, narrow, narrow_cast, narrowing_error, conversion, numeric

.. _libcudacxx-extended-api-numeric-narrow:

``cuda::narrow``
=====================

This page documents ``cuda::narrow`` and ``cuda::narrow_cast``, which perform checked and unchecked narrowing conversions.

.. code:: cpp

   struct narrowing_error;

   template <typename To, typename From>
   [[nodiscard]] constexpr
   To narrow(From from);

   template <typename To, typename From>
   [[nodiscard]] constexpr
   To narrow_cast(From&& __from) noexcept;


Both functions use a ``static_cast`` to cast the value ``from`` to type ``To``.
``From`` needs to be convertible to ``To``, and implement ``operator!=``.
``cuda::narrow`` additionally checks whether the value has changed,
and if so, throws ``cuda::narrowing_error`` in host code and traps in device code.
In this case, ``To`` additionally needs to be convertible to ``From``.
``cuda::narrow_cast`` does not perform such a check (it's a plain cast) and is just intended to show
that narrowing and a potential change of the value is intended.
The functions are modelled after ``gsl::narrow`` and  ``gsl::narrow_cast``.
See also the C++ Core Guidelines
`ES.46 <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-narrowing>`_ and
`ES.49 <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-casts-named>`_.


Example
-------

.. code:: cpp

    #include <cuda/numeric>

    __global__ void kernel(size_t n) {
        unsigned int r1 = cuda::narrow<unsigned int>(n); // traps

        unsigned int r2 = cuda::narrow_cast<unsigned int>(n); // truncation of value is intended
    }

    void host() {
        unsigned char r1 = cuda::narrow<unsigned char>( 200); // ok
        unsigned char r2 = cuda::narrow<unsigned char>( 300); // throws narrowing_error
        unsigned int  r3 = cuda::narrow<unsigned int >(-100); // throws narrowing_error

        unsigned char r4 = cuda::narrow_cast<unsigned char>(300); // truncation of value is intended

        kernel<<<1, 1>>>(2LL << 35); // size larger than unsigned int
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/ahcqv6joY>`_
