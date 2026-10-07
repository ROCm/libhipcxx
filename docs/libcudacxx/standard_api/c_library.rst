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
  :description: Documents the C standard library headers available in libhipcxx, including cassert, ccomplex, cfloat, climits, cmath, cstddef, cstdint, cstdlib, cstring, and ctime.
  :keywords: libhipcxx, ROCm, HIP, C++, C library, cstdint, cstddef, cmath, cassert

.. _libcudacxx-standard-api-c-compat:

C Library
=========

Any Standard C++ header not listed below is omitted.

.. list-table::
   :widths: 25 45 30 20
   :header-rows: 1

   * - **Header**
     - **Content**
     - **libhipcxx Availability**
     - **C++ Reference**

   * - ``<cuda/std/cassert>``
     - Lightweight assumption testing
     - libhipcxx 2.7
     - `\<cassert\> <https://en.cppreference.com/w/cpp/header/cassert>`_

   * - ``<cuda/std/ccomplex>``
     - C complex number arithmetic
     - libhipcxx 2.7
     - `\<ccomplex\> <https://en.cppreference.com/w/cpp/header/ccomplex>`_

   * - ``<cuda/std/cfloat>``
     - Type support library
     - libhipcxx 2.7
     - `\<cfloat\> <https://en.cppreference.com/w/cpp/header/cfloat>`_

   * - ``<cuda/std/cfloat>``
     - Limits of floating point types
     - libhipcxx 2.7
     - `\<cfloat\> <https://en.cppreference.com/w/cpp/header/cfloat>`_

   * - ``<cuda/std/climits>``
     - Limits of integral types
     - libhipcxx 2.7
     - `\<climits\> <https://en.cppreference.com/w/cpp/header/climits>`_

   * - ``<cuda/std/cmath>``
     - Common math functions
     - libhipcxx 2.7
     - `\<cmath\> <https://en.cppreference.com/w/cpp/header/cmath>`_

   * - ``<cuda/std/cstddef>``
     - Fundamental types
     - libhipcxx 2.7
     - `\<cstddef\> <https://en.cppreference.com/w/cpp/header/cstddef>`_

   * - ``<cuda/std/cstdint>``
     - Fundamental integer types
     - libhipcxx 2.7
     - `\<cstdint\> <https://en.cppreference.com/w/cpp/header/cstdint>`_

   * - ``<cuda/std/cstdint>``
     - Fixed-width integer types
     - libhipcxx 2.7
     - `\<cstdint\> <https://en.cppreference.com/w/cpp/header/cstdint>`_

   * - ``<cuda/std/cstdlib>``
     - Common utilities
     - libhipcxx 2.7
     - `\<cstdlib\> <https://en.cppreference.com/w/cpp/header/cstdlib>`_

   * - ``<cuda/std/cstring>``
     - Provides array manipulation functions such as ``memcpy``, ``memset`` and ``memcmp``
     - libhipcxx 3.0
     - `\<cstring\> <https://en.cppreference.com/w/cpp/header/cstring>`_
