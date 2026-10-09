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
  :description: Documents the numerics library headers available in libhipcxx, including bit manipulation, complex numbers, linear algebra, numeric constants, numeric algorithms, and random number generation.
  :keywords: libhipcxx, ROCm, HIP, C++, numerics, bit, complex, linalg, numbers, numeric, random

.. _libcudacxx-standard-api-numerics:

Numerics Library
================

This page covers the numerics library headers available in libhipcxx, including bit manipulation, complex numbers, linear algebra, numeric constants, numeric algorithms, and random number generation.

.. toctree::
   :hidden:
   :maxdepth: 1

   numerics_library/bit
   numerics_library/complex
   numerics_library/linalg
   numerics_library/numbers
   numerics_library/numeric
   numerics_library/random

Any Standard C++ header not listed below is omitted.

.. list-table::
   :widths: 25 45 30 20
   :header-rows: 1

   * - **Header**
     - **Content**
     - **libhipcxx Availability**
     - **C++ Reference**

   * - ``<cuda/std/ratio>``
     - Compile-time rational arithmetic
     - libhipcxx 2.7
     - `\<ratio\> <https://en.cppreference.com/w/cpp/header/ratio>`_

   * - :ref:`\<cuda/std/bit\> <libcudacxx-standard-api-numerics-bit>`
     - Access, manipulate, and process individual bits and bit sequences.
     - libhipcxx 2.7
     - `\<bit\> <https://en.cppreference.com/w/cpp/header/bit>`_

   * - :ref:`\<cuda/std/complex\> <libcudacxx-standard-api-numerics-complex>`
     - Complex number type
     - libhipcxx 2.7
     - `\<complex\> <https://en.cppreference.com/w/cpp/header/complex>`_

   * - :ref:`\<cuda/std/linalg\> <libcudacxx-standard-api-numerics-linalg>`
     - Linear algebra layouts and accessors
     - libhipcxx 3.0
     - `\<linalg\> <https://en.cppreference.com/w/cpp/header/linalg>`_

   * - :ref:`\<cuda/std/numbers\> <libcudacxx-standard-api-numerics-numbers>`
     - Numeric constants
     - libhipcxx 3.0
     - `\<numbers\> <https://en.cppreference.com/w/cpp/header/numbers>`_

   * - :ref:`\<cuda/std/numeric\> <libcudacxx-standard-api-numerics-numeric>`
     - Numeric algorithms
     - libhipcxx 2.7
     - `\<numeric\> <https://en.cppreference.com/w/cpp/header/numeric>`_

   * - :ref:`\<cuda/std/random\> <libcudacxx-standard-api-numerics-random>`
     - Random number generation
     - libhipcxx 3.4
     - `\<random\> <https://en.cppreference.com/w/cpp/header/random>`_
