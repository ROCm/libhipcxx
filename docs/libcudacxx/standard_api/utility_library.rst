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
  :description: Documents the utility library headers in libhipcxx, including bitset, expected, functional, memory, optional, tuple, type_traits, utility, variant, and version.
  :keywords: libhipcxx, ROCm, HIP, C++, utility, optional, tuple, variant, type_traits

.. _libcudacxx-standard-api-utility:

Utility Library
===============

This page covers the utility library headers available in libhipcxx, including bitset, expected, functional, memory, optional, tuple, type_traits, utility, variant, and version.

.. toctree::
   :hidden:
   :maxdepth: 1

   utility_library/bitset
   utility_library/expected
   utility_library/functional
   utility_library/memory
   utility_library/optional
   utility_library/tuple
   utility_library/type_traits
   utility_library/utility
   utility_library/variant
   utility_library/version

Any Standard C++ header not listed below is omitted. Some of the Standard C++ facilities in this header are omitted, see
the information about the individual features for details.

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - :ref:`libcudacxx-standard-api-utility-bitset`
     - Fixed-size sequence of bits
     - libhipcxx 2.7
   * - :ref:`libcudacxx-standard-api-utility-expected`
     - Optional value with error channel
     - libhipcxx 2.7
   * - :ref:`libcudacxx-standard-api-utility-functional`
     - Function objects and function wrappers
     - libhipcxx 2.7
   * - :ref:`libcudacxx-standard-api-utility-memory`
     - Low-level memory management utilities
     - libhipcxx 2.7
   * - :ref:`libcudacxx-standard-api-utility-optional`
     - Optional value
     - libhipcxx 2.7
   * - :ref:`libcudacxx-standard-api-utility-tuple`
     - Fixed-sized heterogeneous container
     - libhipcxx 2.7
   * - :ref:`libcudacxx-standard-api-utility-type-traits`
     - Compile-time type introspection
     - libhipcxx 2.7
   * - :ref:`libcudacxx-standard-api-utility-utility`
     - Various utility components
     - libhipcxx 2.7
   * - :ref:`libcudacxx-standard-api-utility-variant`
     - Type safe union type
     - libhipcxx 2.7
   * - :ref:`libcudacxx-standard-api-utility-version`
     - Compile-time version information and feature test macros
     - libhipcxx 2.7
