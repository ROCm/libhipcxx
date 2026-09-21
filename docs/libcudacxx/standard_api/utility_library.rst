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

.. _libcudacxx-standard-api-utility:

Utility Library
=======================

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
     - CCCL 2.8.0
   * - :ref:`libcudacxx-standard-api-utility-expected`
     - Optional value with error channel
     - CCCL 2.3.0 / CUDA 12.4
   * - :ref:`libcudacxx-standard-api-utility-functional`
     - General-purpose polymorphic function wrapper
     - CCCL 2.9.0 / CUDA 12.9
   * - :ref:`libcudacxx-standard-api-utility-memory`
     - Function objects and function wrappers
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.2
   * - :ref:`libcudacxx-standard-api-utility-optional`
     - Optional value
     - CCCL 2.3.0 / CUDA 12.4
   * - :ref:`libcudacxx-standard-api-utility-tuple`
     - Fixed-sized heterogeneous container
     - libhipcxx 1.3.0 / CCCL 2.0.0 / CUDA 11.2
   * - :ref:`libcudacxx-standard-api-utility-type-traits`
     - Compile-time type introspection
     - libhipcxx 1.0.0 / CCCL 2.0.0 / CUDA 10.2
   * - :ref:`libcudacxx-standard-api-utility-utility`
     - Various utility components
     - libhipcxx 1.3.0 / CCCL 2.0.0 / CUDA 11.2
   * - :ref:`libcudacxx-standard-api-utility-variant`
     - Type safe union type
     - CCCL 2.4.0 / CUDA 12.5
   * - :ref:`libcudacxx-standard-api-utility-version`
     - Compile-time version information and feature test macros
     - libhipcxx 1.2.0 / CCCL 2.0.0 / CUDA 11.1
