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
  :description: Learn how libhipcxx provides C++ Standard Library features for HIP device code, enabling opt-in, incremental heterogeneous programming on AMD GPUs.
  :keywords: libhipcxx, ROCm, HIP, C++, standard library, host, device, heterogeneous, AMD GPU, CUDA alternative

.. _libhipcxx-standard-library-features:

******************************************
C++ Standard Library features in libhipcxx
******************************************

If you are a C++ developer, then you know the C++ Standard Library (`sometimes referred to as "The
STL" <https://cppreference.com/cpp/standard_library>`_)
as what comes along with your compiler and provides things like ``std::string``, ``std::vector``, or
``std::atomic``. It provides the fundamental abstractions that C++ developers need to build high
quality applications and libraries.

By default, these abstractions aren't available when writing HIP C++ device code because they don't
have the necessary ``__host__`` / ``__device__`` decorators, and their implementation may not be suitable
for use in and across host and device code.

libhipcxx solves this problem by providing an opt-in, incremental, heterogeneous implementation of
C++ Standard Library features:

- **Opt-in**: It does not replace the Standard Library provided by your host compiler, meaning anything in ``std::``.
- **Incremental**: It does not provide a complete C++ Standard Library implementation.
- **Heterogeneous**: It works in both host and device code, as well as passing between host and device code.

If you know how to use headers such as ``<atomic>`` or ``<type_traits>`` from the C++ Standard
Library, then you know how to use libhipcxx. For code that must run in both host and device
code, add ``cuda/std/`` to the start of the include and ``cuda::`` before the use of ``std::``.
Host-only code continues to use the plain ``<atomic>`` and ``std::atomic`` from your host
compiler. The ``hip/std/`` and ``hip::std::`` spellings are equivalent and name the same types.

.. code-block:: cpp

    #include <cuda/std/atomic>

    cuda::std::atomic<int> x;

.. note::

   libhipcxx does not provide its own documentation for Standard Library features. Instead,
   libhipcxx :ref:`documents which Standard Library headers <libcudacxx-standard-api>` are made
   available, and defers documentation of individual features within those headers to other sources,
   such as `cppreference <https://en.cppreference.com/w/>`_.

C++ standard version support
=============================

libhipcxx is tested with C++17. C++20 is supported but not yet tested. Many features from newer
standards are backported so they are available in earlier dialect modes. For example, several C++20
chrono calendar types are available in C++14 mode, and many concepts are usable as type traits from
C++17.

Some headers require a minimum dialect:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Minimum dialect
     - Headers
   * - C++20
     - ``<bit>``, ``<concepts>``, ``<span>``, ``<source_location>``
   * - C++23
     - ``<expected>``, ``<mdspan>``
   * - C++26
     - ``<inplace_vector>``, ``<linalg>``

For the complete per-header dialect requirements, see the
:ref:`Standard API feature table <libcudacxx-standard-api>`.

Available headers
=================

libhipcxx provides conforming, host-and-device implementations of a subset of the C++ Standard
Library. The :ref:`Standard API reference <libcudacxx-standard-api>` lists every supported header
together with its ``cuda/std/`` include path and minimum C++ dialect.

Headers span the following groups:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Group
     - Headers
     - Notes
   * - Synchronization
     - * ``<atomic>``
     -
   * - Containers
     - * ``<array>``
       * ``<span>``
       * ``<mdspan>``
       * ``<inplace_vector>``
     -
   * - Numerics
     - * ``<ratio>``
       * ``<bit>``
       * ``<complex>``
       * ``<linalg>``
       * ``<numbers>``
       * ``<numeric>``
     -
   * - Ranges
     - * ``<iterator>``
       * ``<ranges>``
     - Concepts and ``subrange`` only. Range algorithms and views are not implemented.
   * - Time
     - * ``<chrono>``
     - Timezone support and ``steady_clock`` are not available. The host and device clocks are not
       synchronized.
   * - Concepts
     - * ``<concepts>``
     -
   * - Utilities
     - * ``<type_traits>``
       * ``<utility>``
       * ``<tuple>``
       * ``<optional>``
       * ``<expected>``
       * ``<variant>``
       * ``<functional>``
       * ``<bitset>``
       * ``<memory>``
       * ``<initializer_list>``
       * ``<source_location>``
       * ``<version>``
     -
   * - Type support
     - * ``<limits>``
       * ``<climits>``
     -
   * - C library compatibility
     - * ``<cassert>``
       * ``<cfloat>``
       * ``<climits>``
       * ``<cmath>``
       * ``<cstddef>``
       * ``<cstdint>``
       * ``<cstdlib>``
       * ``<cstring>``
       * ``<ctime>``
     -

GPU-specific extensions
=======================

Beyond the conforming Standard Library subset, libhipcxx provides GPU-specific extensions under the
``cuda::`` (or ``hip::`` — both are equivalent) namespace. These cover areas such as thread-scope
atomics, warp intrinsics, async memory operations, and GPU math utilities that have no equivalent in
host-only C++.

See :ref:`C++ Standard Library extensions in libhipcxx <libhipcxx-extensions>` for the full list,
and :ref:`HIP-specific abstractions and namespaces <libhipcxx-hip-abstractions>` for an explanation
of the namespace hierarchy.

Unsupported features
====================

Several APIs from the upstream libcudacxx project are not supported in libhipcxx because they depend
on NVIDIA hardware or NVIDIA PTX instructions. These include the synchronization primitives
``<cuda/std/latch>``, ``<cuda/std/barrier>``, ``<cuda/std/semaphore>``, their scoped
``cuda::`` equivalents, ``<cuda/pipeline>``, ``<cuda/annotated_ptr>``, and the PTX instruction
wrappers in ``<cuda/ptx>``.

For the complete list, see :ref:`Limitations and unsupported APIs <libhipcxx-limitations>`.
