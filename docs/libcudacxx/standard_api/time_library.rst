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
  :description: Documents cuda::std::chrono in libhipcxx, covering the host and device implementation of system_clock, high_resolution_clock, and the omission of steady_clock.
  :keywords: libhipcxx, ROCm, HIP, C++, chrono, system_clock, high_resolution_clock, steady_clock, time

.. _libcudacxx-standard-api-time:

Time Library
============

This page documents the time library headers available in libhipcxx, covering chrono clocks, durations, and time points.

See the documentation of the standard header `\<chrono\> <https://en.cppreference.com/w/cpp/header/chrono>`_

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - ``<cuda/std/chrono>``
     - Times, dates, and clocks
     - libhipcxx 2.7

Implementation-defined behavior
-------------------------------

``system_clock``
~~~~~~~~~~~~~~~~

`std::chrono::system_clock <https://en.cppreference.com/w/cpp/chrono/system_clock>`_ tracks real-world time. The C++
Standard leaves it unspecified whether the clock increases monotonically. In libhipcxx it does not, so ``is_steady``
is ``false``.

- In host code, ``system_clock::now()`` returns the time point of the host standard library's
  ``std::chrono::system_clock``.
- In device code, it reads HIP's
  `wall_clock64() <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#timer-functions>`_,
  which ticks at a constant rate and increases monotonically.

The device clock does not count from the UNIX epoch and is not synchronized with the host clock, so time points
obtained in device code cannot be compared with time points obtained in host code. The two clocks may also tick at
different rates. See :ref:`Limitations and unsupported APIs <libhipcxx-limitations>`.

``high_resolution_clock``
~~~~~~~~~~~~~~~~~~~~~~~~~

`std::chrono::high_resolution_clock <https://en.cppreference.com/w/cpp/chrono/high_resolution_clock>`_ is an alias for
``system_clock``, so it counts real-world time and ``is_steady`` is ``false``. It is steady within device code, which
makes it suitable for measuring elapsed time inside a kernel.

Omissions
---------

The following facilities from section `time.syn <https://eel.is/c++draft/time.syn>`_ of the C++ Standard are not
available in libhipcxx:

- `std::chrono::steady_clock <https://en.cppreference.com/w/cpp/chrono/steady_clock>`_, a monotonically increasing
  clock. libhipcxx provides no clock that is steady across host and device, because the device clock is neither
  synchronized with the host clock nor guaranteed to tick at the same rate.
- `std::chrono::duration I/O operators <https://eel.is/c++draft/time.duration.io>`_, which would require a
  heterogeneous C++ I/O streams implementation.
