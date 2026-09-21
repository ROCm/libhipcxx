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
  :description: Using libhipcxx in a CMake project
  :keywords: libhipcxx, ROCm, cmake, find_package, target_link_libraries

.. _libhipcxx-use-in-a-project:

*******************************************
How to add libhipcxx to a CMake project
*******************************************

To use libhipcxx in your own project, add the following lines to your ``CMakeLists.txt`` file:

.. code-block:: cmake

    # "/opt/rocm" - default ROCm install prefix
    find_package(libhipcxx REQUIRED)

    # ...

    # include the libhipcxx headers
    target_link_libraries(your_target PRIVATE libhipcxx::libhipcxx)

``libhipcxx::libhipcxx`` is an interface target, so linking against it only adds the libhipcxx
include directories to your target.

If you installed libhipcxx into a non-default location, set ``CMAKE_PREFIX_PATH`` to that install
directory when you configure your project:

.. code-block:: shell

    cmake -DCMAKE_PREFIX_PATH=<path to libhipcxx install directory> ..

Including the headers
=====================

To use a Standard Library facility in host and device code, add ``cuda/std/`` to the start of the
include and ``cuda::`` before the use of ``std::``:

.. code-block:: cpp

    #include <cuda/std/atomic>

    cuda::std::atomic<int> x;

To use an extension to a Standard Library facility, drop the ``std``:

.. code-block:: cpp

    #include <cuda/atomic>

    cuda::atomic<int, cuda::thread_scope_device> x;

You can also write these as ``hip/std/`` and ``hip::std::``, or ``hip/`` and ``hip::``. Both
spellings resolve to the same headers and can be used interchangeably. For an explanation of when to
use each namespace, see
:doc:`HIP-specific abstractions and namespaces <../conceptual/libhipcxx-hip-abstractions>`.
