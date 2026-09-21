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

.. _cccl-cpp-libraries:

CUDA C++ Core Libraries
=======================

.. toctree::
   :hidden:
   :maxdepth: 3

   libhipcxx <https://nvidia.github.io/cccl/libhipcxx/>
   CUB <https://nvidia.github.io/cccl/cub/>
   Thrust <https://nvidia.github.io/cccl/thrust/>
   Cuda Experimental <https://nvidia.github.io/cccl/cudax/>

Welcome to the CUDA Core Compute Libraries (CCCL) libraries for C++.

The concept for the  CCCL C++ librarires grew organically out of the Thrust,
CUB, and libhipcxx projects that were developed independently over the years
with a similar goal: to provide high-quality, high-performance, and
easy-to-use C++ abstractions for CUDA developers. Naturally, there was a lot
of overlap among the three projects, and it became clear the community would
be better served by unifying them into a single repository.

- `libhipcxx <https://nvidia.github.io/cccl/libhipcxx/>`__
  is the CUDA C++ Standard Library. It provides an implementation of the C++
  Standard Library that works in both host and device code. Additionally, it
  provides abstractions for CUDA-specific hardware features like
  synchronization primitives, cache control, atomics, and more.

- `CUB <https://nvidia.github.io/cccl/cub/>`__
  is a lower-level, CUDA-specific library designed for speed-of-light parallel
  algorithms across all GPU architectures. In addition to device-wide
  algorithms, it provides *cooperative algorithms* like block-wide reduction
  and warp-wide scan, providing CUDA kernel developers with building blocks to
  create speed-of-light, custom kernels.

- `Thrust <https://nvidia.github.io/cccl/thrust/>`__
  is the C++ parallel algorithms library which inspired the introduction of
  parallel algorithms to the C++ Standard Library. Thrust's high-level
  interface greatly enhances programmer productivity while enabling performance
  portability between GPUs and multicore CPUs via configurable backends that
  allow using multiple parallel programming frameworks (such as CUDA, TBB, and
  OpenMP).

- `Cuda Experimental <https://nvidia.github.io/cccl/cudax/>`__
  is a library of experimental features that are still in the design process.

The main goal of the CCCL C++ libraries is to fill a similar role that the
Standard C++ Library fills for Standard C++: provide general-purpose,
speed-of-light tools to CUDA C++ developers, allowing them to focus on
solving the problems that matter. Unifying these projects is the first step
towards realizing that goal.
