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
  :description: Overview of the libhipcxx extended API, covering bit manipulation, execution model, synchronization, memory, streams, and math utilities for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, extended API, synchronization, memory

.. _libcudacxx-extended-api:

Extended API
============

This section documents the extended API provided by libhipcxx, covering bit manipulation, the memory model, synchronization primitives, memory management, streams, math utilities, and more.

.. toctree::
   :maxdepth: 2

   extended_api/bit
   extended_api/memory_model
   extended_api/thread_groups
   extended_api/synchronization_primitives
   extended_api/functional
   extended_api/streams
   extended_api/memory_resource
   extended_api/math
   extended_api/mdspan

..
   Not supported in libhipcxx, see reference/libhipcxx-limitations.rst. The pages are also
   listed in exclude_patterns in docs/conf.py.

   - cuda::aligned_size_t and cuda::memcpy_async depend on <cuda/barrier> and <cuda/pipeline>.
   - <cuda/annotated_ptr> and cuda::access_property are not provided, and cuda::discard_memory
     is a no-op on AMD GPUs.
   - The warp shuffle functions in <cuda/warp> are only compiled for NVIDIA PTX targets.
   - cuda::for_each_canceled_block does not cancel blocks on AMD GPUs.
   - extended_api/execution_model documents CUDA forward progress guarantees that may not hold
     on AMD GPUs.

   extended_api/shapes
   extended_api/asynchronous_operations
   extended_api/memory_access_properties
   extended_api/warp
   extended_api/work_stealing
   extended_api/execution_model
