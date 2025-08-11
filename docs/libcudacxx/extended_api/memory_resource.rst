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
  :description: Overview of the libhipcxx memory resource extended API, providing a standard C++ interface for heterogeneous, stream-ordered memory allocation in HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, memory resource, cuda::mr, async_resource, resource_ref, stream-ordered allocation


.. _libcudacxx-extended-api-memory-resources:

Memory Resources
================

This page covers the memory resource extended API, providing a standard C++ interface for heterogeneous, stream-ordered memory allocation in HIP.

.. toctree::
   :hidden:
   :maxdepth: 1

   memory_resource/properties
   Resources <memory_resource/resource>
   Resource wrapper <memory_resource/resource_ref>

..
   The ``<cuda/memory_resource>`` header provides a standard C++ interface for *heterogeneous*, *stream-ordered* memory
   allocation tailored to the needs of CUDA C++ developers. This design builds off of the success of the `RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`__
   project and evolves the design based on lessons learned.
The ``<cuda/memory_resource>`` header provides a standard C++ interface for *heterogeneous*, *stream-ordered* memory
allocation tailored to the needs of HIP C++ developers.

``<cuda/memory_resource>`` only defines the memory allocation interface; it does not provide allocator
implementations. For HIP developers, `hipMM <https://github.com/AMD-Ecosystem/hipMM>`__ provides implementations of
the ``cuda::mr`` interfaces optimized for AMD GPUs.

At a high level, the header provides:

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - API
     - Description
     - Since

   * - :ref:`cuda::get_property <libcudacxx-extended-api-memory-resources-properties>`
     - Infrastructure to tag a user defined type with a given property
     - stable CCCL 3.1.0 / CUDA 13.1, experimental CCCL 2.2.0 / CUDA 12.3
   * - :ref:`cuda::mr::{synchronous_}resource <libcudacxx-extended-api-memory-resources-resource>` and
       :ref:`cuda::mr::{synchronous_}resource_with <libcudacxx-extended-api-memory-resources-resource>`
     - Concepts that provide proper constraints for arbitrary memory resources.
     - stable CCCL 3.1.0 / CUDA 13.1, experimental CCCL 2.2.0 / CUDA 12.3
   * - :ref:`cuda::mr::{async_}resource_ref <libcudacxx-extended-api-memory-resources-resource-ref>`
     - A non-owning type-erased memory resource wrapper that enables consumers to specify properties of resources that they expect.
       ``resource_ref`` is still an experimental design, only available if ``LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE`` is defined
     - experimental CCCL 2.2.0 / CUDA 12.3

These features are an evolution of `std::pmr::memory_resource <https://en.cppreference.com/w/cpp/header/memory_resource>`__
that was introduced in C++17. While ``std::pmr::memory_resource`` provides a polymorphic memory resource that can be
adopted through inheritance, it is not properly suited for heterogeneous systems.

With the current design it ranges from cumbersome to impossible to verify whether a memory resource provides allocations
that are e.g. accessible on device, or whether it can utilize other allocation mechanisms.

To better support asynchronous HIP `stream-ordered allocations <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html>`__
libhipcxx provides :ref:`cuda::stream_ref <libcudacxx-extended-api-streams-stream-ref>` as a wrapper around
``hipStream_t``. The definition of ``cuda::stream_ref`` can be found in the ``<cuda/stream_ref>`` header.
