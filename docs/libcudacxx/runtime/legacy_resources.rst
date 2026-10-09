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
  :description: API reference for the legacy pinned and managed memory resources, synchronous allocation interfaces in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, memory resource, pinned memory, managed memory, legacy_pinned_memory_resource

.. _cccl-runtime-legacy-resources:
.. _libcudacxx-extended-api-memory-resources-legacy-resources:

Legacy resources
================

Legacy memory resources provide synchronous allocation interfaces backed by the HIP runtime's legacy allocation APIs.
They are primarily intended for compatibility with platforms that do not support the newer memory
pool-based resources. Prefer the modern memory resources where available.

In libhipcxx, the pinned and managed memory pools are not available, so these resources are the way to allocate
pinned (page-locked) host memory and managed (unified) memory through the memory resource interface.

For the full memory resource model and property system, see
:ref:`Memory Resources (Extended API) <libcudacxx-extended-api-memory-resources>`.

:cpp:class:`cuda::mr::legacy_pinned_memory_resource`
------------------------------------------------------
.. _libcudacxx-memory-resource-legacy-pinned-memory-resource:

Provides pinned (page-locked) host allocations using ``hipHostMalloc`` and ``hipHostFree``. This resource is
*synchronous-only* and is intended as a compatibility fallback.
See `Pinned host memory <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/host_memory.html#pinned-host-memory>`__.

..
   Not supported in libhipcxx: cuda::pinned_memory_pool is not available.

   For CUDA 12.9 and later, prefer :cpp:any:`cuda::pinned_memory_resource`.

.. code:: cpp

   #include <cuda/memory_resource>

   void use_legacy_pinned() {
     cuda::mr::legacy_pinned_memory_resource resource{};
     void* ptr = resource.allocate_sync(1024, 64);
     // Use memory...
     resource.deallocate_sync(ptr, 1024, 64);
   }

:cpp:class:`cuda::mr::legacy_managed_memory_resource`
-------------------------------------------------------
.. _libcudacxx-memory-resource-legacy-managed-memory-resource:

Provides managed (unified) allocations using ``hipMallocManaged`` and ``hipFree``. This resource is
*synchronous-only* and accepts the HIP attachment flags (``hipMemAttachGlobal`` / ``hipMemAttachHost``).
See `Managed memory <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html#managed-memory>`__.

..
   Not supported in libhipcxx: cuda::managed_memory_pool is not available.

   Prefer :cpp:any:`cuda::managed_memory_resource` when available.

.. code:: cpp

   #include <cuda/memory_resource>

   void use_legacy_managed() {
     cuda::mr::legacy_managed_memory_resource resource{hipMemAttachGlobal};
     void* ptr = resource.allocate_sync(1024, 64);
     // Use memory...
     resource.deallocate_sync(ptr, 1024, 64);
   }
