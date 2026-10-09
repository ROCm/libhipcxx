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
  :description: API reference for cuda::is_host_accessible, cuda::is_device_accessible, and cuda::is_managed, which query whether memory is accessible from the host or a device, or is managed memory, in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, is_host_accessible, is_device_accessible, is_managed, pointer attributes, managed memory

.. _libcudacxx-extended-api-memory-is_pointer_accessible:

``cuda::is_host_accessible``, ``cuda::is_device_accessible``, ``cuda::is_managed``
==================================================================================

This page documents ``cuda::is_host_accessible``, ``cuda::is_device_accessible``, and ``cuda::is_managed``, which query the accessibility of the memory referenced by a pointer.

Defined in the ``<cuda/memory>`` header.

.. code:: cpp

   namespace cuda {

   [[nodiscard]] inline
   bool is_host_accessible(const void* ptr); // (1)

   [[nodiscard]] inline
   bool is_device_accessible(const void* ptr, device_ref device); // (2)

   [[nodiscard]] inline
   bool is_managed(const void* ptr); // (3)

   } // namespace cuda

Determines whether the memory referenced by ``ptr`` is accessible from the host (1), from the specified ``device`` (2), or is backed by Unified Memory (managed memory) (3).

- ``is_device_accessible()`` also checks whether the memory is peer-accessible from the specified ``device``.
- Pinned host memory is reported as host-accessible but not device-accessible.

..
   - ``is_device_accessible()`` also checks whether the memory is peer-accessible or allocated from a memory pool accessible to the specified ``device``.
   - ``is_host_accessible()`` also checks whether the memory is allocated from a memory pool accessible to the host.

----

**Parameters**

- ``ptr``: A pointer to the memory location to query.
- ``device``: A ``device_ref`` that denotes the device to query. (2)

**Return value**

- ``true`` if the queried property (host access, device access, or managed allocation) holds; otherwise, ``false``.

.. note::

  A ``__device__`` global array or variable cannot be used directly from host code without first retrieving its address with ``hipGetSymbolAddress()``.

**Prerequisites**

- The functions are available only in host code when the HIP runtime is available.

**Exceptions**

- The functions throw a ``cuda::cuda_error`` if the underlying HIP runtime API calls fail. Note that these functions may also fail with error codes from previously launched asynchronous operations.

**Undefined Behavior**

- The functions have undefined behavior if the pointer is not valid, for example, an already freed pointer.

Example
-------

.. code:: cpp

    #include <cassert>
    #include <cuda/memory>
    #include <hip/hip_runtime_api.h>

    int main() {
        cuda::device_ref dev{0};
        void* host_ptr    = nullptr;
        void* device_ptr  = nullptr;
        void* managed_ptr = nullptr;

        hipHostMalloc(&host_ptr, 1024);
        hipMalloc(&device_ptr, 1024);
        hipMallocManaged(&managed_ptr, 1024);

        assert(cuda::is_host_accessible(host_ptr));
        assert(!cuda::is_device_accessible(host_ptr, dev));

        assert(cuda::is_device_accessible(device_ptr, dev));
        assert(!cuda::is_host_accessible(device_ptr));

        assert(cuda::is_host_accessible(managed_ptr));
        assert(cuda::is_device_accessible(managed_ptr, dev));
        assert(cuda::is_managed(managed_ptr));

        hipHostFree(host_ptr);
        hipFree(device_ptr);
        hipFree(managed_ptr);
    }
