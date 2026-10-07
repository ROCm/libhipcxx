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
  :description: API reference for cuda::get_device_address, which returns a valid device pointer to a device object, replacing uses of hipGetSymbolAddress in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, get_device_address, device pointer, hipGetSymbolAddress, functional

.. _libcudacxx-extended-api-functional-get-device-address:

``cuda::get_device_address``
==================================

This page documents ``cuda::get_device_address``, which returns a valid device pointer to a device object.

Defined in the header ``<cuda/functional>``:

``cuda::get_device_address`` returns a valid pointer to a device object.
It replaces uses of ``hipGetSymbolAddress``, which requires an inout parameter.

Example
-------

.. code:: cpp

  #include <cuda/functional>

  __device__ int device_object[] = {42, 1337, -1, 0};

  __global__ void example_kernel(int *data) { ... }

  void example()
  {
    {
      T* host_address = cuda::std::addressof(device_object);

      hipPointerAttribute_t attributes;
      hipError_t status = hipPointerGetAttributes(&attributes, host_address);
      assert(status == hipSuccess);
      assert(attributes.devicePointer == nullptr);

      // Calling a kernel with host_address would segfault
      // example_kernel<<<1, 1>>>(host_address);
    }

    {
      T* device_address = cuda::get_device_address(device_object);

      hipPointerAttribute_t attributes;
      hipError_t status = hipPointerGetAttributes(&attributes, device_address);
      assert(status == hipSuccess);
      assert(attributes.devicePointer == device_address);

      // Safe to call a kernel
      example_kernel<<<1, 1>>>(device_address);
    }
  }
