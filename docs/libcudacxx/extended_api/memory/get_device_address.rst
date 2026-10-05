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
  :description: API reference for cuda::get_device_address, which returns a valid device pointer to a device object, replacing uses of cudaGetSymbolAddress in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, get_device_address, device pointer, cudaGetSymbolAddress, functional

.. _libcudacxx-extended-api-memory-get-device-address:

``cuda::get_device_address``
============================

This page documents ``cuda::get_device_address``, which returns a valid device pointer to a device object.


Defined in the headers ``<cuda/memory>`` and ``<cuda/functional>``.

.. code:: cuda

  namespace cuda {

  template <typename T>
  [[nodiscard]] __host__ __device__ inline
  T* get_device_address(T& device_object);                    // (1)

  template <typename T>
  [[nodiscard]] __host__ inline
  T* get_device_address(T& device_object, device_ref device); // (2)

  } // namespace cuda

``cuda::get_device_address`` returns a valid pointer to a device object for the current (1) or ``device`` (2) device. It replaces uses of ``cudaGetSymbolAddress``, which requires an inout parameter.

**Parameters**

- ``device_object``: Reference to a device object. (1, 2)
- ``device``: Device for which the object's address shall be retrieved. (2)

**Constraints**

- ``device_object`` must be a ``__device__`` or ``__constant__`` decorated variable.

Example
-------

.. code:: cuda

  #include <cuda/devices>
  #include <cuda/memory>

  __device__ int device_object[] = {42, 1337, -1, 0};

  __global__ void example_kernel(int *data) { ... }

  void example()
  {
    cuda::device_ref device{0};

    {
      T* host_address = cuda::std::addressof(device_object);

      cudaPointerAttributes attributes;
      cudaError_t status = cudaPointerGetAttributes(&attributes, host_address);
      assert(status == cudaSuccess);
      assert(attributes.devicePointer == nullptr);

      // Calling a kernel with host_address would segfault
      // example_kernel<<<1, 1>>>(host_address);
    }

    {
      T* device_address = cuda::get_device_address(device_object, device);

      cudaPointerAttributes attributes;
      cudaError_t status = cudaPointerGetAttributes(&attributes, device_address);
      assert(status == cudaSuccess);
      assert(attributes.devicePointer == device_address);

      // Safe to call a kernel
      example_kernel<<<1, 1>>>(device_address);
    }
  }
