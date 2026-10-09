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
  :description: API reference for cuda::ptr_in_range, which checks whether a pointer lies inside a half-open range of pointers in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, ptr_in_range, pointer range, bounds check, memory

.. _libcudacxx-extended-api-memory-ptr_in_range:

``cuda::ptr_in_range``
======================

This page documents ``cuda::ptr_in_range``, which checks whether a pointer lies inside a range.

Defined in the header ``<cuda/memory>``.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   bool ptr_in_range(T* ptr, T* start, T* end) noexcept;

   } // namespace cuda

Checks whether ``ptr`` lies inside the half-open interval ``[start, end)``.

**Template parameters**

- ``T``: The type of the pointer.

**Parameters**

- ``ptr``: The pointer being tested.
- ``start``: Pointer to the first element in the range.
- ``end``: Pointer to one past the last element in the range.

**Return value**

- ``true`` when the pointer lies in ``[start, end)``, ``false`` otherwise.

**Preconditions**

- ``end`` must be greater than or equal to ``start``.

Example
-------

.. code:: cpp

    #include <cuda/memory>

    __global__ void kernel(float* data, size_t count) {
        float* first = data;
        float* last  = data + count;

        float* elem_ptr = data + threadIdx.x;
        if (cuda::ptr_in_range(elem_ptr, first, last)) {
            *elem_ptr = static_cast<float>(threadIdx.x);
        }
    }

    int main() {
        size_t N          = 32;
        float* device_ptr = nullptr;
        hipMalloc(&device_ptr, N * sizeof(float));

        kernel<<<1, N>>>(device_ptr, N);
        hipDeviceSynchronize();

        hipFree(device_ptr);
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/sMz76hGEc>`__
