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
  :description: API reference for cuda::shared_memory_accessor and cuda::shared_memory_mdspan, which provide mdspan views of the shared memory space with additional safety checks in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, mdspan, shared_memory_accessor, shared_memory_mdspan, shared memory, accessor

.. _libcudacxx-extended-api-mdspan-shared-memory-accessor:

``shared_memory`` ``mdspan`` and ``accessor``
=============================================

``shared_memory`` ``mdspan`` and ``accessor`` allow to express multi-dimensional views of the shared memory space and provide additional safety checks and performance optimizations.

Types and Traits
----------------

.. code:: cpp

    namespace cuda {

    template <typename AccessorPolicy>
    using shared_memory_accessor;

    template <typename ElementType,
              typename Extents,
              typename LayoutPolicy   = cuda::std::layout_right,
              typename AccessorPolicy = cuda::shared_memory_accessor<ElementType>>
    class shared_memory_mdspan;

    } // namespace cuda

``mdspan`` type and accessor tailored for the *shared* memory space.

----

.. code:: cpp

    namespace cuda {

    template <typename T>
    inline constexpr bool is_shared_memory_accessor_v = /* true if T is a shared_memory_accessor, false otherwise */;

    template <typename T>
    inline constexpr bool is_shared_memory_mdspan_v = /* true if T is a shared_memory_mdspan, false otherwise */;

    } // namespace cuda

Features
--------

**Constraints**

- Accessor ``data_handle_type`` must be a pointer type.

**Preconditions**

- Accessing elements through a ``shared_memory_accessor`` is only allowed in device code.
- The underlying pointer must be in the *shared* memory space.
- Access offset must be within the maximum possible shared memory allocation size.

..
   **Performance considerations**

   - The functionality guarantees that the accesses use shared memory instructions (``STS/LDS``) rather than generic memory instructions.

Example
-------

.. code:: cpp

    #include <cuda/mdspan>
    #include <cstdio>

    __global__ void kernel() {
        extern __shared__ int shmem[];

        // Create a shared_memory_mdspan over the dynamic shared memory
        cuda::shared_memory_mdspan md(shmem, cuda:std::dims<2>{32, 32});

        if (threadIdx.x < 32) {
             md[threadIdx.x][threadIdx.x] = threadIdx.x; // write on the diagonal
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            printf("md[5][5] = %d\n", md[5][5]); // read from the diagonal
        }
    }

    int main() {
        kernel<<<1, 32, 32 * 32 * sizeof(int)>>>();
        hipDeviceSynchronize();
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/sojGnKoY9>`_
