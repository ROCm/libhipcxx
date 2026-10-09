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
  :description: Overview of the libhipcxx Runtime API, which provides RAII wrappers for streams, events, devices, memory pools and kernel launches in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, runtime API, streams, events, kernel launch, memory pools

.. _libcudacxx-runtime-api:

Runtime
========

The Runtime API provides higher-level building blocks for core HIP functionality. It takes the existing HIP runtime API
set and removes or replaces some problematic patterns, such as implicit state. It is designed to make common operations
like resource management, work submission, and memory allocation easier to express and safer to compose. These APIs lower
to the HIP runtime/driver API under the hood, but remain composable with the HIP runtime API by reusing runtime handle types
(such as ``hipStream_t``) in the interfaces. This results in an interface that applies RAII for lifetime management,
while remaining composable with existing HIP C++ code that manages resources explicitly.

At a glance, the runtime layer includes:

- Streams and events work submission and synchronization.
- Memory pools to allocate device memory, either synchronously or in stream order.
- Launch API to configure and launch kernels.
- Runtime algorithms like ``copy_bytes`` and ``fill_bytes`` for basic data movement.
- Legacy memory resources as synchronous allocation interfaces for pinned and managed memory.

..
   Not supported in libhipcxx, see reference/libhipcxx-limitations.rst.
   cuda::buffer requires a hipCUB port and is not available yet. Managed and pinned memory pools
   are not available in libhipcxx.

   - Buffers as a typed, stream-ordered storage with property-checked memory container.
   - Memory pools to allocate device, managed, and pinned memory, either directly or through buffers.

See :ref:`HIP runtime interactions <cccl-runtime-cudart-interactions>` if you are interested in HIP runtime interop.

..
   Not supported in libhipcxx, see reference/libhipcxx-limitations.rst.
   The example uses cuda::buffer, which requires a hipCUB port and is not available yet.

   Example: vector add with buffers, pools, and launch
   ---------------------------------------------------

   .. code:: cpp

      #include <cuda/devices>
      #include <cuda/stream>
      #include <cuda/std/span>
      #include <cuda/buffer>
      #include <cuda/memory_pool>
      #include <cuda/launch>

      struct kernel {
        template <typename Config>
        __device__ void operator()(Config config,
                                   cuda::std::span<const float> A,
                                   cuda::std::span<const float> B,
                                   cuda::std::span<float> C) {
          auto tid = cuda::gpu_thread.rank(cuda::grid, config);
          if (tid < A.size())
            C[tid] = A[tid] + B[tid];
        }
      };

      int main() {
        cuda::device_ref device = cuda::devices[0];
        cuda::stream stream{device};
        auto pool = cuda::device_default_memory_pool(device);

        int num_elements = 1000;
        auto A = cuda::make_buffer<float>(stream, pool, num_elements, 1.0);
        auto B = cuda::make_buffer<float>(stream, pool, num_elements, 2.0);
        auto C = cuda::make_buffer<float>(stream, pool, num_elements, cuda::no_init);

        constexpr int threads_per_block = 256;
        auto config = cuda::distribute<threads_per_block>(num_elements);

        cuda::launch(stream, config, kernel{}, A, B, C);
      }

.. toctree::
   :hidden:
   :maxdepth: 1

   runtime/cudart_interactions
   runtime/stream
   runtime/event
   runtime/algorithm
   runtime/device
   runtime/hierarchy
   runtime/launch
   runtime/memory_pools
   runtime/legacy_resources

..
   Not supported in libhipcxx, see reference/libhipcxx-limitations.rst.
   cuda::buffer requires a hipCUB port and is not available yet.

   runtime/buffer

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - **API**
     - **Content**
     - **Since**

   * - :ref:`devices <cccl-runtime-device-devices>`
     - A range of all available HIP devices
     - libhipcxx 3.4

   * - :ref:`device_ref <cccl-runtime-device-device-ref>`
     - A non-owning representation of a HIP device
     - libhipcxx 3.4

   * - :ref:`stream_ref <cccl-runtime-stream-stream-ref>`
     - A non-owning wrapper around a ``hipStream_t``
     - libhipcxx 2.7

   * - :ref:`stream <cccl-runtime-stream-stream>`
     - An owning wrapper around a ``hipStream_t``
     - libhipcxx 3.4

   * - :ref:`event_ref <cccl-runtime-event-event-ref>`
     - A non-owning wrapper around a ``hipEvent_t``
     - libhipcxx 3.4

   * - :ref:`event <cccl-runtime-event-event>`
     - An owning wrapper around a ``hipEvent_t`` (timing disabled)
     - libhipcxx 3.4

   * - :ref:`timed_event <cccl-runtime-event-timed-event>`
     - An owning wrapper around a ``hipEvent_t`` with timing enabled and elapsed-time queries
     - libhipcxx 3.4

   * - :ref:`copy_bytes <cccl-runtime-algorithm-copy_bytes>`
     - Byte-wise copy into a ``cuda::stream_ref`` for ``cuda::std::span``/``cuda::std::mdspan`` sources and destinations
     - libhipcxx 3.4

   * - :ref:`fill_bytes <cccl-runtime-algorithm-fill_bytes>`
     - Byte-wise fill into a ``cuda::stream_ref`` for ``cuda::std::span``/``cuda::std::mdspan`` destinations
     - libhipcxx 3.4

   * - :ref:`hierarchy <cccl-runtime-hierarchy-hierarchy>`
     - Representation of HIP thread hierarchies (grid, block, warp, thread)
     - libhipcxx 3.4

   * - :ref:`launch <cccl-runtime-launch-launch>`
     - Kernel launch with configuration and options
     - libhipcxx 3.4

   * - :ref:`kernel_config <cccl-runtime-launch-kernel-config>`
     - Kernel launch configuration combining hierarchy dimensions and launch options
     - libhipcxx 3.4

   * - :ref:`make_config <cccl-runtime-launch-make-config>`
     - Factory function to create kernel configurations from hierarchy dimensions and launch options
     - libhipcxx 3.4

   * - :ref:`device_memory_pool <cccl-runtime-memory-pools-device-memory-pool>`
     - Stream-ordered device memory pool using the HIP memory pool API
     - libhipcxx 3.4

   * - :ref:`device_default_memory_pool <cccl-runtime-memory-pools-device-default>`
     - Get the default device memory pool for a device
     - libhipcxx 3.4

   * - :ref:`legacy resources <cccl-runtime-legacy-resources>`
     - Synchronous compatibility resources backed by legacy HIP allocation APIs.
     - libhipcxx 3.4

..
   Not supported in libhipcxx, see reference/libhipcxx-limitations.rst.
   - The architecture traits describe NVIDIA GPU architectures (cuda::arch_id).
   - Managed and pinned memory pools are not available in libhipcxx; use the legacy resources instead.
   - cuda::buffer requires a hipCUB port and is not available yet.

   * - :ref:`arch_traits <cccl-runtime-device-arch-traits>`
     - Per-architecture trait accessors
     - libhipcxx 3.4

   * - :ref:`managed_memory_pool <cccl-runtime-memory-pools-managed-memory-pool>`
     - Stream-ordered managed (unified) memory pool
     - libhipcxx 3.4

   * - :ref:`pinned_memory_pool <cccl-runtime-memory-pools-pinned-memory-pool>`
     - Stream-ordered pinned (page-locked) host memory pool
     - libhipcxx 3.4

   * - :ref:`managed_default_memory_pool <cccl-runtime-memory-pools-managed-default>`
     - Get the default managed (unified) memory pool
     - libhipcxx 3.4

   * - :ref:`pinned_default_memory_pool <cccl-runtime-memory-pools-pinned-default>`
     - Get the default pinned (page-locked) host memory pool
     - libhipcxx 3.4

   * - :ref:`buffer <cccl-runtime-buffer-buffer>`
     - Typed data container allocated from memory resources. It handles stream-ordered allocation, initialization, and deallocation of memory.
     - libhipcxx 3.4
