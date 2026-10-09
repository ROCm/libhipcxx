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
  :description: API reference for cuda::launch, cuda::make_config, the kernel launch options and cuda::host_launch in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, kernel launch, launch configuration, cooperative launch, dynamic shared memory, host_launch

.. _cccl-runtime-launch:

.. |cuda_launch| replace:: ``cuda::launch``
.. |cuda_kernel_config| replace:: ``cuda::kernel_config``
.. |cuda_make_config| replace:: ``cuda::make_config``
.. |cuda_cooperative_launch| replace:: ``cuda::cooperative_launch``
.. |cuda_dynamic_shared_memory| replace:: ``cuda::dynamic_shared_memory``
.. |cuda_launch_priority| replace:: ``cuda::launch_priority``
.. |cuda_host_launch| replace:: ``cuda::host_launch``

Launch
======

The launch API provides abstractions for launching HIP kernels with a given configuration. It supports kernel functions and device callable objects, cooperative launches, dynamic shared memory, and other launch options.
In libhipcxx, kernels are launched with ``hipModuleLaunchKernel``, or ``hipModuleLaunchCooperativeKernel`` for cooperative launches.

|cuda_launch|
--------------------------------------------------------------------------------------------
.. _cccl-runtime-launch-launch:

|cuda_launch| launches a kernel function or a device callable object on the specified stream with a given
configuration. The kernel can accept the configuration as its first argument to enable some device-side functionality,
but it is not required. If the kernel does accept the configuration as its first argument, |cuda_launch|
will automatically pass it into the kernel without the need to pass the configuration as an argument twice.

*Note:* Configuration won't be passed automatically into the kernel if it is an extended device lambda, it needs to be passed as the second launch function argument and as the first kernel argument.

The benefit of using a callable object with a device call operator (later called a kernel functor) is that it can have its
template arguments deduced from the arguments, while a kernel function needs to be explicitly instantiated. It also
allows attaching a default configuration that is later combined with the configuration passed to the launch.

Availability: libhipcxx 3.4

Example with kernel function:

.. code:: cpp

   #include <cuda/launch>
   #include <cstdio>

   template <typename Configuration>
   __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
     if (cuda::gpu_thread.rank(cuda::grid, conf) == thread_to_print) {
       printf("Hello from the GPU\n");
     }
   }

   void launch_kernel(cuda::stream_ref stream) {
     auto config = cuda::make_config(cuda::block_dims<128>(), cuda::grid_dims(4), cuda::cooperative_launch{});
     // Here the template needs to be explicitly instantiated, unlike in the kernel functor example where it can be deduced
     cuda::launch(stream, config, kernel<decltype(config)>, 42);
   }

Example with kernel functor:

.. code:: cpp

   #include <cuda/launch>
   #include <cstdio>

   struct kernel {
     template <typename Configuration>
     __device__ void operator()(Configuration conf, unsigned int thread_to_print) {
       if (cuda::gpu_thread.rank(cuda::grid, conf) == thread_to_print) {
         printf("Hello from the GPU\n");
       }
     }
   };

   void launch_kernel(cuda::stream_ref stream) {
     auto config = cuda::make_config(cuda::block_dims<128>(), cuda::grid_dims(4), cuda::cooperative_launch{});
     // It's enough to pass the configuration object once and launch will automatically pass it into the kernel
     cuda::launch(stream, config, kernel{}, 42);
   }

Example with extended device lambda:

.. code:: cpp

   #include <cuda/launch>
   #include <cstdio>

   void launch_kernel(cuda::stream_ref stream) {
     auto config = cuda::make_config(cuda::block_dims<128>(), cuda::grid_dims(4), cuda::cooperative_launch{});
     auto lambda = [](cuda::config conf, unsigned int thread_to_print) {
       if (cuda::gpu_thread.rank(cuda::grid, conf) == thread_to_print) {
         printf("Hello from the GPU\n");
       }
     };
     // Note that the configuration needs to be passed twice, unlike in other examples
     cuda::launch(stream, config, lambda, config, 42);
   }

|cuda_kernel_config|
-------------------------------------------------------------------------------
.. _cccl-runtime-launch-kernel-config:

|cuda_kernel_config| represents a kernel launch configuration combining hierarchy dimensions and launch
options. It should be created using |cuda_make_config| rather than being constructed directly.

A |cuda_kernel_config| provides:

- ``hierarchy()`` - Access to the hierarchy dimensions
- ``options()`` - Access to launch options
- ``combine(other_config)`` - Combine with another configuration
- ``combine_with_default(kernel)`` - Combine with default options from a kernel of a kernel functor accessed via
  ``kernel.default_config()``, equivalent to ``combine(kernel.default_config())``

Availability: libhipcxx 3.4

|cuda_make_config|
-------------------------------------------------------------------------------------------------
.. _cccl-runtime-launch-make-config:

|cuda_make_config| creates a kernel configuration from `hierarchy dimensions <cccl-runtime-hierarchy>` and
optional launch options. It can be called with:

- A hierarchy and options: ``make_config(hierarchy, option1, option2, ...)``
- Dimensions directly: ``make_config(grid_dims(...), block_dims<...>(), option1, option2, ...)``

In the last case, the dimensions arguments must come first, followed by options.

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/launch>
   #include <hip/hip_cooperative_groups.h>

   // Create config with cooperative launch
   auto config1 = cuda::make_config(cuda::grid_dims(256), cuda::cooperative_launch{});

   // Create config with dynamic shared memory
   auto config2 = cuda::make_config(
     cuda::block_dims<128>(),
     cuda::grid_dims(512),
     cuda::dynamic_shared_memory<float>(1024)
   );

   // Combine configurations, configuration that combine was called on is prioritized.
   auto config3 = config1.combine(config2);
   assert(cuda::gpu_thread.count(cuda::grid, config3) == 256 * 128);

   // Kernel functor can have a default configuration attached to it, that is later combined with the configuration passed to the launch.
   struct kernel {
     template <typename Configuration>
     __global__ void operator()(Configuration conf, unsigned int thread_to_print) {
       auto grid = cooperative_groups::this_grid();
       grid.sync();
       if (cuda::gpu_thread.rank(cuda::grid, conf) == thread_to_print) {
         printf("Hello from the GPU\n");
       }
     }

     auto default_config() const {
       return cuda::make_config(cuda::block_dims<128>(), cuda::cooperative_launch{});
     }
   };

   cuda::launch(stream, cuda::make_config(cuda::grid_dims(4)), kernel{});

Launch Options
--------------
.. _cccl-runtime-launch-options:

The launch API provides several launch options:

|cuda_cooperative_launch|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enables cooperative launch, restricting the grid to a number of blocks that can simultaneously execute on the device. This enables usage of ``cooperative_groups::grid_group::sync()`` in the kernel. This is a struct that can be default-constructed.
See `Cooperative groups <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html>`__.

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/launch>
   #include <hip/hip_cooperative_groups.h>

   template <typename Configuration>
   __global__ void kernel(Configuration conf) {
     auto grid = cooperative_groups::this_grid();
     grid.sync();
   }

   void launch(cuda::stream_ref stream) {
     auto config = cuda::make_config(cuda::block_dims<128>(), cuda::grid_dims(4), cuda::cooperative_launch{});
     cuda::launch(stream, config, kernel<decltype(config)>);
   }

|cuda_dynamic_shared_memory|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specifies dynamic shared memory configuration. It provides a type-safe way to specify shared memory content and later access it through the configuration object passed to the kernel.

- For non-array ``T`` (e.g., a struct), call |cuda_dynamic_shared_memory| with no size argument.
- For bounded array ``T[n]`` (e.g., ``int[10]``), call |cuda_dynamic_shared_memory| with no size argument.
- For unbounded array ``T[]`` (e.g., ``float[]``), pass the element count to |cuda_dynamic_shared_memory|.
- To opt in to non-portable dynamic shared memory sizes (greater than 48 KiB per block), pass
  :cpp:any:`cuda::non_portable` to |cuda_dynamic_shared_memory|. The maximum amount of shared memory per block depends
  on the AMD GPU and can be queried with ``cuda::device_attributes::max_shared_memory_per_block``.

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/launch>

   template <typename Configuration>
   __global__ void kernel(Configuration conf) {
     auto smem = cuda::dynamic_shared_memory(conf);
     // Use smem as span<T> in case of an array type or T& in case of a non-array type
   }

   void launch(cuda::stream_ref stream) {
     auto config = cuda::make_config(
     cuda::block_dims<128>(),
     cuda::grid_dims(4),
     cuda::dynamic_shared_memory<float[]>(1024)
     );
     cuda::launch(stream, config, kernel<decltype(config)>);
   }

   void launch_non_portable(cuda::stream_ref stream) {
     auto config = cuda::make_config(
     cuda::block_dims<128>(),
     cuda::grid_dims(4),
     cuda::dynamic_shared_memory<float[]>(14336, cuda::non_portable) // 56 KiB
     );
     cuda::launch(stream, config, kernel<decltype(config)>);
   }

|cuda_launch_priority|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specifies the priority launch option used when scheduling the kernel launch.
In libhipcxx this option is accepted for source compatibility but ignored, because HIP kernel launches have no
per-launch priority attribute. Use a stream created with the desired priority instead.

..
   Not supported in libhipcxx: the launch priority attribute is ignored on HIP.

   Overrides the priority specified in the stream.

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/launch>

   auto config = cuda::make_config(
     cuda::block_dims<128>(),
     cuda::grid_dims(4),
     cuda::launch_priority{0}
   );

|cuda_host_launch|
-------------------------------------------------------------------------------------------------
.. _cccl-runtime-launch-host-launch:

|cuda_host_launch| launches a host callable for a stream-ordered execution. The callable can be a lambda
function, a function pointer, or a callable object.
The callable and arguments are taken by value and stored for later execution. This requires a dynamic allocation to store the callable and arguments. If the callable is a function pointer or cuda::std::reference_wrapper and there are no arguments, the dynamic allocation is avoided.

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/launch>
   #include <iostream>

   cuda::host_launch(stream, [](int arg) {
     std::cout << "Callback executed" << '\n';
     std::cout << "Argument: " << arg << '\n';
   }, 42);

   // Passing by reference requires using cuda::std::ref without arguments to avoid dynamic allocation,
   // but the callable must live long enough for the callable to execute.
   int arg = 42;
   auto lambda = [&arg]() {
     std::cout << "Callback executed" << '\n';
     std::cout << "Argument: " << arg << '\n';
   };
   cuda::host_launch(stream, cuda::std::ref(lambda));
   stream.sync();
