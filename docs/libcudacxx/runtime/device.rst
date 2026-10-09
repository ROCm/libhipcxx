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
  :description: API reference for cuda::device_ref, cuda::devices and the device attribute queries in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, device_ref, devices, device attributes, peer access

.. _cccl-runtime-device:

Devices
========

This page documents ``cuda::device_ref``, ``cuda::devices`` and ``cuda::device_attributes``, which describe and query the
HIP devices available in the system.

:cpp:class:`cuda::device_ref`
-------------------------------
.. _cccl-runtime-device-device-ref:

:cpp:class:`cuda::device_ref` is a lightweight, non-owning handle to a HIP device ordinal. It allows to query
information about a device and serves as an argument to other runtime APIs which are tied to a specific device.
It offers:

- ``get()``: native device ordinal
- ``name()``: device name
- ``init()``: initialize the device context (a no-op in libhipcxx, the HIP runtime initializes devices implicitly)
- ``peers()``: list peers for which peer access can be enabled
- ``has_peer_access_to(cuda::device_ref)``: query if peer access can be enabled to the given device
- ``attribute(attr)`` / ``attribute<::hipDeviceAttribute_t>()``: attribute queries

Availability: libhipcxx 3.4

:cpp:var:`cuda::devices`
----------------------------
.. _cccl-runtime-device-devices:

:cpp:var:`cuda::devices` is a random-access view of all available HIP devices in the form of
:cpp:class:`cuda::device_ref` objects. It
provides indexing, size, and iteration for use
in range-based loops.

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/devices>
   #include <iostream>

   void print_devices() {
     for (auto& dev : cuda::devices) {
       std::cout << "Device " << dev.get() << ": " << dev.name() << '\n';
     }
   }

Device attributes
-----------------
.. _cccl-runtime-device-attributes:

``cuda::device_attributes`` provides strongly-typed attribute query objects usable with
:cpp:func:`cuda::device_ref::attribute`. The attributes map to the corresponding ``hipDeviceAttribute_t`` values
queried with ``hipDeviceGetAttribute``. Selected examples:

- ``multiprocessor_count``
- ``concurrent_managed_access``
- ``clock_rate``

..
   Not supported in libhipcxx, see reference/libhipcxx-limitations.rst.
   Compute capability queries are not supported on HIP; numa_id is not available on HIP.

   - ``compute_capability``
   - ``numa_id``

Availability: libhipcxx 3.4

Example:

.. code:: cpp

   #include <cuda/devices>

   int get_max_blocks_on_device(cuda::device_ref dev) {
     return cuda::device_attributes::multiprocessor_count(dev) * cuda::device_attributes::max_blocks_per_multiprocessor(dev);
   }

..
   Not supported in libhipcxx, see reference/libhipcxx-limitations.rst.
   The architecture traits are keyed on NVIDIA GPU architectures (cuda::arch_id), and
   cuda::device::current_arch_id / current_arch_traits / current_compute_capability are not supported on HIP.

   :cpp:any:`cuda::arch_traits`
   --------------------------------
   .. _cccl-runtime-device-arch-traits:

   Per-architecture trait accessors providing limits and capabilities common to all devices of an architecture.
   Compared to ``cuda::device_attributes``, :cpp:any:`cuda::arch_traits` provide a compile-time accessible
   structure that describes common characteristics of all devices of an architecture, while attributes are run-time
   queries of a single characteristic of a specific device.

   - :cpp:any:`cuda::arch_traits` and :cpp:any:`cuda::arch_traits_for` (compile-time and run-time forms).
   - Returns a :cpp:struct:`cuda::arch_traits_t` with fields like
     ``max_threads_per_block``, ``max_shared_memory_per_block``, ``cluster_supported`` and other capability flags.
   - Traits for the current architecture can be accessed with :cpp:func:`cuda::device::current_arch_traits`

   Availability: libhipcxx 3.4

   Example:

   .. code:: cpp

      #include <cuda/devices>

      template <cuda::arch_id Arch>
      __device__ void fn() {
        auto traits = cuda::arch_traits<Arch>();
        if constexpr (traits.cluster_supported) {
          // cluster specific code
        } else {
          // non-cluster code
        }
      }

      __global__ void kernel() {
        fn<cuda::arch_id::sm_90>();
      }
