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
   :description: Learn what libhipcxx is, how it fits in the ROCm ecosystem, its features, key components, and who benefits from using this C++ Standard Library for HIP.
   :keywords: libhipcxx, ROCm, HIP, C++, standard library, heterogeneous computing, AMD GPU, CUDA alternative, HIP C++, atomic, synchronization, header-only

.. _what-is-libhipcxx:

**********************
What is libhipcxx?
**********************

libhipcxx is the C++ Standard Library for HIP. It provides an opt-in, incremental,
heterogeneous implementation of C++ Standard Library features that work in both host
and device code, along with extensions to those features and abstractions that are
fundamental to the HIP C++ programming model.

libhipcxx reduces the friction between writing standard C++ and writing
high-performance HIP code. Without it, developers face a choice:

* Use the Standard Library in host code only, duplicating logic across the CPU and
  GPU code paths.
* Write custom device-side implementations of standard utilities from scratch.

libhipcxx eliminates both options by providing a single set of headers that work
everywhere. Code written against ``cuda::std::atomic<T>`` compiles and runs
correctly whether it is called from the host, from a HIP kernel, or from a function
marked ``__host__ __device__``.

A secondary benefit is portability across GPU vendors. libhipcxx is derived from
libcudacxx, NVIDIA's equivalent library for CUDA. Code written against the
``cuda::std::`` or ``cuda::`` API is source-compatible between the two libraries,
which reduces the cost of porting between AMD and NVIDIA platforms.

Where libhipcxx fits in ROCm
=============================

ROCm is AMD's open-source GPU computing platform. It is organized into layers:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Layer
     - Role
   * - Driver and runtime (`HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/>`__)
     - GPU management, kernel launch, memory allocation
   * - **libhipcxx** (this library)
     - C++ Standard Library primitives for heterogeneous code
   * - Domain libraries (`rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/>`__, `MIOpen <https://rocm.docs.amd.com/projects/MIOpen/en/latest/>`__, `rocFFT <https://rocm.docs.amd.com/projects/rocFFT/en/latest/>`__, and so on)
     - Optimized domain-specific algorithms built on the runtime
   * - Frameworks (`PyTorch for ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html>`__, `TensorFlow for ROCm <https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/tensorflow-compatibility.html>`__, and so on)
     - High-level ML and scientific computing frameworks

libhipcxx sits between the HIP runtime and higher-level domain libraries. It
provides the building blocks — atomics, synchronization, type utilities — that
both application code and library code need when working at the HIP level.

libhipcxx ships as part of the ROCm Core SDK starting with ROCm Core SDK 10.1.
It is also available in the ``amdrocm-ccl`` package alongside rocThrust, hipCUB,
and rocPRIM.

Definition and scope
====================

The C++ Standard Library (`sometimes called "the STL" <https://stackoverflow.com/questions/5205491/whats-the-difference-between-stl-and-c-standard-library>`_) is the set of headers such as
``<atomic>``, ``<type_traits>``, and ``<vector>`` that ship with every C++ compiler.
These headers are not usable in GPU device code because:

* They lack the ``__host__ __device__`` annotations that HIP requires for code
  compiled for both CPU and GPU.
* Their implementations often depend on OS services or compiler intrinsics that are
  unavailable on the GPU.

libhipcxx solves this by providing parallel versions of Standard Library headers
that carry the necessary annotations and use GPU-safe implementations. The library
is:

* **Opt-in** — it does not replace the host compiler's Standard Library. Anything
  in ``std::`` continues to come from the host compiler; libhipcxx only activates
  when you include its headers.
* **Incremental** — it does not provide a complete Standard Library implementation.
  It focuses on the subset most useful for heterogeneous code: atomics, type traits,
  containers, and math utilities.
* **Heterogeneous** — the Standard Library subset it provides works in ``__host__`` code,
  ``__device__`` code, and code that passes data between the two. A small number of
  extensions in ``cuda::device::`` are device-only.

Relationship to libcudacxx
==========================

libhipcxx is a fork of libcudacxx, NVIDIA's C++ Standard Library for CUDA. The
two libraries share the same API surface. Code written against ``cuda::std::`` or
``cuda::`` compiles against both:

* libhipcxx on AMD GPUs via HIP
* libcudacxx on NVIDIA GPUs via CUDA

This shared API is intentional. It allows library authors to write portable
heterogeneous C++ that runs on both vendors' hardware with no source-level changes.
Differences between the two libraries exist primarily at the hardware level — for
example, libhipcxx targets AMD GPU hardware and does not support CUDA-specific
backends.

Not every libcudacxx API is available in libhipcxx. For the list of unsupported
APIs, see :ref:`libhipcxx-limitations`.

How libhipcxx works
===================

libhipcxx is a header-only library. There is nothing to compile or link against:
you add the libhipcxx include directory to your project and include headers directly.

Header mapping
--------------

libhipcxx maps Standard Library headers to GPU-safe equivalents by adding a path
prefix. To use a standard header in host and device code, add ``cuda/std/`` or
``hip/std/`` to the beginning of the include path:

.. code-block:: cpp

   // Standard C++, host only
   #include <atomic>
   std::atomic<int> x;

   // libhipcxx, host and device — cuda:: spelling
   #include <cuda/std/atomic>
   cuda::std::atomic<int> y;

   // libhipcxx, host and device — hip:: spelling (identical result)
   #include <hip/std/atomic>
   hip::std::atomic<int> z;

The ``cuda::std::`` / ``<cuda/std/*>`` and ``hip::std::`` / ``<hip/std/*>`` spellings
are interchangeable aliases that resolve to the same headers and the same types.
Use the ``cuda::`` spelling for code that also targets NVIDIA GPUs via libcudacxx,
or the ``hip::`` spelling for code that is AMD-only.

Extensions beyond the standard
-------------------------------

For capabilities that have no C++ Standard equivalent, libhipcxx provides
*extensions* — additional APIs in the ``cuda::`` (or ``hip::``) namespace. These
extensions add GPU-specific concepts such as thread scope to standard primitives.
To use an extension, drop the ``std`` from the path:

.. code-block:: cpp

   // Extension: atomic with explicit thread scope
   #include <cuda/atomic>
   cuda::atomic<int, cuda::thread_scope_device> counter;

Extensions are always opt-in and composable with conforming APIs.

Namespace structure
-------------------

libhipcxx uses three namespace layers. Each layer builds on the previous one and
has a broader execution scope:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Namespace
     - Header prefix
     - Where it runs
   * - ``std::``
     - ``<*>``
     - Host code only (from your host compiler)
   * - ``cuda::std::`` / ``hip::std::``
     - ``<cuda/std/*>`` / ``<hip/std/*>``
     - Host and device; conforming Standard Library subset
   * - ``cuda::`` / ``hip::``
     - ``<cuda/*>`` / ``<hip/*>``
     - Host and device; Standard Library extensions
   * - ``cuda::device::`` / ``hip::device::``
     - ``<cuda/warp>``, ``<cuda/work_stealing>``
     - Device code only; extensions requiring GPU hardware features


Features and components
=======================

Standard Library subset
-----------------------

libhipcxx makes the following Standard Library headers available in device code.
See :ref:`Standard API <libcudacxx-standard-api>` for the complete list and
per-header feature availability by C++ standard version.

Key supported headers include:

* **Atomics**: ``<cuda/std/atomic>``
* **Type support**: ``<cuda/std/type_traits>``, ``<cuda/std/limits>``,
  ``<cuda/std/cstdint>``
* **Utilities**: ``<cuda/std/tuple>``, ``<cuda/std/optional>``,
  ``<cuda/std/expected>``, ``<cuda/std/variant>``
* **Containers**: ``<cuda/std/array>``, ``<cuda/std/span>``,
  ``<cuda/std/mdspan>``
* **Numerics**: ``<cuda/std/bit>``, ``<cuda/std/complex>``,
  ``<cuda/std/numbers>``
* **Ranges**: ``<cuda/std/ranges>`` (requires C++20)

Extended API
------------

The Extended API (``cuda::`` namespace) adds GPU-specific capabilities that have
no Standard Library equivalent. Key extensions include:

* **Thread-scoped atomics** — ``cuda::atomic<T, Scope>`` and
  ``cuda::atomic_ref<T, Scope>`` allow specifying the scope of memory ordering:
  block, device, or system. This maps directly to the AMD GPU memory model and
  enables fine-grained control over coherence.
* **Math extensions** — utilities such as ``cuda::ceil_div`` and ``cuda::ilog2``
  that fill gaps in the Standard Library for GPU numeric code.

Who should use libhipcxx
========================

libhipcxx is intended for:

* **HIP C++ developers** who want to use familiar Standard Library types such as
  ``atomic<T>``, ``optional<T>``, or ``mdspan`` in device code without writing
  their own implementations.
* **Library authors** targeting AMD GPUs who need portable synchronization
  primitives or type utilities in headers shared between host and device code.
* **CUDA developers porting to ROCm** whose existing code already uses libcudacxx.
  Because the two libraries share the same API, source-level changes are minimal.
* **Researchers and scientists** writing custom HIP kernels who need standard
  numeric utilities or memory coordination primitives.

libhipcxx is not a replacement for domain libraries. If you need high-performance
matrix multiplication, use rocBLAS. If you need deep learning primitives, use
MIOpen. libhipcxx provides the low-level C++ building blocks those libraries — and
your own kernels — are built on top of.

Getting started
===============

* To install libhipcxx as part of the ROCm Core SDK, see :doc:`Install libhipcxx <../install/install>`.
* To add libhipcxx to an existing CMake project, see :ref:`Add libhipcxx to a CMake project <libhipcxx-use-in-a-project>`.
* To understand the available APIs, see the :ref:`Standard API <libcudacxx-standard-api>` and :ref:`Extended API <libcudacxx-extended-api>` reference sections.
