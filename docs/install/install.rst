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
  :description: Learn how to install libhipcxx as part of the ROCm Core SDK or ROCm CCL package on Linux, with package installation instructions for all major distributions.
  :keywords: install, libhipcxx, AMD, ROCm, installation, TheRock, ROCm Core SDK, apt, dnf, zypper, Linux

.. _installation:

*****************
Install libhipcxx
*****************

Starting with ROCm Core SDK 10.1, libhipcxx is distributed as part of the
ROCm Core SDK.

Before you begin, verify that your system is supported. For more information,
see `ROCm compatibility matrix
<https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>`_.

For advanced workflows, source builds, or custom configurations, see
:doc:`Build from source <./source-build>`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

libhipcxx is included with the ROCm Core SDK on Linux. For the most complete
installation, use the ``amdrocm-core-sdk`` meta package.

For instructions, see `Install AMD ROCm
<https://rocm.docs.amd.com/en/latest/install/rocm.html>`_. Use the selector
panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install ROCm C++ template libraries on Linux
============================================

Alternatively, if you want to install libhipcxx as part of the ROCm CCL
package (a subset of the ROCm Core SDK ``amdrocm-core-sdk``) without
additional ROCm libraries and tools, install the ``amdrocm-ccl`` development
package. This includes libhipcxx, rocThrust, hipCUB, and rocPRIM.

1. Complete the `ROCm installation prerequisites
   <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_ to install
   dependencies and configure GPU access permissions.

2. Install the ROCm CCL development package that matches your desired ROCm
   version and AMD GPU architecture. Package names use the following format:

   .. code-block:: shell-session

      amdrocm-ccl<dev/devel><rocm_version>-<llvm_target>

   Where:

   * ``<dev/devel>`` specifies that the package includes library files and
     headers. libhipcxx is header-only, so a development package is required.

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
     suffix to install the latest available version.

   * ``<llvm_target>`` (starting with ``gfx``) is used if you are installing
     for a single AMD GPU architecture. Omit this suffix to install for all
     architectures at the cost of disk space.

   For example, to install the latest ROCm CCL development package release for
   all supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-ccl-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-ccl-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-ccl-devel

.. _install-nightly:

Install a nightly build
=======================

`TheRock <https://github.com/ROCm/TheRock>`_ build system also publishes
nightly builds for the ROCm Core SDK and its components, including libhipcxx.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`_ for details.

.. _libhipcxx-use-in-a-project:

Add libhipcxx to a CMake project
=================================

libhipcxx is a header-only library, so adding it to your project means making its headers
discoverable by the compiler — there is no shared library to link against and no ABI compatibility
concern from libhipcxx itself. After this step, any target you link against
``libhipcxx::libhipcxx`` can include libhipcxx headers without specifying the include path manually.

Prerequisites
-------------

ROCm must be installed before configuring your project. libhipcxx ships as part of ROCm, so if
ROCm is installed at its default prefix (``/opt/rocm``), no separate libhipcxx install step is
needed. CMake integration also does not configure the HIP compiler for you; your
``CMakeLists.txt`` must separately enable HIP, for example with ``enable_language(HIP)`` or by
setting ``CMAKE_HIP_COMPILER``.

CMake integration
-----------------

To use libhipcxx in your own project, add the following lines to your ``CMakeLists.txt`` file:

.. code-block:: cmake

    # "/opt/rocm" - default ROCm install prefix
    find_package(libhipcxx REQUIRED)

    # ...

    # include the libhipcxx headers
    target_link_libraries(your_target PRIVATE libhipcxx::libhipcxx)

``libhipcxx::libhipcxx`` is an interface target, so linking against it only adds the libhipcxx
include directories to your target.

If you installed libhipcxx into a non-default location, set ``CMAKE_PREFIX_PATH`` to that install
directory when you configure your project:

.. code-block:: shell

    cmake -DCMAKE_PREFIX_PATH=<path to libhipcxx install directory> ..

Include the headers
-------------------

To use a Standard Library facility in host and device code, add ``cuda/std/`` to the start of the
include and ``cuda::`` before the use of ``std::``:

.. code-block:: cpp

    #include <cuda/std/atomic>

    cuda::std::atomic<int> x;

To use an extension to a Standard Library facility, drop the ``std``:

.. code-block:: cpp

    #include <cuda/atomic>

    cuda::atomic<int, cuda::thread_scope_device> x;

You can also write these as ``hip/std/`` and ``hip::std::``, or ``hip/`` and ``hip::``. Both
spellings resolve to the same headers and can be used interchangeably. For an explanation of when to
use each namespace, see
:doc:`Namespace hierarchy in libhipcxx <../conceptual/libhipcxx-hip-abstractions>`.

If you are not using CMake, add the libhipcxx include root to your compiler flags directly.
ROCm installs the libhipcxx headers under ``/opt/rocm/include/hipccl``, which is not on the
compiler's default search path. Pass the GPU architecture to compile for with ``--offload-arch``:

.. code-block:: shell

    amdclang++ -std=c++17 --offload-arch=gfx942 -I/opt/rocm/include/hipccl -c main.hip