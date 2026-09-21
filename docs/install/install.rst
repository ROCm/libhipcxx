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
  :description: libhipcxx installation
  :keywords: install, libhipcxx, AMD, ROCm, installation, TheRock, ROCm Core SDK

.. _installation:

*****************
Install libhipcxx
*****************

Starting with ROCm Core SDK 10.1, libhipcxx is distributed as part of the
ROCm Core SDK.

Before you begin, verify that your system is supported. For more information,
see `ROCm Core SDK components
<https://rocm.docs.amd.com/en/latest/about/release-components.html>`_.

For advanced workflows, source builds, or custom configurations, see
:doc:`Build from source <./source-build>`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

libhipcxx is included with the ROCm Core SDK on Linux. For the most complete
installation, we recommend that developers use the ``amdrocm-core-sdk`` meta
package.

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

   * ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
     suffix to install the latest available version.

   * ``<dev/devel>`` specifies that the package includes library files and
     headers. libhipcxx is header-only, so a development package is required.

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

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

The `TheRock <https://github.com/ROCm/TheRock>`_ build system also publishes
nightly builds for the ROCm Core SDK and its components, including libhipcxx.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`_ for details.
