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
   :description: Build and install libhipcxx from source
   :keywords: install, building, libhipcxx, AMD, ROCm, TheRock, source code, cmake, Linux

.. _build-from-source:

***************************
Build libhipcxx from source
***************************

To build libhipcxx as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`_.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build libhipcxx standalone using the following
instructions.

.. _libhipcxx-prerequisites:

Prerequisites
=============

On Linux, `ROCm <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_ must
be installed before libhipcxx is built.

libhipcxx has the following prerequisites:

* `CMake <https://cmake.org/>`_ version 3.21 or higher
* `hipcc <https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html>`_
* A C++17 or C++20 host toolchain

libhipcxx has these additional prerequisites to build and run the tests:

* `LLVM <https://github.com/llvm/llvm-project>`_ version 18.1.8 or higher.
  Only its CMake modules are required.
* `lit <https://pypi.org/project/lit/>`_ version 18.1.8
* Ninja
* sccache

.. _libhipcxx-get-source:

Get the libhipcxx source code
=============================

The libhipcxx source code is available from the `libhipcxx GitHub repository
<https://github.com/ROCm/libhipcxx>`_. Clone the repository:

.. code-block:: shell

   git clone https://github.com/ROCm/libhipcxx.git
   cd libhipcxx

Then use ``git checkout`` to check out the branch you need.

The ``amd-develop`` branch is intended for users who want to preview new
features or contribute to the libhipcxx code base.

If you don't intend to contribute to the libhipcxx code base and won't be
previewing features, use a branch that matches the version of ROCm installed
on your system.

.. _libhipcxx-build-cmake:

Build with CMake
================

Set ``CXX`` to ``hipcc`` and set ``CMAKE_CXX_COMPILER`` to hipcc's absolute
path. For example:

.. code-block:: shell

   CXX=hipcc
   CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc

Create the ``build`` directory under the ``libhipcxx`` root directory, then
change directory to the ``build`` directory:

.. code-block:: shell

   mkdir build
   cd build

Generate the libhipcxx build files using the ``cmake`` command:

.. code-block:: shell

   cmake -GNinja ../. [-D<OPTION1=VALUE1> [-D<OPTION2=VALUE2>] ...]

The build options are:

* ``LIBHIPCXX_TEST_WITH_HIPRTC``: Set this to ``ON`` to build and run the
  tests with HIPRTC support. Default is ``OFF``.
* ``LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS``: Set this to ``OFF`` to skip
  configuring the lit test suite. Default is ``ON``.
* ``libhipcxx_ENABLE_INSTALL_RULES``: Set this to ``OFF`` to skip generating
  install rules. Default is ``ON`` when libhipcxx is the top-level project.
* ``libhipcxx_ENABLE_CODEGEN``: Set this to ``ON`` to enable the atomics
  backend code generation and its tests. Default is ``OFF``.
* ``CMAKE_HIP_ARCHITECTURES``: Set this to the GPU architectures to build for,
  for example ``gfx942``. ``AMDGPU_TARGETS`` and ``GPU_TARGETS`` are also
  accepted.
* ``CMAKE_INSTALL_PREFIX``: Set this to the installation directory. For
  example, use ``/opt/rocm`` to install alongside ROCm.

Build libhipcxx using the generated build files:

.. code-block:: shell

   ninja

After you've built libhipcxx, you can optionally generate TGZ, ZIP, and DEB
packages:

.. code-block:: shell

   cpack .

To generate an RPM package, run:

.. code-block:: shell

   cpack -G RPM .

Finally, install libhipcxx:

.. code-block:: shell

   ninja install
