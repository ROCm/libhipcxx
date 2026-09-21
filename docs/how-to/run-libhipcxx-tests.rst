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
  :description: Running the libhipcxx test suite
  :keywords: libhipcxx, ROCm, tests, lit, HIPRTC, ninja, CI

.. _libhipcxx-run-tests:

*******************************************
How to run the libhipcxx tests
*******************************************

libhipcxx uses `lit <https://pypi.org/project/lit/>`_, the LLVM Integrated Tester, to run its host
and device tests. Running the tests requires a configured build directory and a GPU that libhipcxx
supports.

Before you begin, install the test dependencies described in the
:ref:`source-build prerequisites <libhipcxx-prerequisites>`, then configure
and build libhipcxx as described in :doc:`Build from source
<../install/source-build>`.

Running the tests with Ninja
============================

From the ``build`` directory, run:

.. code-block:: shell

    ninja check-hipcxx

Running the tests with the helper script
========================================

The ``utils/amd/linux/perform_tests.bash`` helper script runs the same suite. From the ``build``
directory, run:

.. code-block:: shell

    bash ../utils/amd/linux/perform_tests.bash

Whether the tests run with HIPRTC support depends on how you configured the build. If you passed
``-DLIBHIPCXX_TEST_WITH_HIPRTC=ON`` to CMake, the tests run with HIPRTC support enabled; otherwise
they run without it.

Running the tests with the CI scripts
=====================================

For automated testing, or when you want a predefined testing workflow, use the scripts in the ``ci``
directory. These scripts configure and run the tests for you, in both HIP and HIPRTC configurations.

Change to the ``ci`` directory:

.. code-block:: shell

    cd ci

To run the tests without HIPRTC, run:

.. code-block:: shell

    bash ./test_libhipcxx.sh

To run the tests with HIPRTC, run:

.. code-block:: shell

    bash ./hiprtc_libhipcxx.sh
