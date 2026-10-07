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
  :description: Learn how to run the libhipcxx test suite using lit, Ninja, or CI scripts for HIP and HIPRTC configurations on AMD GPUs.
  :keywords: libhipcxx, ROCm, tests, lit, HIPRTC, ninja, CI, test suite, AMD GPU

.. _libhipcxx-run-tests:

*******************
Run libhipcxx tests
*******************

The libhipcxx test suite verifies that the library's headers compile correctly and produce correct
results on AMD GPU hardware. Run the tests when contributing to libhipcxx, validating a build
against a specific ROCm version, or checking compatibility after a toolchain upgrade.

The suite uses `lit <https://pypi.org/project/lit/>`_, the LLVM Integrated Tester, and contains
approximately 2,400 test files organized into the following categories:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Test category
     - What it covers
   * - Standard library (``std/``)
     - Conforming ``cuda::std::`` implementations of C++ Standard Library headers, verified to
       compile and run in GPU device code. Covers atomics, concepts, containers, iterators,
       numerics, ranges, utilities, and more.
   * - Extended API (``cuda/``)
     - GPU-specific extensions in the ``cuda::`` namespace: ``atomic``,
       ``stream_ref``, ``memory_resource``, and others.
   * - Heterogeneous (``heterogeneous/``)
     - Objects shared across host and device, and interoperability between the ``cuda::`` and
       ``hip::`` namespaces.
   * - Public headers (``public_headers/``)
     - Confirms that each public header is self-contained and compiles as a standalone HIP
       translation unit.
   * - HIP aliasing (``hip/``)
     - Confirms that ``hip::std::`` and ``cuda::std::`` are interchangeable aliases.

HIP and HIPRTC configurations
=============================

The suite can run in two compilation modes:

- **HIP mode** (default): ``amdclang++`` compiles tests offline. Use this for standard development
  and validation.
- **HIPRTC mode**: Tests are compiled at runtime using the HIPRTC library, which is AMD's
  runtime compilation API. Use this when you are embedding libhipcxx headers in a
  runtime-compilation (JIT) pipeline and want to verify that headers work under that model.

To enable HIPRTC mode, pass ``-DLIBHIPCXX_TEST_WITH_HIPRTC=ON`` to CMake when configuring the
build.

Before you begin, install the test dependencies described in the
:ref:`source-build prerequisites <libhipcxx-prerequisites>`, then configure
and build libhipcxx as described in :doc:`Build from source <../install/source-build>`.

Choose a method
===============

Three methods are available, suited to different situations:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Method
     - Best for
     - Key differences
   * - Ninja
     - Iterative development. Fastest path when you already have a configured build.
     - Runs lit directly through the build system. No additional options. HIPRTC support
       is determined by your CMake configuration.
   * - Helper script
     - Development with more control. Use when you want to run a subset of tests, skip
       specific categories, or get a detailed pass/fail summary.
     - Automatically detects your GPU architecture. Limits parallelism to eight workers.
       Supports flags to run specific tests, skip categories, do dry runs, and produce
       verbose or per-test output. Prints a final percentage score.
   * - CI scripts
     - Automated pipelines or reproducing CI results locally. Use when you want a
       clean, predefined run without manually configuring CMake first.
     - Handles CMake configuration for you. Separate scripts for HIP and HIPRTC
       configurations make the two modes explicit.

Run the tests with Ninja
========================

From the ``build`` directory, run:

.. code-block:: shell

    ninja check-hipcxx

Run the tests with the helper script
====================================

The ``utils/amd/linux/perform_tests.bash`` helper script adds GPU architecture detection,
controlled parallelism, and a summary score on top of the same lit suite. From the ``build``
directory, run:

.. code-block:: shell

    bash ../utils/amd/linux/perform_tests.bash

To run a specific subset of tests rather than the full suite, pass the test paths as arguments:

.. code-block:: shell

    bash ../utils/amd/linux/perform_tests.bash std/atomics cuda/atomic

Useful flags include ``--verbose`` for full per-test output, ``--pretty`` for individual test
results, ``--dry-run`` to preview commands without executing them, and
``--skip-tests-runs`` to build without running (for example, to pre-warm a compiler cache).

HIPRTC support is determined by your CMake configuration. If you passed
``-DLIBHIPCXX_TEST_WITH_HIPRTC=ON`` when configuring, the tests run with HIPRTC enabled;
otherwise they run without it.

Run the tests with the CI scripts
=================================

The scripts in the ``ci`` directory configure and build libhipcxx before running the tests, so
they do not require a pre-existing build. Use them to reproduce CI results locally or to run
a clean end-to-end validation. HIP and HIPRTC configurations are separate scripts, making the
distinction explicit.

Change to the ``ci`` directory:

.. code-block:: shell

    cd ci

To run the tests without HIPRTC, run:

.. code-block:: shell

    bash ./test_libhipcxx.sh

To run the tests with HIPRTC, run:

.. code-block:: shell

    bash ./hiprtc_libhipcxx.sh

Verify the results
==================

All three methods exit with code ``0`` when all tests pass and a non-zero code when any test
fails. Each prints a lit summary at the end:

.. code-block:: none

    Testing Time: 42.3s
      Unsupported : 221
      Passed      : 2180
      Failed      : 3

The helper script additionally prints a percentage score, for example ``Score: 99.86%``. A
score of ``100.00%`` means every applicable test passed; unsupported tests (those marked
``UNSUPPORTED`` in the test file, such as HIPRTC-incompatible tests in HIP mode) do not count
against the score.

To check the exit code explicitly after any of the commands above, run:

.. code-block:: shell

    echo $?

A value of ``0`` indicates success.
