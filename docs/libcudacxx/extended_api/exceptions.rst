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
  :description: Description of exception handling in host and device code and of the cuda::cuda_error exception class in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, exceptions, cuda_error, terminate, device code, CCCL_DISABLE_EXCEPTIONS

.. _libcudacxx-extended-api-exceptions:

Exception Handling
==================

Standard C++ exception handling (``try``, ``catch``, ``throw``) is not supported in HIP device code, while it is enabled by default in host code.

**Device code**

``libhipcxx`` maps exceptions to ``cuda::std::terminate()`` calls in device code, which traps and terminates the kernel.

**Host code**

``libhipcxx`` allows users to manually disable exceptions in host code in two ways:

- By defining ``CCCL_DISABLE_EXCEPTIONS`` before including any library headers.
- By compiling with ``-fno-exceptions`` compiler flag with ``gcc`` or ``clang``, or ``/EH-`` compiler flag with ``msvc``.

If exceptions are disabled, a ``throw`` exception is translated into a `cuda::std::terminate() <https://en.cppreference.com/w/cpp/error/terminate.html>`__ call, which terminates the program.

``cuda::cuda_error``
--------------------

Exception class thrown when a HIP runtime error is encountered. It inherits from ``std::runtime_error``.

.. code-block:: cpp

    class cuda_error : public std::runtime_error
    {
    public:
        cuda_error(hipError_t status, const char* msg);

        hipError_t status() const noexcept;
    };
