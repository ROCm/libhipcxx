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
  :description: Documents cuda::std::functional in libhipcxx, explaining the omission of std::function, std::bind, and std::hash due to heterogeneous implementation constraints.
  :keywords: libhipcxx, ROCm, HIP, C++, functional, function objects, std::function, std::bind, std::hash

.. _libcudacxx-standard-api-utility-functional:

``<cuda/std/functional>``
=========================

This page documents the ``cuda::std`` functional utilities available in libhipcxx, including function objects and wrappers, with notes on omissions from the standard.

See the documentation of the standard header `\<functional\> <https://en.cppreference.com/w/cpp/header/functional>`_

Omissions
---------

The following facilities in section
`functional.syn <https://eel.is/c++draft/functional.syn>`_ of ISO/IEC
IS 14882 (the C++ Standard) are not available in the NVIDIA C++ Standard
Library today:

- `std::function <https://en.cppreference.com/w/cpp/utility/functional/function>`_
   - Polymorphic function object wrapper.
- `std::bind <https://en.cppreference.com/w/cpp/utility/functional/bind>`_
   - Generic function object binder / lambda facility.
- `std::hash <https://en.cppreference.com/w/cpp/utility/hash>`_
   - Hash function object.

std::function
~~~~~~~~~~~~~~~~~

`std::function <https://en.cppreference.com/w/cpp/utility/functional/function>`_
is a polymorphic function object wrapper. Implementing it requires both
polymorphism (either hand built dispatch tables or the use of C++
virtual functions) and memory allocation. This means that it is
non-trivial to implement a heterogeneous version of this facility today.
As such, we have deferred it.

std::bind
~~~~~~~~~~~~~

`std::bind <https://en.cppreference.com/w/cpp/utility/functional/bind>`_
is a general-purpose function object binder / lambda facility. It relies
on constexpr global variables for placeholders, which presents
heterogeneous implementation challenges today due to how global
variables work in NVCC. E.g. We cannot easily ensure the placeholders
are the same object with the same address in host and device code.
Therefore, we've decided to hold off on providing this feature for now.

std::hash
~~~~~~~~~~~~~

`std::hash <https://en.cppreference.com/w/cpp/utility/hash>`_ is a
function object which hashes entities. While this is an important
feature, it is also important that we pick a hash implementation that
makes sense for GPUs. That implementation might be different from the
default that the upstream libc++ uses. Further research and
investigation is required before we can provide this feature.
