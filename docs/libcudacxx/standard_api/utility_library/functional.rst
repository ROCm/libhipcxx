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
  :description: Documents the cuda::std function objects available in libhipcxx and the omission of std::function, std::bind, and std::hash.
  :keywords: libhipcxx, ROCm, HIP, C++, functional, function objects, std::function, std::bind, std::hash

.. _libcudacxx-standard-api-utility-functional:

``<cuda/std/functional>``
=========================

This page documents the ``cuda::std`` functional utilities available in libhipcxx, including function objects and wrappers, with notes on omissions from the standard.

See the documentation of the standard header `\<functional\> <https://en.cppreference.com/w/cpp/header/functional>`_

Omissions
---------

The following facilities from section `functional.syn <https://eel.is/c++draft/functional.syn>`_ of the C++ Standard
are not available in libhipcxx:

- `std::function <https://en.cppreference.com/w/cpp/utility/functional/function>`_, a polymorphic function object
  wrapper.
- `std::bind <https://en.cppreference.com/w/cpp/utility/functional/bind>`_ and its placeholders, a generic function
  object binder.
- `std::hash <https://en.cppreference.com/w/cpp/utility/hash>`_, a hash function object.
