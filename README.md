<!---
    MIT License

    Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

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
-->

# libhipcxx

libhipcxx is the C++ Standard Library for HIP. It provides an opt-in,
incremental, heterogeneous implementation of C++ Standard Library features
that work in both host and device code, along with extensions to those
features and abstractions that are fundamental to the HIP C++ programming
model. It is derived from [libcudacxx](https://github.com/NVIDIA/cccl/tree/main/libcudacxx)
and is part of ROCm.

libhipcxx is a header-only library.

## Documentation

The published documentation is the
[libhipcxx documentation](https://rocm.docs.amd.com/projects/libhipcxx/en/latest/).

The most current versions of the docs can be found here:

- [What libhipcxx is](docs/what-is-libhipcxx.rst)
- [Install libhipcxx](docs/install/install.rst), including how to add it to a CMake project
- [Build from source](docs/install/source-build.rst), including requirements
- [C++ Standard Library features](docs/conceptual/libhipcxx-standard-library-features.rst)
- [C++ Standard Library extensions](docs/conceptual/libhipcxx-extensions.rst)
- [Namespace hierarchy](docs/conceptual/libhipcxx-hip-abstractions.rst)
- [Run the tests](docs/how-to/run-libhipcxx-tests.rst)
- [Standard API](docs/libcudacxx/standard_api.rst) and [extended API](docs/libcudacxx/extended_api.rst)
- [Limitations and unsupported APIs](docs/reference/libhipcxx-limitations.rst)
- [Conformance and versioning](docs/reference/libhipcxx-conformance.rst)
- [License](docs/license.rst)

To contribute to the documentation, see
[Contributing to ROCm](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

The public source repository is [ROCm/libhipcxx](https://github.com/ROCm/libhipcxx).
