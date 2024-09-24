<!-- MIT License
  -- 
  -- Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
  -- 
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  -- 
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  -- 
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
-->

# Requirements

All requirements are applicable to the `main` branch on GitHub.

## Usage Requirements

To use libhipcxx, you must meet the following
  requirements.

### System Software

Libhipcxx requires [ROCm] 6.0.0+.

### C++ Dialects

Libhipcxx supports the following C++ dialects:

- C++11
- C++14
- C++17

A number of features have been backported to earlier standards.
Please see the [API section] for more details.

### Host Compilers

Libhipcxx presently supports the following host compilers:

- GCC 5, 6, 7, 8, 9, and 10.

### Device Architectures

Libhipcxx supports the following AMD device
  architectures:

- gfx908 (MI100)
- gfx90a (MI210 + MI250)
- gfx940, gfx941 and gfx942 (MI300)

### Host Architectures

Libhipcxx supports the following host architectures:

- aarch64.
- x86-64.
- ppc64le.

### Host Operating Systems

Libhipcxx supports the following host operating systems:

- Linux.

## Build and Test Requirements

To build and test libhipcxx yourself, you will need the following in addition to
  the usage requirements:

- [CMake].
- [LLVM].
  - You do not have to build LLVM; we only need its CMake modules.
- [lit], the LLVM Integrated Tester.
  - We recommend installing lit using Python's pip package manager.

[ROCm]: https://rocm.docs.amd.com/en/latest/ 

[Standard API section]: ../standard_api.md
[Extended API section]: ../extended_api.md
[synchronization primitives section]: ../extended_api/synchronization_primitives.md
[changelog]: ../releases/changelog.md

[CMake]: https://cmake.org
[LLVM]: https://github.com/llvm
[lit]: https://pypi.org/project/lit/
