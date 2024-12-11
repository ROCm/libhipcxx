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


# libhipcxx: The C++ Standard Library for Your Entire System

> [!CAUTION] 
> This release is an *early-access* software technology preview. Running production workloads is *not* recommended.
***

**libhipcxx, is a HIP enabled C++ Standard Library for your entire system.**
It provides a heterogeneous implementation of the C++ Standard Library that
can be used in and between CPU and GPU code.

Libhipcxx provides a compatible interface to the C++ Standard Library, so you only need to add `hip/std` to the start of your Standard Library
include and `hip::` before any uses of `std::`:

```hip
#include <hip/hip_runtime.h>
#include <hip/std/atomic>
hip::std::atomic<int> x;
```

*IMPORTANT*: Please make sure to always include the header `hip/hip_runtime.h` *before* including any header from libhipcxx.

## `hip::` and `hip::std::`

When used with hipcc, libhipcxx facilities live in their own
  header hierarchy and namespace with the same structure as, but distinct from,
  the host compiler's Standard Library:

* `std::`/`<*>`: When using hipcc, this is your host compiler's Standard Library
      that works in `__host__` code only.
    With hipcc, libhip++ does not replace or interfere with host compiler's
      Standard Library.
* `hip::std::`/`<hip/std/*>`: Strictly conforming implementations of
      facilities from the Standard Library that work in `__host__ __device__`
      code.
* `hip::`/`<hip/*>`: Conforming extensions to the Standard Library that
      work in `__host__ __device__` code.
* `hip::device`/`<hip/device/*>`: Conforming extensions to the Standard
      Library that work only in `__device__` code.

```hip
// Standard C++, __host__ only.
#include <atomic>
std::atomic<int> x;

// HIP C++, __host__ __device__.
// Strictly conforming to the C++ Standard.
#include <hip/std/atomic>
hip::std::atomic<int> x;

// HIP C++, __host__ __device__.
// Conforming extensions to the C++ Standard.
#include <hip/atomic>
hip::atomic<int, hip::thread_scope_block> x;
```

## libhipcxx is Heterogeneous

libhipcxx works across your entire codebase, both in and
  across host and device code.
libhipcxx is a C++ Standard Library for your entire system, not just your CPU or
  GPU.
Everything in `hip::` is `__host__ __device__`.

libhipcxx facilities are designed to be passed between host and device code.
Unless otherwise noted, any libhipcxx object which is copyable or movable can be
  copied or moved between host and device code.

### `hip::device::`

A small number of libhipcxx facilities only work in device code, usually because
  there is no sensible implementation in host code.

Such facilities live in `hip::device::`.

## libhipcxx is Incremental

Today, the libhipcxx delivers a high-priority subset of the
  C++ Standard Library today, and each release increases the feature set.
But it is a subset; not everything is available today.

## Conformance

libhipcxx aims to be a conforming implementation of the
  C++ Standard, [ISO/IEC IS 14882], Clause 16 through 32.

## Experimental feature: "CUDA interoperability layer"

To minimize porting efforts for existing codes that use libcudacxx, 
namespace aliases for `hip::*` to `cuda::*` are provided. 
Furthermore, libhipcxx's headers are also made available in the folder
`cuda/*`.
Please see `examples/hip/concurrent_hash_table.hip` for an
example application that was ported from libcudacxx and that uses
this feature to minimize source code changes. 

# Requirements 
- CMake >=3.12
- ROCm with HIP >=6.2.0 
- AMD MI100, MI200, MI300, RDNA3 GPU/gfx1100 (NVIDIA GPUs are currently not supported)
- Linux OS (Windows is currently not supported)

For running the integrated LIT unit tests:
- Python 3
- lit 16.0.0 (more recent versions are currently not supported!)

# Build and Installation

The following commands can be run from the root directory to configure libhipcxx, build it and run the unit tests:

1) Create build directory
`mkdir build && cd build`
2) Run CMake to configure LIT testing
`cmake -DCMAKE_INSTALL_PREFIX=<path to install directory> ..`
3) Compile all headers that are part of the library
`make`
4) `make install`

# Running the Tests
To run the tests on host and device based on LIT, you can use
`make check-hipcxx`.

Alternatively, there is a helper script at `utils/amd/linux/perform_tests.bash` which can be used as follows:
1) Change directory to build directory: `cd build`
2) `bash ../utils/amd/linux/perform_tests.bash --skip-libcxx-tests`

# How to use libhipcxx in your CMake Project

Example `CMakeLists.txt`:
```
...
find_package(libhipcxx)
...
target_link_libraries(<your_target> PRIVATE libhipcxx::libhipcxx)
```
Make sure to set `CMAKE_PREFIX_PATH` when running CMake for your project, in case you installed libhipcxx in a non-default installation directory.

# Limitations/Unsupported Features/APIs
- Libhipcxx does not support for CUDA backend/NVIDIA hardware.
- Libhipcxx does not support the Windows OS.
- `hip::std::chrono::system_clock::now()` does not return a UNIX timestamp, host system clock and device system clock are not synchronized and they may run at different clock rates.
- The following APIs from [libcudacxx] are *NOT* supported in libhipcxx:

| Group                   | API Header                 | Description                                             |
| ----------------------- | -------------------------  | ------------------------------------------------------- |
| Synchronization Library | `<hip/std/latch>`       | Single-phase asynchronous thread-coordination mechanism |
| Synchronization Library | `<hip/std/barrier>`      | Multi-phase asynchronous thread-coordination mechanism  | 
| Synchronization Library | `<hip/std/semaphore>`    | Primitives for constraining concurrent access           |
| Extended Synchronization Library | `<hip/latch>`    | System-wide `hip::std::latch` single-phase asynchronous thread coordination mechanism.|
| Extended Synchronization Library | `<hip/barrier>`    | System-wide `hip::std::barrier` multi-phase asynchronous thread coordination mechanism.|
| Extended Synchronization Library | `<hip/semaphore>`    | System-wide primitives for constraining concurrent access.|
| Extended Synchronization Library | `<hip/pipeline>`    |  Coordination mechanisms to sequence asynchronous operations.|
| Extended Utility Library  | `<hip/functional>`         | Utility for proclaiming return types from device lambdas. |
| Extended Memory Access Properties Library  | `<hip/annotated_ptr>`         | Memory access properties for pointers. |

# License

libhipcxx is an open source project. It is derived from [libcudacxx] 
and [LLVM's libc++]. The original [libcudacxx] and [LLVM's libc++] are distributed under the [Apache License v2.0 with LLVM Exceptions]. Any new files and modifications made to exisiting files by AMD are distributed under MIT.

[libcudacxx]: https://github.com/nvidia/libcudacxx
[LLVM's libc++]: https://libcxx.llvm.org
[Apache License v2.0 with LLVM Exceptions]: https://llvm.org/LICENSE.txt
