//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once
#include "nvrtc_runtime_api.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

#define NVRTC_SAFE_CALL(x)                                                                 \
  do                                                                                       \
  {                                                                                        \
    nvrtcResult result = x;                                                                \
    if (result != NVRTC_SUCCESS)                                                           \
    {                                                                                      \
      printf("\nNVRTC ERROR: %s failed with error %s\n", #x, nvrtcGetErrorString(result)); \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

#define CUDA_SAFE_CALL(x)                                         \
  do                                                              \
  {                                                               \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS)                                   \
    {                                                             \
      const char* msg;                                            \
      (void)cuGetErrorName(result, &msg);                         \
      printf("\nCUDA ERROR: %s failed with error %s\n", #x, msg); \
      exit(1);                                                    \
    }                                                             \
  } while (0)

#define CUDA_API_CALL(x)                                                                \
  do                                                                                    \
  {                                                                                     \
    cudaError_t err = x;                                                                \
    if (err != cudaSuccess)                                                             \
    {                                                                                   \
      printf("\nCUDA ERROR: %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

static void write_output_file(const char* data, size_t datasz, const std::string& file)
{
  std::ofstream ostr(file, std::ios::binary);
  assert(!!ostr);

  ostr.write(data, datasz);
  ostr.close();
}

static std::string load_input_file(const std::string& file)
{
    if (file == "-")
  {
    return std::string(std::istream_iterator<char>{std::cin}, std::istream_iterator<char>{});
  }
  else
  {
    std::ifstream istr(file);
    assert(!!istr);
    return std::string(std::istreambuf_iterator<char>{istr}, std::istreambuf_iterator<char>{});
  }
}

static int parse_int_assignment(const std::string& input, std::string var, int def)
{
  auto lineBegin = input.find(var);
  auto lineEnd   = input.find('\n', lineBegin);

  if (lineBegin == std::string::npos || lineEnd == std::string::npos)
  {
    return def;
  }

  std::string line(input.begin() + lineBegin, input.begin() + lineEnd);
  std::regex varRegex("^" + var + ".*?([0-9]+).*?$");
  std::smatch match;
  std::regex_match(line, match, varRegex);

  if (match.size())
  {
    return std::stoi(match[1].str(), nullptr);
  }

  return def;
}

struct RunConfig
{
  int threadCount = 1;
  int shmemSize   = 0;
};

static RunConfig parse_run_config(const std::string& input)
{
  return RunConfig{
    parse_int_assignment(input, "cuda_thread_count", 1),
    parse_int_assignment(input, "cuda_block_shmem_size", 0),
  };
}

// Fake main for adapting kernels
static const char* program = R"program(
// C++ standard library types needed for libhipcxx headers
// Guard against redefinition from system headers (use clang's guard)
#ifndef __CLANG_MAX_ALIGN_T_DEFINED
#define __CLANG_MAX_ALIGN_T_DEFINED
typedef struct {
  long long __clang_max_align_nonce1 __attribute__((__aligned__(__alignof__(long long))));
  long double __clang_max_align_nonce2 __attribute__((__aligned__(__alignof__(long double))));
} max_align_t;
#endif

// Taken from the reference implementation repo
// TODO(HIP/AMD): this is a temporary WAR to create leass file modifications.
// This should be only in the test_macros.h. Unfortunately many tests do not
// include this header.
#ifndef NV_IF_TARGET
#define NV_IF_TARGET NV_IF_TARGET
#endif
#ifndef NV_IS_HOST
#define NV_IS_HOST NV_IS_HOST
#endif
#ifndef NV_IS_DEVICE
#define NV_IS_DEVICE NV_IS_DEVICE
#endif

// NOTE(HIP/AMD): HIPRTC does not include hip/hip_runtime.h, so types and
// constants from the HIP runtime API are not available. Provide minimal stub
// definitions needed by libhipcxx headers (referenced via cuda_*->hip_* macro
// mappings in include/libhipcxx/__amd/cuda_runtime.h).
//
// hipSuccess MUST be a plain enumerator (NOT a '#define' to a parenthesized
// cast like '((hipError_t)0)'). The cast spelling causes a hard parse error
// wherever libhipcxx headers reference '::cudaSuccess' on a HIPRTC-reachable
// code path, because '::' requires an unqualified-id on the right and the
// expansion '::((hipError_t)0)' is unparseable. Mirroring the real HIP
// runtime's '<hip/hip_runtime_api.h>:enum hipError_t { hipSuccess = 0, ... }'
// shape via an enum typedef makes 'hipSuccess' an identifier and lets the
// upstream '::cudaSuccess' spelling parse unchanged.
typedef enum hipError_t {
  hipSuccess = 0,
} hipError_t;
typedef enum hipMemoryType {
  hipMemoryTypeUnregistered = 0,
  hipMemoryTypeHost         = 1,
  hipMemoryTypeDevice       = 2,
  hipMemoryTypeManaged      = 3,
  hipMemoryTypeArray        = 10,
  hipMemoryTypeUnified      = 11
} hipMemoryType;
typedef struct hipPointerAttribute_t {
  enum hipMemoryType type;
  int device;
  void* devicePointer;
  void* hostPointer;
  int isManaged;
  unsigned allocationFlags;
} hipPointerAttribute_t;
// Stub declarations for runtime APIs (not actually callable from device code,
// but allow code paths that reference them to compile).
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);
hipError_t hipGetLastError(void);
// __managed__ qualifier is hipcc-specific; provide empty definition under HIPRTC.
#ifndef __managed__
#  define __managed__
#endif

// Define all macros required for reverse hipification here
#ifndef __nv_bfloat16
#  define __nv_bfloat16 __hip_bfloat16
#endif
#ifndef __nv_bfloat16_raw
#  define __nv_bfloat16_raw __hip_bfloat16_raw
#endif
#ifndef __nv_bfloat162
#  define __nv_bfloat162 __hip_bfloat162
#endif

__host__ __device__ void __trap(){
    __hip_assert(false);
}

// NOTE(HIP/AMD): HIPRTC doesn't provide <cassert>, so define assert as a polyfill
// using __builtin_trap which is available in HIPRTC device code. Using a function
// expression form so it works in ternary expressions like `(cond ? (void)0 : assert(...))`.
#ifndef assert
#  ifdef NDEBUG
#    define assert(expr) ((void)0)
#  else
__host__ __device__ inline void __libhipcxx_assert_fail() { __builtin_trap(); }
#    define assert(expr) ((expr) ? (void)0 : __libhipcxx_assert_fail())
#  endif
#endif

__host__ __device__ int fake_main(int argc, char ** argv);
#if defined(__HIP_PLATFORM_AMD__)
#define main __device__ __host__ fake_main
#else
#define main fake_main
#endif

// extern "C" to stop the name from being mangled
extern "C" __global__ void main_kernel() {
    fake_main(0, nullptr);
}
)program";
