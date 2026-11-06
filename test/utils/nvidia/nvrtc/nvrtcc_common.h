//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

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
    hipError_t result = x;                                          \
    if (result != hipSuccess)                                   \
    {                                                             \
      const char* msg;                                            \
      cuGetErrorName(result, &msg);                               \
      printf("\nCUDA ERROR: %s failed with error %s\n", #x, msg); \
      exit(1);                                                    \
    }                                                             \
  } while (0)

#define CUDA_API_CALL(x)                                                                \
  do                                                                                    \
  {                                                                                     \
    hipError_t err = x;                                                                \
    if (err != hipSuccess)                                                             \
    {                                                                                   \
      printf("\nCUDA ERROR: %s: %s\n", hipGetErrorName(err), hipGetErrorString(err)); \
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

  fprintf(stderr, "ERROR: Could not find an integer literal for '%s' on line '%s':\r\n", var.c_str(), line.c_str());
  exit(1);

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
__host__ __device__ int fake_main(int argc, char ** argv);
#define main fake_main

// extern "C" to stop the name from being mangled
extern "C" __global__ void main_kernel() {
    fake_main(0, nullptr);
}
)program";
