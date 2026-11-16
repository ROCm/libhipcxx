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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "hiprtcc_common.h"
#include <stdio.h>

// Arch configs are strings and bools determining architecture and ptx/sass compilation
using ArchConfig          = std::tuple<std::string, bool>;
constexpr auto archString = [](const ArchConfig& a) -> const auto& {
  return std::get<0>(a);
};
constexpr auto isArchReal = [](const ArchConfig& a) -> const auto& {
  return std::get<1>(a);
};

using ArgList = std::vector<std::string>;

using GpuProg = std::vector<char>;

// Takes arguments for building a file and returns the path to the output file
GpuProg hiprtc_build_prog(const std::string& testCu, const ArchConfig& config, const ArgList& argList)
{
  // Assemble arguments
  std::vector<const char*> optList;
  // Be careful with lifetimes here
  std::for_each(argList.begin(), argList.end(), [&](const auto& it) {
    optList.emplace_back(it.c_str());
  });
  
  // Use the translated architecture
  std::string gpu_arch("--offload-arch=" + archString(config));
  optList.emplace_back(gpu_arch.c_str());

  fprintf(stderr, "HIPRTC opt list:\r\n");
  for (const auto& it : optList)
  {
    fprintf(stderr, "  %s\r\n", it);
  }
  
  fprintf(stderr, "Compiling program...\r\n");
  hiprtcProgram prog;
  HIPRTC_SAFE_CALL(hiprtcCreateProgram(&prog, testCu.c_str(), "test.cu", 0, nullptr, nullptr));

  hiprtcResult compile_result = hiprtcCompileProgram(prog, optList.size(), optList.data());

  fprintf(stderr, "Collecting logs...\r\n");
  size_t log_size;
  HIPRTC_SAFE_CALL(hiprtcGetProgramLogSize(prog, &log_size));

  {
    std::unique_ptr<char[]> log{new char[log_size]};
    HIPRTC_SAFE_CALL(hiprtcGetProgramLog(prog, log.get()));
    printf("%s\r\n", log.get());
  }

  if (compile_result != HIPRTC_SUCCESS)
  {
    exit(1);
  }

  // NOTE(HIPRTC): take the code object straight from the compile. hiprtcLinkComplete
  // always targets the current device and ignores HIPRTC_JIT_TARGET, so linking would
  // need a GPU even with an explicit --offload-arch.
  size_t codeSize;
  HIPRTC_SAFE_CALL(hiprtcGetCodeSize(prog, &codeSize));
  GpuProg code(codeSize);
  HIPRTC_SAFE_CALL(hiprtcGetCode(prog, code.data()));

  HIPRTC_SAFE_CALL(hiprtcDestroyProgram(&prog));

  return code;
}
