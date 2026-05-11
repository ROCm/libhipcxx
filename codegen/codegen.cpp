//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "generators/compare_and_swap.h"
#include "generators/exchange.h"
#include "generators/fence.h"
#include "generators/fetch_ops.h"
#include "generators/header.h"
#include "generators/hip_intrinsics.h"
#include "generators/ld_st.h"

using namespace std::string_literals;

namespace
{
// Top-of-file usage banner. Mirrors the upstream-NV invocation shape and
// adds a HIP emit-mode that produces 'atomic_hip_generated.h' (instead of
// 'cuda_ptx_generated.h'). Both modes share argv[1] as the output path.
void print_usage(const char* argv0)
{
  std::cerr << "Usage:\n"
            << "  " << argv0 << " [<out.h>]              # emit cuda_ptx_generated.h (NV)\n"
            << "  " << argv0 << " --hip [<out.h>]        # emit atomic_hip_generated.h (HIP)\n"
            << "When <out.h> is omitted the generated source is written to stdout.\n";
}
} // namespace

int main(int argc, char** argv)
{
  // Tiny ad-hoc CLI: '--hip' (or '-h') as the FIRST argument switches to the
  // HIP emitter; otherwise we keep the upstream behaviour where argv[1] (if
  // present) is the output path. Kept deliberately minimal -- adding a
  // dependency like cxxopts/CLI11 would pull a third-party library into the
  // codegen tool just for one flag.
  bool emit_hip            = false;
  const char* output_path  = nullptr;

  for (int i = 1; i < argc; ++i)
  {
    if (std::strcmp(argv[i], "--hip") == 0)
    {
      emit_hip = true;
    }
    else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
    {
      print_usage(argv[0]);
      return 0;
    }
    else if (output_path == nullptr)
    {
      output_path = argv[i];
    }
    else
    {
      std::cerr << "codegen: unexpected positional argument '" << argv[i] << "'\n";
      print_usage(argv[0]);
      return 2;
    }
  }

  std::fstream filestream;
  if (output_path != nullptr)
  {
    filestream.open(output_path, filestream.out);
    if (!filestream.is_open())
    {
      std::cerr << "codegen: cannot open '" << output_path << "' for writing\n";
      return 1;
    }
  }

  std::ostream& stream = filestream.is_open() ? filestream : std::cout;

  if (emit_hip)
  {
    // HIP path: single self-contained emitter. The output drops in as
    // include/cuda/std/__atomic/functions/atomic_hip_generated.h. See
    // codegen/generators/hip_intrinsics.h for the scope/payload table.
    FormatHip(stream);
  }
  else
  {
    // NV path: unchanged from upstream.
    FormatHeader(stream);
    FormatFence(stream);
    FormatLoad(stream);
    FormatStore(stream);
    FormatCompareAndSwap(stream);
    FormatExchange(stream);
    FormatFetchOps(stream);
    FormatTail(stream);
  }

  return 0;
}
