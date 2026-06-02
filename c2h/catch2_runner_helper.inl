// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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

//! @file
//! This file includes implementation of CUDA-specific utilities for custom Catch2 main. When CMake is configured to
//! include all the tests into a single executable, this file is only included into catch2_runner_helper.cu. When CMake
//! is configured to compile each test as a separate binary, this file is included into each test.

// NOTE(HIP/AMD): restore the runtime-API wrapper include dropped upstream so
// cudaGetDeviceCount / cudaSetDevice / cudaSuccess resolve (shimmed to the
// hip* equivalents in <libhipcxx/__amd/cuda_runtime.h>) for the shared
// cccl.c2h.main runner on HIP.
#include <cuda/__runtime/api_wrapper.h>

#include <iostream>

int device_guard(int device_id)
{
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess)
  {
    std::cerr << "Failed getting number of devices" << std::endl;
    std::exit(-1);
  }

  if (device_id >= device_count || device_id < 0)
  {
    std::cerr << "Invalid device ID: " << device_id << std::endl;
    std::exit(-1);
  }

  return device_id;
}

void set_device(int device_id)
{
  if (cudaSetDevice(device_guard(device_id)) != cudaSuccess)
  {
    std::cerr << "Failed to set device ID: " << device_id << std::endl;
    std::exit(-1);
  }
}
