# MIT License
#
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.21)

# NOTE(HIP/AMD): build the nvbench_helper TU as HIP on a ROCm build.
if (LIBCUDACXX_ENABLE_HIP)
  project(nvbench_helper LANGUAGES CXX HIP)
else()
  project(nvbench_helper LANGUAGES CXX CUDA)
endif()

include(CMakeDependentOption)

cccl_get_catch2()
cccl_get_cub()
cccl_get_cudatoolkit()
cccl_get_libcudacxx()
cccl_get_nvbench()
cccl_get_thrust()

add_library(cccl.nvbench_helper OBJECT nvbench_helper/nvbench_helper.cu)
if (LIBCUDACXX_ENABLE_HIP)
  set_source_files_properties(nvbench_helper/nvbench_helper.cu PROPERTIES LANGUAGE HIP)
  # The hip* runtime calls are [[nodiscard]]; the upstream CUDA paths ignore the
  # returns (cudaError_t is not nodiscard on CUDA). Keep the warning but don't fail
  # -Werror on it, rather than editing the upstream source.
  target_compile_options(cccl.nvbench_helper PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-error=unused-value -Wno-error=unused-result>)
endif()
cccl_configure_target(cccl.nvbench_helper)
target_link_libraries(
  cccl.nvbench_helper
  PUBLIC
    libcudacxx::libcudacxx
    CUB::CUB
    # This Thrust target does not define any host/device system. Those are added later if needed:
    Thrust::Thrust
    nvbench::nvbench
  PRIVATE #
    cccl.compiler_interface
    CUDA::curand
)

target_include_directories(
  cccl.nvbench_helper
  PUBLIC "${CMAKE_CURRENT_LIST_DIR}/nvbench_helper"
)

# Old NVCC incorrectly infers the host device annotation of std::map::map() which calls a non-host device member
# NOTE(HIP/AMD): `-diag-suppress` is nvcc-only. Without a CUDA toolkit
# CUDAToolkit_VERSION is undefined, which VERSION_LESS reads as "0" -- so the
# bare upstream condition would force the flag on and break the HIP-clang
# benchmark compile (clang treats the diagnostic number `20011` as a positional
# filename: "no such file or directory: '20011'"), and it is PUBLIC, so it
# propagates to every bench linking nvbench_helper. Require a real toolkit.
if (CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_LESS "12.5.0")
  target_compile_options(cccl.nvbench_helper PUBLIC -diag-suppress 20011)
endif()

# Don't enable tests when pulled in as an internal dependency.
# Require CCCL_ENABLE_NVBENCH_HELPER to be explicitly enabled.
cmake_dependent_option(
  nvbench_helper_ENABLE_TESTING
  "Enable tests for nvbench_helper"
  ON
  CCCL_ENABLE_NVBENCH_HELPER
  OFF
)
mark_as_advanced(nvbench_helper_ENABLE_TESTING)

if (nvbench_helper_ENABLE_TESTING)
  cccl_get_boost()

  thrust_create_target(cccl.nvbench_helper.test.Thrust.cpp.cuda DEVICE CUDA)
  thrust_create_target(cccl.nvbench_helper.test.Thrust.cpp.cpp DEVICE CPP)

  function(add_nvbench_helper_test device_system)
    set(nvbench_helper_test_target nvbench_helper.test.${device_system})
    cccl_add_executable(
      ${nvbench_helper_test_target}
      ADD_CTEST
      NO_METATARGETS
      SOURCES
        test/gen_seed.cu
        test/gen_range.cu
        test/gen_entropy.cu
        test/gen_uniform_distribution.cu
        test/gen_power_law_distribution.cu
    )
    target_link_libraries(
      ${nvbench_helper_test_target}
      PRIVATE
        cccl.compiler_interface
        cccl.nvbench_helper
        cccl.nvbench_helper.test.Thrust.cpp.${device_system}
        Catch2::Catch2WithMain
        Boost::math
    )
  endfunction()

  add_nvbench_helper_test(cpp)
  add_nvbench_helper_test(cuda)
endif()
