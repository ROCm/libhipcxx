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

set(_cccl_cpm_file "${CMAKE_CURRENT_LIST_DIR}/CPM.cmake")

macro(cccl_get_boost)
  include("${_cccl_cpm_file}")
  CPMAddPackage(
    NAME Boost
    GITHUB_REPOSITORY boostorg/boost
    GIT_TAG "boost-1.83.0"
    EXCLUDE_FROM_ALL TRUE
    SYSTEM TRUE
    GIT_SHALLOW TRUE
    # Boost requests compatibility with obsolete CMake versions. Disable warning:
    OPTIONS "CMAKE_POLICY_VERSION_MINIMUM 3.5"
  )
endmacro()

# The CCCL Catch2Helper library:
macro(cccl_get_c2h)
  if (NOT TARGET cccl.c2h)
    add_subdirectory("${CCCL_SOURCE_DIR}/c2h" "${CCCL_BINARY_DIR}/c2h")
  endif()
endmacro()

macro(cccl_get_catch2)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:catchorg/Catch2@3.12.0")
endmacro()

macro(cccl_get_cccl)
  find_package(
    CCCL
    CONFIG
    REQUIRED
    NO_DEFAULT_PATH # Only check the explicit HINTS below:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cccl/"
  )
endmacro()

macro(cccl_get_cub)
  if (LIBCUDACXX_ENABLE_HIP)
    # NOTE(HIP/AMD): use hipCUB (over rocPRIM) as the CUB implementation. Expose it
    # as CUB::CUB and attach the fork's forwarding shims (cmake/hip_bench_compat) so
    # the unmodified upstream benchmark / nvbench_helper sources resolve <cub/...>
    # and <curand.h>. See cmake/hip_bench_compat/README.md.
    find_package(hipcub CONFIG REQUIRED)
    if (NOT TARGET CUB::CUB)
      add_library(CUB::CUB INTERFACE IMPORTED GLOBAL)
      target_link_libraries(CUB::CUB INTERFACE hip::hipcub)
      target_include_directories(CUB::CUB INTERFACE "${CCCL_SOURCE_DIR}/cmake/hip_bench_compat")
    endif()
  else()
    find_package(
      CUB
      CONFIG
      REQUIRED
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cub/"
    )
  endif()
endmacro()

macro(cccl_get_cudatoolkit)
  # NOTE(HIP/AMD): there is no CUDA Toolkit on HIP. The benchmark registry only
  # reads CUDAToolkit_VERSION (for the meta CSV); provide a benign value and skip
  # the REQUIRED find_package(CUDAToolkit), which hard-fails ("Could not find
  # nvcc") on a ROCm system.
  if (LIBCUDACXX_ENABLE_HIP)
    if (NOT DEFINED CUDAToolkit_VERSION)
      set(CUDAToolkit_VERSION "0.0")
    endif()
    # nvbench_helper links CUDA::curand (part of the CUDA toolkit). Map it to
    # hipRAND on HIP so the unmodified nvbench_helper/CMakeLists.txt link line works.
    find_package(hiprand CONFIG REQUIRED)
    if (NOT TARGET CUDA::curand)
      add_library(CUDA::curand INTERFACE IMPORTED GLOBAL)
      target_link_libraries(CUDA::curand INTERFACE hip::hiprand)
    endif()
  else()
    find_package(CUDAToolkit REQUIRED)
  endif()
endmacro()

macro(cccl_get_cudax)
  find_package(
    cudax
    CONFIG
    REQUIRED
    NO_DEFAULT_PATH # Only check the explicit HINTS below:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cudax/"
  )
endmacro()

macro(cccl_get_dlpack)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:dmlc/dlpack#v1.2")
endmacro()

macro(cccl_get_json)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:nlohmann/json@3.12.0")
endmacro()

macro(cccl_get_libcudacxx)
  find_package(
    libcudacxx
    CONFIG
    REQUIRED
    NO_DEFAULT_PATH # Only check the explicit HINTS below:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/libcudacxx/"
  )
endmacro()

set(
  CCCL_NVBENCH_SHA
  "34f1e2a7eeeeec7bb2965a3ec9667d297b479cfe"
  CACHE STRING
  "SHA/tag to use for CCCL's NVBench."
)
mark_as_advanced(CCCL_NVBENCH_SHA)

# NOTE(HIP/AMD): NVBench is CUDA-only; the HIP port is hipBench
# (github.com/ROCm/hipbench). It bundles its OWN libhipcxx via CPM, so
# add_subdirectory'ing it here would create a second libcudacxx::libcudacxx target
# and collide with ours. Instead we consume a PRE-BUILT hipBench tree through
# IMPORTED targets (nvbench::nvbench -> libnvbench.so, nvbench::main -> main.cu).
# Build hipBench once with:
#   cmake -GNinja -S <hipBench> -B <hipBench>/build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES} && ninja -C <hipBench>/build nvbench
set(CCCL_HIPBENCH_ROOT  "" CACHE PATH "hipBench source tree (HIP NVBench port); defaults to <repo>/../hipBench")
set(CCCL_HIPBENCH_BUILD "" CACHE PATH "hipBench build tree (libnvbench.so + generated nvbench/config.cuh); defaults to <root>/build")

macro(cccl_get_nvbench)
  if (LIBCUDACXX_ENABLE_HIP)
    # Resolve defaults here (not at file scope) so CCCL_SOURCE_DIR is populated.
    if (NOT CCCL_HIPBENCH_ROOT)
      get_filename_component(CCCL_HIPBENCH_ROOT "${CCCL_SOURCE_DIR}/../hipBench" ABSOLUTE)
    endif()
    if (NOT CCCL_HIPBENCH_BUILD)
      set(CCCL_HIPBENCH_BUILD "${CCCL_HIPBENCH_ROOT}/build")
    endif()
    if (NOT EXISTS "${CCCL_HIPBENCH_BUILD}/lib/libnvbench.so")
      message(FATAL_ERROR
        "hipBench not built at '${CCCL_HIPBENCH_BUILD}/lib/libnvbench.so'. "
        "Clone ROCm/hipbench to '${CCCL_HIPBENCH_ROOT}' and build the 'nvbench' "
        "target, or set -DCCCL_HIPBENCH_ROOT / -DCCCL_HIPBENCH_BUILD.")
    endif()
    if (NOT TARGET nvbench::nvbench)
      add_library(nvbench::nvbench SHARED IMPORTED GLOBAL)
      set_target_properties(nvbench::nvbench PROPERTIES
        IMPORTED_LOCATION "${CCCL_HIPBENCH_BUILD}/lib/libnvbench.so"
        INTERFACE_INCLUDE_DIRECTORIES "${CCCL_HIPBENCH_ROOT};${CCCL_HIPBENCH_BUILD}")
    endif()
    if (NOT TARGET nvbench::main)
      # nvbench::main is the benchmark entry point (nvbench/main.cu) the benches link.
      add_library(cccl.nvbench.main STATIC "${CCCL_HIPBENCH_ROOT}/nvbench/main.cu")
      set_source_files_properties("${CCCL_HIPBENCH_ROOT}/nvbench/main.cu" PROPERTIES LANGUAGE HIP)
      target_link_libraries(cccl.nvbench.main PUBLIC nvbench::nvbench)
      add_library(nvbench::main ALIAS cccl.nvbench.main)
    endif()
  else()
    include("${_cccl_cpm_file}")
    CPMAddPackage("gh:NVIDIA/nvbench#${CCCL_NVBENCH_SHA}")
  endif()
endmacro()

# CCCL-specific NVBench utilities
macro(cccl_get_nvbench_helper)
  if (NOT TARGET cccl.nvbench_helper)
    add_subdirectory(
      "${CCCL_SOURCE_DIR}/nvbench_helper"
      "${CCCL_BINARY_DIR}/nvbench_helper"
    )
  endif()
endmacro()

macro(cccl_get_nvtx)
  include("${_cccl_cpm_file}")
  CPMAddPackage(
    NAME NVTX
    GITHUB_REPOSITORY NVIDIA/NVTX
    GIT_TAG release-v3
    DOWNLOAD_ONLY ON
    SYSTEM ON
  )
  include("${NVTX_SOURCE_DIR}/c/nvtxImportedTargets.cmake")
endmacro()

macro(cccl_get_thrust)
  if (LIBCUDACXX_ENABLE_HIP)
    # NOTE(HIP/AMD): rocThrust is the HIP Thrust implementation. Expose it as
    # Thrust::Thrust so the unmodified upstream link lines work.
    find_package(rocthrust CONFIG REQUIRED)
    if (NOT TARGET Thrust::Thrust)
      add_library(Thrust::Thrust INTERFACE IMPORTED GLOBAL)
      target_link_libraries(Thrust::Thrust INTERFACE roc::rocthrust)
    endif()
  else()
    find_package(
      Thrust
      CONFIG
      REQUIRED
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS "${CCCL_SOURCE_DIR}/lib/cmake/thrust/"
    )
  endif()
endmacro()
