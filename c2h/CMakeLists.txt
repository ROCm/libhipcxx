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

# NOTE(HIP/AMD): under HIP we don't have rocThrust / hipCUB / hipRAND
# wired up in this tree, so the generators*.cu library sources -- which
# all pull <thrust/...> and (some) <cub/...> -- are skipped on HIP.
# What's built on HIP is a stripped-down Catch2-only cccl.c2h library
# that exposes the c2h::detail::* helpers from <c2h/checked_allocator.cuh>
# and the catch2_runner.cpp / catch2_runner_helper.cu entry points.
# The thrust-using c2h headers (c2h/{generators,vector,test_util_vec,
# device_policy,bfloat16,half,operator,check_results,extended_types,
# fill_striped,custom_type,detail/generators}.{cuh,h}) and the templates
# at the bottom of checked_allocator.cuh are gated under
# !_CCCL_HIP_COMPILATION() in their respective headers.
if (LIBCUDACXX_ENABLE_HIP)
  project(C2H LANGUAGES CXX HIP)
else()
  project(C2H LANGUAGES CXX CUDA)
endif()

cccl_get_catch2()

# NOTE(HIP/AMD): on HIP the parent libhipcxx CMakeLists pre-defines
# CCCL::CCCL as an alias for libcudacxx::libcudacxx (the only CCCL
# component we ship under HIP). Skip the find_package(CCCL) call here
# -- it would try to find lib/cmake/cccl/cccl-config.cmake which only
# exists in the full CCCL distribution and transitively pulls CUB +
# Thrust which we don't have under HIP.
if (NOT LIBCUDACXX_ENABLE_HIP)
  find_package(
    CCCL
    CONFIG
    REQUIRED
    NO_DEFAULT_PATH # Only check the explicit HINTS below:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cccl/"
  )
endif()

if (LIBCUDACXX_ENABLE_HIP)
  find_package(hip REQUIRED)
else()
  find_package(CUDAToolkit)
endif()

set(curand_default OFF)
if (CUDA_curand_LIBRARY AND NOT LIBCUDACXX_ENABLE_HIP)
  set(curand_default ON)
endif()

option(C2H_ENABLE_CURAND "Use CUDA CURAND library in c2h." ${curand_default})
if (LIBCUDACXX_ENABLE_HIP AND C2H_ENABLE_CURAND)
  message(FATAL_ERROR "C2H_ENABLE_CURAND is not supported under HIP (no hipRAND wiring in this tree).")
endif()

# Library sources: under HIP the thrust-heavy generators*.cu translation
# units are skipped; the Catch2 framework + helpers are still present
# via cccl.c2h.main (catch2_runner.cpp + catch2_runner_helper.cu).
if (LIBCUDACXX_ENABLE_HIP)
  add_library(cccl.c2h INTERFACE)
else()
  add_library(
    cccl.c2h
    STATIC
    generators.cu
    generators_gen_values.cu
    generators_uniform_offsets.cu
    generators_vector.cu
  )
endif()
if (LIBCUDACXX_ENABLE_HIP)
  target_include_directories(cccl.c2h INTERFACE "${C2H_SOURCE_DIR}/include")
  target_link_libraries(
    cccl.c2h
    INTERFACE #
      CCCL::CCCL
      Catch2::Catch2
      cccl.compiler_interface_cpp17
      hip::host
  )
else()
  target_include_directories(cccl.c2h PUBLIC "${C2H_SOURCE_DIR}/include")
  target_link_libraries(
    cccl.c2h
    PUBLIC #
      CCCL::CCCL
      Catch2::Catch2
      cccl.compiler_interface_cpp17
  )
  cccl_configure_target(cccl.c2h DIALECT 17)
endif()

if (C2H_ENABLE_CURAND)
  target_link_libraries(cccl.c2h PRIVATE CUDA::curand)
  target_compile_definitions(cccl.c2h PRIVATE C2H_HAS_CURAND=1)
elseif (NOT LIBCUDACXX_ENABLE_HIP)
  target_compile_definitions(cccl.c2h PRIVATE C2H_HAS_CURAND=0)
endif()

# The Catch2 runner. Under HIP we tag the .cu file with LANGUAGE HIP so
# it compiles through hipcc / clang -x hip. catch2_runner_helper.{inl,cu}
# only uses cuda* runtime API symbols that are shimmed by
# <amd/cuda_runtime.h>, so no source changes needed.
add_library(cccl.c2h.main OBJECT catch2_runner.cpp catch2_runner_helper.cu)
if (LIBCUDACXX_ENABLE_HIP)
  # NOTE(HIP/AMD): catch2_runner.cpp transitively includes
  # <amd/cuda_runtime.h> which pulls <hip/hip_runtime.h>. The latter
  # uses _Float16 / __bf16 which are HIP-clang specific extensions, so
  # we have to compile the runner through clang -x hip even though
  # the file extension is .cpp.
  set_source_files_properties(
    catch2_runner.cpp catch2_runner_helper.cu
    PROPERTIES LANGUAGE HIP
  )
  target_link_libraries(cccl.c2h.main PUBLIC cccl.c2h hip::host)
else()
  target_link_libraries(cccl.c2h.main PUBLIC cccl.c2h)
endif()
