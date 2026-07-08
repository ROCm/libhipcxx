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

# when CCCL is used from the CTK, C2H does not need to be rebuilt when making local changes to CCCL (faster iteration)
option(C2H_USE_CCCL_FROM_CTK "Use CCCL from the CTK in c2h." OFF)

# Get dependencies.
cccl_get_catch2()
if (LIBCUDACXX_ENABLE_HIP)
  # NOTE(HIP/AMD): on HIP we don't ship a full CCCL distribution
  # (no CUB / Thrust packages), so find_package(CCCL CONFIG REQUIRED)
  # would fail looking for lib/cmake/cccl/cccl-config.cmake. The only
  # CCCL component we ship is libcudacxx; the parent CMakeLists
  # (CMakeLists.txt:245-249) pre-defines CCCL::CCCL as an alias for
  # libcudacxx::libcudacxx so that subsequent
  # target_link_libraries(... CCCL::CCCL ...) calls in this file resolve
  # correctly. Verify the alias actually exists rather than silently
  # reaching the link line later with a missing target. Once a HIP-aware
  # cccl-config.cmake exists upstream, this whole branch can collapse to
  # the unconditional find_package() below.
  if (NOT TARGET CCCL::CCCL)
    message(FATAL_ERROR
      "CCCL::CCCL alias must be pre-defined by the parent project on HIP "
      "(see top-level CMakeLists.txt:245-249). Cannot continue without it.")
  endif()
  find_package(hip REQUIRED)
  # NOTE(HIP/AMD): rocThrust 4.x ships under /opt/rocm/lib/cmake/rocthrust
  # and provides the 'roc::rocthrust' interface target, which exposes
  # the 'thrust::' namespace via Thrust 2.8 with the HIP backend
  # selected. Make it OPTIONAL so a HIP install without rocThrust still
  # builds the Catch2-only c2h library; consumers (per-test executables)
  # that need thrust check 'TARGET roc::rocthrust' before linking and
  # otherwise stay out of the build via the LIBCUDACXX_C2H_HIP_SKIP_LIST.
  find_package(rocthrust QUIET CONFIG)
  if (rocthrust_FOUND)
    message(STATUS "Found rocThrust: ${rocthrust_VERSION}")
  else()
    message(STATUS "rocThrust not found; thrust-using c2h tests will be skip-listed.")
  endif()
  # NOTE(HIP/AMD): force '-pthread' onto every c2h test executable.
  # cuda::__physical_device (in include/cuda/__device/physical_device.h)
  # uses 'std::once_flag' on the very first device touch
  # (__primary_context, __name, __peers) -- that is, every C2H_CCCLRT_TEST
  # that constructs 'cuda::device_ref{0}', 'cuda::stream{...}' or
  # '__ensure_current_context'. On glibc < 2.34 (manylinux_2_28 ships
  # glibc 2.28) libpthread is still a separate '.so'; libstdc++'s
  # 'std::call_once' resolves '__gthread_once' to a weak no-op stub
  # when the binary was not linked '-pthread', the stub returns -1, and
  # 'call_once' rethrows that as 'std::system_error(-1, generic_category)'
  # whose 'what()' is exactly 'Unknown error -1' (== glibc strerror(-1)).
  # The callable is *never invoked*, so no HIP API name appears in the
  # message, which matched the CCCLRT failure pattern (same source built
  # & passing on Ubuntu 22.04 / glibc 2.35 -- where libpthread is folded
  # into libc -- and failing only inside the manylinux_2_28 container on
  # the same physical host). Propagating Threads::Threads through
  # cccl.c2h (INTERFACE) and cccl.c2h.main (PUBLIC) makes every per-test
  # executable inherit '-pthread' transitively. NV-only builds are
  # unaffected; this is HIP-side only by intention.
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
else()
  if (NOT C2H_USE_CCCL_FROM_CTK)
    cccl_get_cccl()
  endif()
  cccl_get_cudatoolkit()
endif()

set(has_curand OFF)
if (TARGET CUDA::curand AND NOT LIBCUDACXX_ENABLE_HIP)
  set(has_curand ON)
endif()

option(C2H_ENABLE_CURAND "Use CUDA CURAND library in c2h." ${has_curand})

# Disable C2H_ENABLE_CURAND if cuRAND is unavailable.
if (C2H_ENABLE_CURAND AND NOT has_curand)
  # NOTE(HIP/AMD): on HIP the generators*.cu sources that actually call
  # curand* (via C2H_HAS_CURAND=1) aren't compiled -- cccl.c2h is an
  # INTERFACE library on HIP (Catch2 framework + helpers only; see the
  # add_library branch below). So opting in to C2H_ENABLE_CURAND under
  # HIP is a no-op rather than an error. The ROCm equivalent (hipRAND)
  # is shipped and supported in /opt/rocm/lib, we just don't wire it
  # into the Catch2-only c2h slice today. Status-only message so a
  # cross-backend CI matrix doesn't have to special-case the flag.
  if (LIBCUDACXX_ENABLE_HIP)
    message(STATUS
      "C2H_ENABLE_CURAND is a no-op under HIP: c2h is built as Catch2-only "
      "INTERFACE here and generators*.cu (the curand consumer) is skipped.")
  else()
    message(
      WARNING
      "C2H_ENABLE_CURAND is requested, but CUDA::curand is unavailable. Disabling."
    )
  endif()
  set(C2H_ENABLE_CURAND OFF)
endif()

# Library sources: under HIP the thrust-heavy generators*.cu translation
# units are skipped; the Catch2 framework + helpers are still present
# via cccl.c2h.main (catch2_runner.cu + catch2_runner_helper.cu).
if (LIBCUDACXX_ENABLE_HIP)
  add_library(cccl.c2h INTERFACE)
  include(${CMAKE_SOURCE_DIR}/cmake/CCCLC2hHipDeps.cmake)
  target_include_directories(cccl.c2h INTERFACE "${C2H_SOURCE_DIR}/include")
  target_link_libraries(
    cccl.c2h
    INTERFACE #
      CCCL::CCCL
      Catch2::Catch2
      cccl.compiler_interface
  )
  # Threads::Threads + hip::host attached via the shared helper -- see
  # cmake/CCCLC2hHipDeps.cmake for the manylinux/glibc < 2.34 pthread
  # weak-stub rationale that requires '-pthread' on every HIP test
  # executable that constructs a 'cuda::device_ref' / 'cuda::stream'.
  cccl_c2h_attach_hip_deps(cccl.c2h SCOPE INTERFACE)
else()
  add_library(
    cccl.c2h
    STATIC
    generators.cu
    generators_gen_values.cu
    generators_uniform_offsets.cu
    generators_vector.cu
  )
  target_include_directories(cccl.c2h PUBLIC "${C2H_SOURCE_DIR}/include")
  target_link_libraries(
    cccl.c2h
    PUBLIC #
      cccl.compiler_interface
      Catch2::Catch2
  )
  if (NOT C2H_USE_CCCL_FROM_CTK)
    target_link_libraries(cccl.c2h PUBLIC CCCL::CCCL)
  endif()
  cccl_configure_target(cccl.c2h DIALECT 17)
endif()

if (C2H_ENABLE_CURAND)
  target_link_libraries(cccl.c2h PRIVATE CUDA::curand)
  target_compile_definitions(cccl.c2h PRIVATE C2H_HAS_CURAND=1)
elseif (NOT LIBCUDACXX_ENABLE_HIP)
  target_compile_definitions(cccl.c2h PRIVATE C2H_HAS_CURAND=0)
endif()

set_target_properties(
  cccl.c2h
  PROPERTIES CXX_VISIBILITY_PRESET "default" CUDA_VISIBILITY_PRESET "default"
)

# The Catch2 runner. Under HIP we tag the .cu file with LANGUAGE HIP so
# it compiles through hipcc / clang -x hip. catch2_runner_helper.{inl,cu}
# only uses cuda* runtime API symbols that are shimmed by
# <libhipcxx/__amd/cuda_runtime.h>, so no source changes needed.
add_library(cccl.c2h.main OBJECT catch2_runner.cu catch2_runner_helper.cu)
if (LIBCUDACXX_ENABLE_HIP)
  # NOTE(HIP/AMD): catch2_runner.cu transitively includes
  # <libhipcxx/__amd/cuda_runtime.h> which pulls <hip/hip_runtime.h>. The latter
  # uses _Float16 / __bf16 which are HIP-clang specific extensions, so
  # we have to compile the runner through clang -x hip even though
  # the file extension is .cu.
  set_source_files_properties(
    catch2_runner.cu catch2_runner_helper.cu
    PROPERTIES LANGUAGE HIP
  )
  target_link_libraries(cccl.c2h.main PUBLIC cccl.c2h)
  cccl_c2h_attach_hip_deps(cccl.c2h.main SCOPE PUBLIC)
else()
  target_link_libraries(cccl.c2h.main PUBLIC cccl.c2h)
endif()
