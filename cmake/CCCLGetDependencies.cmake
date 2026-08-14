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
  # NOTE(HIP/AMD): the c2h test executables are HIP targets, so amdclang++ links
  # them with '-pie' (CLANG_DEFAULT_PIE_ON_LINUX=ON), while Catch2 is a plain
  # CXX project built by whatever the host compiler is. When that compiler does
  # not itself default to PIE -- a gcc-toolset g++, say, which is built without
  # --enable-default-pie -- libCatch2.a ships objects carrying absolute
  # relocations and ld.lld refuses them in the PIE image:
  #   ld.lld: error: relocation R_X86_64_32S cannot be used against symbol
  #   'vtable for Catch::CompactReporter'; recompile with -fPIC
  # ci/internal/build.sh now uses amdclang++ on every builder, whose -fPIE
  # default is enough for an executable, so nothing in CI hits this. It is kept
  # as a guard for LIBHIPCXX_HOST_COMPILER pointing somewhere else, since the
  # failure surfaces as a link error inside a dependency rather than anything
  # naming the host compiler. Ubuntu's g++ defaults to PIE and hides it too.
  #
  # The arch variables are no help here: CMAKE_HIP_ARCHITECTURES / GPU_TARGETS /
  # AMDGPU_TARGETS reach HIP-language compiles only, never a CXX one. Setting
  # the property on the two archives beats CMAKE_POSITION_INDEPENDENT_CODE for
  # the whole tree; they are the only CXX static libraries a HIP test
  # executable links. hipBench needs no equivalent -- it sets
  # CMAKE_POSITION_INDEPENDENT_CODE itself and forwards it to its fmt fetch.
  if (LIBCUDACXX_ENABLE_HIP)
    foreach (_cccl_c2_tgt IN ITEMS Catch2 Catch2WithMain)
      if (TARGET ${_cccl_c2_tgt})
        set_target_properties(${_cccl_c2_tgt} PROPERTIES POSITION_INDEPENDENT_CODE ON)
      endif()
    endforeach()
  endif()
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
  # NOTE(HIP/AMD): there is no CUDA Toolkit on HIP, so skip the REQUIRED
  # find_package(CUDAToolkit), which hard-fails ("Could not find nvcc") on a
  # ROCm system. CUDAToolkit_VERSION and CUDAToolkit_FOUND are deliberately left
  # undefined -- there is no CUDA toolkit to report a version for, and every
  # consumer must key off CUDAToolkit_FOUND rather than a stand-in version.
  if (LIBCUDACXX_ENABLE_HIP)
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
  "373970323f3e2a3995761ea682ca64dfcbdd1e26"
  CACHE STRING
  "SHA/tag to use for CCCL's NVBench."
)
mark_as_advanced(CCCL_NVBENCH_SHA)

# NOTE(HIP/AMD): NVBench is CUDA-only; the HIP port is hipBench
# (github.com/ROCm/hipbench). Its exec-tag API (nvbench::exec_tag::gpu / no_batch)
# lives on the 'amd-develop' branch ('main' / 'release/rocmds-25.10' are too old).
# By default cccl_get_nvbench() auto-downloads and builds hipBench via CPM, reusing
# our in-tree libcudacxx so hipBench's rapids_cpm_libhipcxx does not fetch a second
# copy (which would collide on libcudacxx::libcudacxx). Set CCCL_HIPBENCH_ROOT /
# CCCL_HIPBENCH_BUILD to consume a PRE-BUILT tree instead (offline / faster CI).
set(
  CCCL_HIPBENCH_SHA
  "631f22085a07df66652fb765674abaeac19ac145"
  CACHE STRING
  "SHA/tag to use for hipBench (HIP NVBench port)."
)
mark_as_advanced(CCCL_HIPBENCH_SHA)
set(CCCL_HIPBENCH_ROOT  "" CACHE PATH "Pre-built hipBench source tree (set to skip CPM auto-download)")
set(CCCL_HIPBENCH_BUILD "" CACHE PATH "Pre-built hipBench build tree (libnvbench.so + generated nvbench/config.cuh)")

macro(cccl_get_nvbench)
  if (LIBCUDACXX_ENABLE_HIP)
    if (CCCL_HIPBENCH_ROOT OR CCCL_HIPBENCH_BUILD)
      # ---- Pre-built hipBench (explicit override): consume via IMPORTED targets ----
      if (NOT CCCL_HIPBENCH_ROOT)
        get_filename_component(CCCL_HIPBENCH_ROOT "${CCCL_SOURCE_DIR}/../hipBench" ABSOLUTE)
      endif()
      if (NOT CCCL_HIPBENCH_BUILD)
        set(CCCL_HIPBENCH_BUILD "${CCCL_HIPBENCH_ROOT}/build")
      endif()
      if (NOT EXISTS "${CCCL_HIPBENCH_BUILD}/lib/libnvbench.so")
        message(FATAL_ERROR
          "hipBench not built at '${CCCL_HIPBENCH_BUILD}/lib/libnvbench.so'. "
          "Build it, or unset CCCL_HIPBENCH_ROOT/CCCL_HIPBENCH_BUILD to auto-download.")
      endif()
      if (NOT TARGET nvbench::nvbench)
        add_library(nvbench::nvbench SHARED IMPORTED GLOBAL)
        set_target_properties(nvbench::nvbench PROPERTIES
          IMPORTED_LOCATION "${CCCL_HIPBENCH_BUILD}/lib/libnvbench.so"
          INTERFACE_INCLUDE_DIRECTORIES "${CCCL_HIPBENCH_ROOT};${CCCL_HIPBENCH_BUILD}")
      endif()
      if (NOT TARGET nvbench::main)
        set(_cccl_nvbench_main "${CCCL_HIPBENCH_ROOT}/nvbench/main.cu")
        if (NOT EXISTS "${_cccl_nvbench_main}")
          set(_cccl_nvbench_main "${CCCL_HIPBENCH_ROOT}/nvbench/main.hip")
        endif()
        add_library(cccl.nvbench.main STATIC "${_cccl_nvbench_main}")
        set_source_files_properties("${_cccl_nvbench_main}" PROPERTIES LANGUAGE HIP)
        target_link_libraries(cccl.nvbench.main PUBLIC nvbench::nvbench)
        add_library(nvbench::main ALIAS cccl.nvbench.main)
      endif()
    else()
      # ---- Auto-download hipBench via CPM ----
      # hipBench pulls its own libhipcxx through rapids_cpm_libhipcxx; left alone
      # that creates a SECOND libcudacxx::libcudacxx target that collides with ours.
      # rapids_cpm_find skips its fetch when the GLOBAL_TARGET (libhipcxx::libhipcxx)
      # already exists (rapids-cmake cpm/find.cmake), so expose our in-tree
      # libcudacxx under that name first -- the same dedup upstream relies on for
      # nvbench reusing CCCL's libcudacxx.
      if (NOT TARGET libhipcxx::libhipcxx AND TARGET libcudacxx::libcudacxx)
        add_library(libhipcxx::libhipcxx INTERFACE IMPORTED GLOBAL)
        set_target_properties(libhipcxx::libhipcxx PROPERTIES
          INTERFACE_LINK_LIBRARIES libcudacxx::libcudacxx)
      endif()
      include("${_cccl_cpm_file}")
      CPMAddPackage(
        NAME hipBench
        GITHUB_REPOSITORY ROCm/hipbench
        GIT_TAG ${CCCL_HIPBENCH_SHA}
        EXCLUDE_FROM_ALL TRUE
        OPTIONS "CMAKE_HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES}"
      )
      # hipBench provides nvbench::nvbench and nvbench::main directly. It maps .cu
      # to the HIP language via its overrides.cmake, which only fires when hipBench
      # is the top-level project -- under add_subdirectory it does not, so CMake
      # cannot determine a language for its .cu sources ("cannot determine linker
      # language for target: nvbench.main / nvbench.ctl"). Tag those sources as HIP
      # here instead (TARGET_DIRECTORY needs CMake >= 3.18).
      foreach (_cccl_hb_tgt IN ITEMS nvbench nvbench.main nvbench.ctl)
        if (TARGET ${_cccl_hb_tgt})
          get_target_property(_cccl_hb_srcs ${_cccl_hb_tgt} SOURCES)
          get_target_property(_cccl_hb_sdir ${_cccl_hb_tgt} SOURCE_DIR)
          foreach (_cccl_hb_src IN LISTS _cccl_hb_srcs)
            if (_cccl_hb_src MATCHES "\\.cu$")
              if (NOT IS_ABSOLUTE "${_cccl_hb_src}")
                set(_cccl_hb_src "${_cccl_hb_sdir}/${_cccl_hb_src}")
              endif()
              set_source_files_properties("${_cccl_hb_src}"
                TARGET_DIRECTORY ${_cccl_hb_tgt}
                PROPERTIES LANGUAGE HIP)
            endif()
          endforeach()
        endif()
      endforeach()
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
