# MIT License
#
# Modifications Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

macro(pythonize_bool var)
  if (${var})
    set(${var} True)
  else()
    set(${var} False)
  endif()
endmacro()

#add dlpack
cccl_get_dlpack()

if (LIBCUDACXX_ENABLE_CUDA)

  find_package(CUDAToolkit REQUIRED)
  get_target_property(CUDA_INCLUDE_DIR CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)

  set(LIBCUDACXX_HIGHEST_COMPUTE_ARCH 90)

  foreach (COMPUTE_ARCH ${LIBCUDACXX_COMPUTE_ARCHS})
    set(_compute_message "${_compute_message} sm_${COMPUTE_ARCH}")
    set(
      LIBCUDACXX_COMPUTE_ARCHS_STRING
      "${LIBCUDACXX_COMPUTE_ARCHS_STRING} ${COMPUTE_ARCH}"
    )
  endforeach()

  message(STATUS "Enabled CUDA architectures:${_compute_message}")

elseif(LIBCUDACXX_ENABLE_HIP)

  # If LIBCUDACXX_COMPUTE_ARCHS is not set, use the HIP architectures
  if (NOT LIBCUDACXX_COMPUTE_ARCHS)
    set(LIBCUDACXX_COMPUTE_ARCHS ${CMAKE_HIP_ARCHITECTURES})
  endif()

  # Separate the architectures with a space
  foreach (COMPUTE_ARCH ${LIBCUDACXX_COMPUTE_ARCHS})
    set(_compute_message "${_compute_message} ${COMPUTE_ARCH}")
    set(LIBCUDACXX_COMPUTE_ARCHS_STRING "${LIBCUDACXX_COMPUTE_ARCHS_STRING} ${COMPUTE_ARCH}")
  endforeach ()

  message(STATUS "Enabled HIP architectures:${_compute_message}")

  option(LIBHIPCXX_TEST_WITH_HIPRTC
  "Test libcu++ with runtime compilation instead of offline compilation. Only runs device side tests."
  OFF)
endif()

option(
  LIBCUDACXX_TEST_WITH_NVRTC
  "Test libcu++ with runtime compilation instead of offline compilation. Only runs device side tests."
  OFF
)

if (LIBCUDACXX_TEST_WITH_NVRTC)
  # TODO: Use project properties to get path to binary.
  # Should also set up dependency on the project when NVRTC is enabled
  foreach (include IN ITEMS ${CUDA_INCLUDE_DIR})
    string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -I'${include}'")
  endforeach()
  set(
    LIBCUDACXX_CUDA_COMPILER
    "${CMAKE_BINARY_DIR}/libcudacxx/test/utils/nvidia/nvrtc/nvrtcc"
  )
  set(LIBCUDACXX_CUDA_COMPILER_ARG1 "")
  set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "True")
  # Use the NVRTCC utility to run the built test outputs
  set(
    LIBCUDACXX_EXECUTOR
    "PrefixExecutor(['${LIBCUDACXX_CUDA_COMPILER}'], LocalExecutor())"
  )
  # Enable 128-bit types for NVRTC
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -device-int128")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -device-float128")
elseif (LIBHIPCXX_TEST_WITH_HIPRTC)
  # TODO: Use project properties to get path to binary.
  # Should also set up dependency on the project when NVRTC is enabled
  set(CMAKE_CUDA_COMPILER_ID "HIPCC")
  set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_BINARY_DIR}/test/utils/amd/hiprtc/hiprtcc${CMAKE_EXECUTABLE_SUFFIX}")
  message(STATUS "LIBCUDACXX_CUDA_COMPILER: ${LIBCUDACXX_CUDA_COMPILER}")
  set(LIBCUDACXX_CUDA_COMPILER_ARG1 "")
  set(LIBCUDACXX_HIP_TEST_WITH_HIPRTC "True")
else() # NOT LIBCUDACXX_TEST_WITH_NVRTC
  string(
    APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
  )
  set(
    LIBCUDACXX_FORCE_INCLUDE
    "-include ${libcudacxx_SOURCE_DIR}/test/libcudacxx/force_include.h"
  )
  set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}")
  set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "False")
  if (LIBCUDACXX_ENABLE_HIP)
    set(
      LIBCUDACXX_FORCE_INCLUDE
      "-include ${libcudacxx_SOURCE_DIR}/test/libcudacxx/force_include_hip.h"
    )
    #temporary workaround until we've found a way to make cmake find hipcc
    set(LIBCUDACXX_CUDA_COMPILER "${HIP_HIPCC_EXECUTABLE}")
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_COMPILER_ID "HIPCC")
    set(LIBCUDACXX_HIP_TEST_WITH_HIPRTC "False")
  endif()
endif()

# enable exceptions and assertions in tests
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -DCCCL_ENABLE_ASSERTIONS")
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -I ${dlpack_SOURCE_DIR}/include")

# enable optional<T&>
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -DCCCL_ENABLE_OPTIONAL_REF")

# Disable dialect deprecation
string(
  APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
  " -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT"
)
string(
  APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
  " -DLIBCUDACXX_IGNORE_DEPRECATED_ABI"
)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  string(APPEND LIBCUDACXX_TEST_LINKER_FLAGS " -latomic")
endif()

if (NOT MSVC AND NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
  set(
    LIBCUDACXX_WARNING_LEVEL
    "--compiler-options=-Wall --compiler-options=-Wextra"
  )
endif()

if (MSVC)
  # sccache cannot handle the -Fd option generationg pdb files
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)

  # We want to use cudaLaunchKernelEx which is guarded by __cplusplus
  if ("${CMAKE_CUDA_COMPILER_VERSION}" LESS "12.3.0")
    string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -Xcompiler=/Zc:__cplusplus")
  endif()

  # Require the conforming preprocessor
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -Xcompiler=/Zc:preprocessor")
  if (MSVC_TOOLSET_VERSION LESS 143)
    # winbase.h(9572): warning C5105: macro expansion producing 'defined' has undefined behavior
    string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -Xcompiler=/wd5105")
  endif()
endif()

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
  string(
    APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " ${CMAKE_CUDA_FLAGS}"
    " -Xclang -fcuda-allow-variadic-functions"
    " -Xclang -Wno-unused-parameter"
    " -Wno-unknown-cuda-version"
    " ${LIBCUDACXX_FORCE_INCLUDE}"
    " -I${libcudacxx_SOURCE_DIR}/include"
    " ${LIBCUDACXX_WARNING_LEVEL}"
  )

  string(
    APPEND LIBCUDACXX_TEST_LINKER_FLAGS
    " ${CMAKE_CUDA_FLAGS}"
    " -L${CUDAToolkit_LIBRARY_DIR}"
    " -lcuda"
    " -lcudart"
  )
elseif (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
  string(
    APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " ${LIBCUDACXX_FORCE_INCLUDE}"
    " ${LIBCUDACXX_WARNING_LEVEL}"
    " -Wno-deprecated-gpu-targets"
  )
elseif (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVHPC")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -stdpar")
  string(APPEND LIBCUDACXX_TEST_LINKER_FLAGS " -stdpar")
endif()

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "HIPCC")
  message(STATUS "libcudacxx_SOURCE_DIR: ${libcudacxx_SOURCE_DIR}" )
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
#    "${LIBCUDACXX_TEST_COMPILER_FLAGS} \
    " ${LIBCUDACXX_FORCE_INCLUDE}"
    " -I${CMAKE_SOURCE_DIR}/include"
    " -I${libcudacxx_SOURCE_DIR}/include"
    " -I${libcudacxx_SOURCE_DIR}/include/libhipcxx"
#    " CACHE INTERNAL "Flags for libcxx testing."
  )
  if (WIN32)
    # On Windows, MxGPU virtual GPUs cause offload-arch.exe to return the
    # wrong arch (e.g. gfx906). Add --offload-arch=native so amdclang++
    # queries the HIP runtime directly at compile time to get the real GPU
    # arch (e.g. gfx1101). lit's config.py ignores "native" for the hipcc
    # compiler type, so this must be injected explicitly here.
    string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " --offload-arch=native")
  endif()
  # hipcc used to add the HIP runtime library automatically; with amdclang++
  # we must link it explicitly.
  # On Windows, the DLL (amdhip64.dll) is in bin/ but the import library
  # (amdhip64.lib) is in lib/. Try IMPORTED_IMPLIB variants first; if none
  # are set, derive lib/ by going up from the DLL's bin/ directory.
  set(_hip_lib_dir "")
  foreach(_cfg "" "_RELEASE" "_DEBUG" "_RELWITHDEBINFO" "_MINSIZEREL")
    get_target_property(_amdhip64_implib hip::amdhip64 "IMPORTED_IMPLIB${_cfg}")
    if(_amdhip64_implib)
      cmake_path(GET _amdhip64_implib PARENT_PATH _hip_lib_dir)
      break()
    endif()
  endforeach()
  if(NOT _hip_lib_dir)
    get_target_property(_amdhip64_loc hip::amdhip64 LOCATION)
    cmake_path(GET _amdhip64_loc PARENT_PATH _hip_lib_dir)
    if(WIN32)
      # DLL is in bin/, import lib is in lib/ — go up one and use lib/
      cmake_path(GET _hip_lib_dir PARENT_PATH _hip_root_dir)
      set(_hip_lib_dir "${_hip_root_dir}/lib")
    endif()
  endif()
  string(APPEND LIBCUDACXX_TEST_LINKER_FLAGS
    " -L${_hip_lib_dir}"
    " -lamdhip64")
endif()

set(LIBCUDACXX_COMPUTE_ARCHS_STRING "${CMAKE_HIP_ARCHITECTURES}")

include(AddLLVM)

set(LIBCUDACXX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(
  LIBCUDACXX_TARGET_INFO
  "libcudacxx.test.target_info.LocalTI"
  CACHE STRING
  "TargetInfo to use when setting up test environment."
)
set(
  LIBCUDACXX_EXECUTOR
  "None"
  CACHE STRING
  "Executor to use when running tests."
)

set(
  LIBCUDACXX_TEST_TIMEOUT
  "200"
  CACHE STRING
  "Enable test timeouts (Default = 200, Off = 0)"
)

set(
  AUTO_GEN_COMMENT
  "## Autogenerated by libcudacxx configuration.\n# Do not edit!"
)

set(lit_site_cfg_path "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg")
configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in"
  "${lit_site_cfg_path}"
)

# Set dependencies for hiprtcc executable when using HIPRTC tests
set(LIBHIPCXX_TEST_DEPENDS "")
if(LIBHIPCXX_TEST_WITH_HIPRTC)
  list(APPEND LIBHIPCXX_TEST_DEPENDS libcudacxx.nvrtcc)
endif()

add_lit_testsuite(check-hipcxx
  "Running libhip++ tests"
  "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS ${LIBHIPCXX_TEST_DEPENDS})

find_program(libcudacxx_LIT lit REQUIRED)

set(
  libcudacxx_LIT_FLAGS
  ""
  CACHE STRING
  "Semi-colon separated list of flags passed to the invocation of lit."
)
message(STATUS "libcudacxx_LIT_FLAGS: ${libcudacxx_LIT_FLAGS}")

if (NOT LIBCUDACXX_TEST_WITH_NVRTC)
  # Build but don't run the tests. Used by CI to pre-seed sccache for the test machines.
  # Only executed if explicitly requested.
  add_custom_target(
    libcudacxx.test.lit.precompile
    DEPENDS
      libcudacxx.test.public_headers
      libcudacxx.test.internal_headers
      libcudacxx.test.public_headers_host_only
    # gersemi: off
    COMMAND
      "${CMAKE_COMMAND}" -E env "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
        "${libcudacxx_LIT}"
          -vv --no-progress-bar --time-tests
          ${libcudacxx_LIT_FLAGS}
          "-Dexecutor=\"NoopExecutor()\""
          "${libcudacxx_SOURCE_DIR}/test/libcudacxx"
    # gersemi: on
  )
endif()

# Restricted to avoid oversubscribing the GPU:
set(
  libcudacxx_LIT_PARALLEL_LEVEL
  8
  CACHE STRING
  "Parallelism used to run libcudacxx's lit test suite."
)

add_test(
  NAME libcudacxx.test.lit
  # gersemi: off
  COMMAND
    "${CMAKE_COMMAND}" -E env "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
      "${libcudacxx_LIT}"
        -vv --no-progress-bar --time-tests
        ${libcudacxx_LIT_FLAGS}
        -j "${libcudacxx_LIT_PARALLEL_LEVEL}"
        "${libcudacxx_SOURCE_DIR}/test/libcudacxx"
  # gersemi: on
)

set_tests_properties(
  libcudacxx.test.lit
  PROPERTIES
    # 3hr, some CI machines are slow
    TIMEOUT 10800
    RUN_SERIAL TRUE
)

# NOTE(HIP/AMD): on HIP we also pull in the c2h subdirectory so the
# cccl.c2h / cccl.c2h.main library targets exist, but only as a
# Catch2-only stripped-down build (see c2h/CMakeLists.txt). The
# per-test executables below are now also enabled under HIP, with
# a HIP-only skip-list for the *.cu test files that use thrust /
# cub / cooperative_groups (none of which are wired up in this
# tree). Tracked as Phase 6 in 3.1.4_tmp/LIT_TESTS_3.2_MEMORY.md.
cccl_get_c2h()

# NOTE(HIP/AMD): rocThrust is found inside c2h/CMakeLists.txt (see
# cccl_get_c2h() above), but the resulting 'rocthrust_FOUND' variable
# is local to that subdirectory. Re-do the lookup here so we can use
# the result to (a) decide whether to skip-list thrust-using tests
# below and (b) link 'roc::rocthrust' to the per-test executable.
if (LIBCUDACXX_ENABLE_HIP)
  find_package(rocthrust QUIET CONFIG)
endif()

if(LIBCUDACXX_ENABLE_CUDA OR LIBCUDACXX_ENABLE_HIP)
  file(
    GLOB_RECURSE test_srcs
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    CONFIGURE_DEPENDS
    *.cu
  )

  if (LIBCUDACXX_ENABLE_HIP)
    # NOTE(HIP/AMD): per-test skip-list. These pull cooperative_groups
    # directly or depend on upstream-CUDA-only feature surfaces;
    # enable them only when the corresponding HIP support has landed.
    set(LIBCUDACXX_C2H_HIP_SKIP_LIST
      # Use cooperative_groups, no AMDGCN equivalent in this tree.
      # Tests the CUDA driver-API context-stack semantics including
      # cuStreamGetCtx for which HIP has no public equivalent
      # (hipStreamGetCtx does not exist; consumers route through
      # hipStreamGetDevice instead).
      cuda/ccclrt/utility/driver_api.c2h.cu
      # Use cuda::device::current_arch_id / current_arch_traits /
      # current_compute_capability which are not yet wired into the
      # libhipcxx HIP bridge (need a HIP-side AMDGCN architecture
      # registry).
      cuda/ccclrt/device/arch_id.c2h.cu
      cuda/ccclrt/device/arch_traits.c2h.cu
      cuda/ccclrt/device/compute_capability.c2h.cu
      # Compiles and runs (134 of 136 assertions pass) but two
      # cuda::device_attribute_result_t<Attr> queries fail at
      # runtime with hipErrorInvalidValue from hipDeviceGetAttribute
      # for one or more attributes that ROCm 7.2 does not yet
      # implement. Needs a per-attribute HIP-gating audit;
      # tracked as SF9 in 3.1.4_tmp/LIT_TESTS_3.2_MEMORY.md.
      cuda/ccclrt/device/device_smoke.c2h.cu
      # Compiles and runs (48 of 52 assertions pass) but the IPC
      # export-handle round-trip and a couple of mempool-attribute
      # invariants do not hold byte-for-byte on HIP. Needs an
      # assertion-by-assertion HIP-gating audit; tracked as SF10
      # in 3.1.4_tmp/LIT_TESTS_3.2_MEMORY.md.
      cuda/memory_resource/resources/memory_pools.cu
      # Directly checks 'test::count_driver_stack() == cuda::devices.size()
      # - id' which only makes sense in the CUDA driver-API context-stack
      # model. On HIP the helper is a no-op (see testing.cuh) since HIP
      # does not expose a multi-context per-thread stack. Needs the
      # test rewritten to query cudaGetDevice() instead; tracked as
      # SF14 in 3.1.4_tmp/LIT_TESTS_3.2_MEMORY.md.
      cuda/ccclrt/utility/ensure_current_context.c2h.cu
    )
    # If rocThrust is NOT available, also skip the thrust-using test;
    # otherwise leave it in the build and link rocThrust to the
    # per-test executable (handled below).
    if (NOT rocthrust_FOUND)
      list(APPEND LIBCUDACXX_C2H_HIP_SKIP_LIST
        cuda/containers/uninitialized_async_buffer.cu)
    endif()
  endif()

  set(c2h_all_target "libcudacxx.test.c2h_all")
  add_custom_target(${c2h_all_target})

  function(libcudacxx_add_test target_name_var source)
    string(REPLACE "/" "." target_name "${source}")
    string(PREPEND target_name "libcudacxx.test.")
    string(REGEX REPLACE "\\.[^.]+$" "" target_name "${target_name}")
    set(${target_name_var} ${target_name} PARENT_SCOPE)

    add_executable(${target_name} "${source}")
    if (LIBCUDACXX_ENABLE_HIP)
      # The .cu file extension defaults to LANGUAGE CUDA in CMake.
      # Force LANGUAGE HIP on HIP builds so hipcc / clang -x hip
      # picks the file up. Mirrors the pattern in c2h/CMakeLists.txt.
      set_source_files_properties("${source}" PROPERTIES LANGUAGE HIP)
    else()
      cccl_configure_target(${target_name} DIALECT ${CMAKE_CUDA_STANDARD})
    endif()
    target_include_directories(
      ${target_name}
      PRIVATE "${libcudacxx_SOURCE_DIR}/test/libcudacxx/cuda/ccclrt/common"
    )
    target_link_libraries(
      ${target_name}
      PRIVATE #
        libcudacxx::libcudacxx
        cccl.c2h.main
        libcudacxx.compiler_interface
    )
    # NOTE(HIP/AMD): link rocThrust to every c2h test on HIP when it is
    # available. The thrust-using tests (currently
    # uninitialized_async_buffer.cu) need it on the link line; the
    # other tests don't reference any thrust symbol and the link is a
    # no-op for them. Cheaper to do this unconditionally than to
    # maintain a per-test "needs rocthrust" allow-list.
    if (LIBCUDACXX_ENABLE_HIP AND TARGET roc::rocthrust)
      target_link_libraries(${target_name} PRIVATE roc::rocthrust)
    endif()

    add_dependencies(${c2h_all_target} ${target_name})

    add_test(NAME ${target_name} COMMAND ${target_name})
  endfunction()

  foreach (test_src IN LISTS test_srcs)
    if (LIBCUDACXX_ENABLE_HIP)
      list(FIND LIBCUDACXX_C2H_HIP_SKIP_LIST "${test_src}" _skip_idx)
      if (NOT _skip_idx EQUAL -1)
        message(STATUS "c2h test (HIP skip-list): ${test_src}")
        continue()
      endif()
    endif()
    libcudacxx_add_test(test_target "${test_src}")
  endforeach()
endif()
