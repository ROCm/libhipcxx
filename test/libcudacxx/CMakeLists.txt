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

option(
  LIBCUDACXX_TEST_WITH_NVRTC
  "Test libcu++ with runtime compilation instead of offline compilation. Only runs device side tests."
  OFF
)
###############################################################################
### C2H tests:
# NOTE(HIP/AMD): on HIP we also pull in the c2h subdirectory so the
# cccl.c2h / cccl.c2h.main library targets exist, but only as a
# Catch2-only stripped-down build (see c2h/CMakeLists.txt). The
# per-test executables below are now also enabled under HIP, with
# a HIP-only skip-list for the *.cu test files that use thrust /
# cub / cooperative_groups (none of which are wired up in this
# tree).
#
# NOTE(HIP/AMD): cccl_get_c2h() pulls add_subdirectory(c2h), and
# c2h/CMakeLists.txt unconditionally calls cccl_get_catch2(), which
# CPM-resolves to 'git clone gh:catchorg/Catch2#3.8.1'. That git
# clone fails in air-gapped CI containers. Provide an opt-out
# 'LIBCUDACXX_ENABLE_C2H_TESTS' (default ON to preserve the existing
# behaviour) so air-gapped builds can run with
# '-DLIBCUDACXX_ENABLE_C2H_TESTS=OFF'. The lit test suite (which is
# the bulk of the test coverage) does not depend on c2h / Catch2 at
# all, so disabling c2h on air-gapped CI loses only the c2h-style
# per-test executables (libcudacxx.test.*.cu); 'ctest' / lit / the
# header-self-contained tests still run.
option(
  LIBCUDACXX_ENABLE_C2H_TESTS
  "Enable the c2h test framework (catch2-based per-test .cu executables). Disabling this skips the cccl_get_catch2() git fetch and the libcudacxx.test.c2h_all umbrella target. Default ON."
  ON
)
if (LIBCUDACXX_ENABLE_C2H_TESTS)
  cccl_get_c2h()
endif()

# NOTE(HIP/AMD): rocThrust is found inside c2h/CMakeLists.txt (see
# cccl_get_c2h() above), but the resulting 'rocthrust_FOUND' variable
# is local to that subdirectory. Re-do the lookup here so we can use
# the result to (a) decide whether to skip-list thrust-using tests
# below and (b) link 'roc::rocthrust' to the per-test executable.
if (LIBCUDACXX_ENABLE_HIP)
  find_package(rocthrust QUIET CONFIG)
endif()

if(LIBCUDACXX_ENABLE_C2H_TESTS AND NOT LIBCUDACXX_TEST_WITH_NVRTC AND (LIBCUDACXX_ENABLE_CUDA OR LIBCUDACXX_ENABLE_HIP))
  file(
    GLOB_RECURSE test_srcs
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    CONFIGURE_DEPENDS
    *.cu
  )

  if (LIBCUDACXX_ENABLE_HIP)
    # NOTE(HIP/AMD): per-test skip-list (regex/substring patterns matched
    # against the test source path below). These depend on upstream-CUDA-only
    # feature surfaces; re-enable them when the corresponding HIP support has
    # landed.
    set(LIBCUDACXX_C2H_HIP_SKIP_LIST
      # cuda::device::current_arch_id / current_arch_traits /
      # current_compute_capability are not yet wired into the libhipcxx HIP
      # bridge (need a HIP-side AMDGCN architecture registry).
      cuda/ccclrt/device/arch_id.c2h.cu
      cuda/ccclrt/device/arch_traits.c2h.cu
      cuda/ccclrt/device/compute_capability.c2h.cu
      # launch: host_launch / dynamic_shared_memory / launch_smoke /
      # configuration ARE enabled on HIP (driver launch ported to
      # hipModuleLaunchKernel; device-side hierarchy queries enabled).
      # hierarchy_smoke / hierarchy_custom_types are enabled too (cuda::vector_type_t
      # is now HIP-enabled via <hip/hip_vector_types.h>; matches 3.2.1).
      # containers: cuda::buffer / uninitialized_async_buffer additionally need
      # <cub/device/device_transform.cuh> ported to hipCUB.
      cuda/containers/
      # shared_resource instantiates cuda::__uninitialized_async_buffer (above).
      cuda/memory_resource/shared_resource.cu
      # PSTL parallel algorithms (cuda::execution) -- new in 3.3, need a HIP
      # execution / device_transform port.
      pstl_
    )
  endif()

  set(c2h_all_target "libcudacxx.test.c2h_all")
  add_custom_target(${c2h_all_target})

  # NOTE(HIP/AMD): see the long-form comment inside libcudacxx_add_test()
  # below for the rationale. Briefly: commit 7b770425d9 extended the strict
  # cccl warning regime ('-Wall -Wextra -Werror') to the HIP language;
  # several upstream NVIDIA c2h test sources (which we import verbatim and
  # don't fork) carry pre-existing low-severity warnings that now hard-fail
  # the build. We relax exactly four categories on HIP for c2h test
  # executables only, via this INTERFACE library. It MUST be linked AFTER
  # libcudacxx.compiler_interface (which carries the '-Wall -Wextra') so
  # that clang sees the '-Wno-*' flags LATER on the command line and
  # therefore wins the left-to-right precedence battle. Linking order in
  # target_link_libraries() is what controls the resulting -W flag order
  # on the compile command.
  if (LIBCUDACXX_ENABLE_HIP)
    add_library(libcudacxx.test.c2h_warning_relaxation INTERFACE)
    target_compile_options(libcudacxx.test.c2h_warning_relaxation INTERFACE
      -Wno-unused-parameter
      -Wno-unused-variable
      -Wno-unused-value
      -Wno-sign-compare
    )
  endif()

  function(libcudacxx_add_test target_name_var source)
    string(REPLACE "/" "." target_name "${source}")
    string(PREPEND target_name "libcudacxx.test.")
    string(REGEX REPLACE "\\.[^.]+$" "" target_name "${target_name}")
    set(${target_name_var} ${target_name} PARENT_SCOPE)

# <<<<<<< OLD CODE from 313f9fe948 (67794b2eb4) - COMMENTED OUT
#     add_executable(${target_name} "${source}")
#     if (LIBCUDACXX_ENABLE_HIP)
#       # The .cu file extension defaults to LANGUAGE CUDA in CMake.
#       # Force LANGUAGE HIP on HIP builds so hipcc / clang -x hip
#       # picks the file up. Mirrors the pattern in c2h/CMakeLists.txt.
#       set_source_files_properties("${source}" PROPERTIES LANGUAGE HIP)
#     else()
#       cccl_configure_target(${target_name} DIALECT ${CMAKE_CUDA_STANDARD})
#     endif()
# =======
    cccl_add_executable(
      ${target_name}
      ADD_CTEST
      NO_METATARGETS
      DIALECT ${CMAKE_CUDA_STANDARD}
      SOURCES "${source}"
    )
# >>>>>>> END NEW CODE (67794b2eb4)
    target_include_directories(
      ${target_name}
      PRIVATE
        "${libcudacxx_SOURCE_DIR}/test/libcudacxx/cuda/ccclrt/common"
        "${libcudacxx_SOURCE_DIR}/test/support"
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

    # NOTE(HIP/AMD): the c2h test sources are upstream NVIDIA cccl test
    # code imported wholesale; we don't maintain a HIP-side fork of them.
    # libcudacxx.compiler_interface (extended to HIP in commit
    # 7b770425d9 "cccl warning flags to HIP language") is intentionally
    # strict for OUR headers (catches HIP-port regressions), but several
    # of the imported tests carry pre-existing low-severity warnings
    # that surface as -Werror under that strict regime. Relax these four
    # categories on HIP for c2h test executables only, via the INTERFACE
    # library 'libcudacxx.test.c2h_warning_relaxation' LINKED AFTER
    # libcudacxx.compiler_interface so the '-Wno-*' flags win the
    # left-to-right precedence battle on the clang command line.
    if (LIBCUDACXX_ENABLE_HIP AND TARGET libcudacxx.test.c2h_warning_relaxation)
      target_link_libraries(
        ${target_name}
        PRIVATE libcudacxx.test.c2h_warning_relaxation
      )
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(${target_name} PRIVATE "-Wno-attributes")
    endif()

    add_dependencies(${c2h_all_target} ${target_name})
  endfunction()

  foreach (test_src IN LISTS test_srcs)
    if (LIBCUDACXX_ENABLE_HIP)
      set(_skip_test OFF)
      foreach (_skip_pat IN LISTS LIBCUDACXX_C2H_HIP_SKIP_LIST)
        if (test_src MATCHES "${_skip_pat}")
          set(_skip_test ON)
          break()
        endif()
      endforeach()
      if (_skip_test)
        message(STATUS "c2h test (HIP skip-list): ${test_src}")
        continue()
      endif()
    endif()
    libcudacxx_add_test(test_target "${test_src}")
  endforeach()
elseif (LIBCUDACXX_ENABLE_CUDA OR LIBCUDACXX_ENABLE_HIP)
  # NOTE(HIP/AMD): when c2h tests are disabled (LIBCUDACXX_ENABLE_C2H_TESTS=OFF,
  # used in air-gapped CI to avoid the cccl_get_catch2() git fetch), still
  # define an *empty* 'libcudacxx.test.c2h_all' target. The
  # 'libcudacxx-base' build preset in CMakePresets.json lists this target
  # by name; without this stub a 'cmake --build --preset libcudacxx-cppNN'
  # invocation would fail with 'ninja: error: unknown target
  # libcudacxx.test.c2h_all'. Building the empty target is a near-zero-cost
  # no-op.
  add_custom_target(libcudacxx.test.c2h_all)
  message(STATUS
    "c2h tests disabled (LIBCUDACXX_ENABLE_C2H_TESTS=OFF). "
    "libcudacxx.test.c2h_all defined as an empty stub.")
endif()

###############################################################################
### Lit tests:
macro(pythonize_bool var)
  if (${var})
    set(${var} True)
  else()
    set(${var} False)
  endif()
endmacro()

# NOTE(HIP/AMD): dlpack is *only* consumed by the NV-only TMA descriptor
# code in 'include/cuda/__tma/make_tma_descriptor.h', whose entire body is
# guarded by '_CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) &&
# _CCCL_HAS_INCLUDE(<dlpack/dlpack.h>)'. The matching test
# 'test/libcudacxx/cuda/tma/make_tma_descriptor.pass.cpp' is itself marked
# '// UNSUPPORTED: hipcc, hiprtc' and 'pre-sm-90'. On HIP-only builds the
# dlpack include is therefore dead code -- and 'cccl_get_dlpack()' would
# otherwise call CPM/'git clone gh:dmlc/dlpack#v1.2', which fails in
# air-gapped CI containers (observed on the ROCm-DS Jenkins job
# 'libhipcxx/155': 'Failed to clone repository ...dlpack...'). Gate the
# fetch on 'LIBCUDACXX_ENABLE_CUDA' so HIP-only builds skip it entirely.
# NOTE(HIP/AMD): 3.3 adds general mdspan<->dlpack conversion tests
# (cuda/containers/views/mdspan/{mdspan_to_dlpack,dlpack_to_mdspan}) that need
# the header on HIP too, so also fetch it for HIP builds.
if (LIBCUDACXX_ENABLE_CUDA OR LIBCUDACXX_ENABLE_HIP)
  cccl_get_dlpack()
endif()

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
  # NOTE(HIP/AMD): the lit.site.cfg.in template substitutes
  # '@LIBCUDACXX_CUDA_TEST_WITH_NVRTC@' into 'config.is_nvrtc = ...'.
  # If the variable is unset (which is the case in this HIPRTC branch
  # because we don't go through the LIBCUDACXX_TEST_WITH_NVRTC arm
  # above), the substitution yields 'config.is_nvrtc =' which is a
  # Python syntax error. Set the value explicitly to "False" so the
  # generated lit.site.cfg parses.
  set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "False")
  set(LIBCUDACXX_FORCE_INCLUDE "")
  # NOTE(HIP/AMD): HIPRTC's COMGR-bundled inline-clang sandbox does
  # NOT have <hip/hip_runtime.h>, <hip/hip_bf16.h>, <hip/hip_fp16.h>,
  # <hip/hip_fp8.h> on its include search path -- only the C++ stdlib
  # and clang's builtin headers are reachable. Pass the host ROCm
  # <hip/...> include dir AND clang's own resource dirs through to the
  # hiprtcc test driver via -I; hiprtcc forwards every -I to the
  # nvrtcCompileProgram() options, so the source files compiled inside
  # COMGR can finally find these runtime headers. Without this the
  # bf16/fp8 type forward-declarations in
  # <cuda/std/__cccl/extended_data_types.h> never get a complete
  # definition and every TU touching __fp_from_storage<__nv_bfloat16>
  # fails. Mirrors the 3.1.4 setup that was lost during the 3.2 port.
  set(LIBCUDACXX_TEST_COMPILER_FLAGS "-I'${HIP_INCLUDE_DIR}'")
  # NOTE(HIP/AMD): We need to identify clang version to find from the path to clang new header.
  # The header is used in libhipcxx/include/cuda/std/__memory/construct_at.h.
  execute_process(
    COMMAND "${HIP_HIPCC_EXECUTABLE}" --print-resource-dir
    OUTPUT_VARIABLE _clang_abs_resource_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  cmake_path(GET _clang_abs_resource_dir FILENAME _clang_version)
  set(_clang_include_dir "lib/llvm/lib/clang/${_clang_version}/include")
  set(_clang_cuda_wrappers_dir "${_clang_include_dir}/cuda_wrappers")
  string(JOIN " " LIBCUDACXX_TEST_COMPILER_FLAGS
    -I${HIP_INCLUDE_DIR}
    -I${HIP_INCLUDE_DIR}/../${_clang_include_dir}
    -I${HIP_INCLUDE_DIR}/../${_clang_cuda_wrappers_dir}
  )
  # NOTE(HIP/AMD): the hiprtcc test driver does NOT produce a host
  # '.exe' -- it writes per-arch '.gpu' device-code files plus a
  # '<test>.build.yml' index next to where the executable would
  # normally land. The lit harness therefore cannot exec the .exe
  # directly. Wrap the executable invocation with hiprtcc itself
  # (PrefixExecutor) so 'hiprtcc <test>.exe ...' detects the .exe
  # extension, loads the matching .build.yml, and launches the
  # pre-built device kernel via hipModuleLaunchKernel. Mirrors the
  # NVRTC branch above and the 3.1.4 setup that was lost during the
  # 3.2 port. Without this, every '.compile.pass.cpp' / '.pass.cpp'
  # ends up UNRESOLVED with 'FileNotFoundError: <test>.exe'.
  set(
    LIBCUDACXX_EXECUTOR
    "PrefixExecutor(['${LIBCUDACXX_CUDA_COMPILER}'], LocalExecutor())"
  )
else() # NOT LIBCUDACXX_TEST_WITH_NVRTC
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
    # NOTE(HIP/AMD): hipcc does NOT predefine '__HIP_PLATFORM_AMD__' /
    # '__HIP_ROCclr__' on its own (verified via 'hipcc --offload-arch=...
    # -x hip -dM -E'); the macros are normally added by CMake's HIP
    # language module on the cccl.c2h / public-header-test paths. The
    # lit harness doesn't run through that module, so without these
    # explicit '-D's every '_CCCL_HIP_COMPILATION()'-gated body in
    # libhipcxx headers (and in HIP-only tests like
    # 'ptx.hip_emu_bmsk_parity.compile.pass.cpp') silently compiles to a
    # no-op under lit. Force the defines here so the HIP-arm code paths
    # actually exercise in lit runs -- pairs with the
    # ptx.{bmsk,elect.sync,fence,get_sreg,shfl,shl,shr,trap}.compile.pass.cpp
    # tests enabled on HIP in commit 9521b2f98e.
    string(
      APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
      " -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1"
    )
  endif()
endif()

# enable exceptions and assertions in tests
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -DCCCL_ENABLE_ASSERTIONS")
# NOTE(HIP/AMD): dlpack is now fetched for HIP too (see comment at
# 'cccl_get_dlpack()' above) for the mdspan<->dlpack conversion tests.
if (LIBCUDACXX_ENABLE_CUDA OR LIBCUDACXX_ENABLE_HIP)
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -I ${dlpack_SOURCE_DIR}/include")
endif()

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

set(LIBCUDACXX_TEST_STANDARD_VER "c++${CMAKE_CUDA_STANDARD}")

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
    # HACK: There is no way to tell CMake/ninja to always build a target serially,
    # so we make this target depend on all other libcudacxx targets to avoid oversubscribing
    # the build machine.
    # FIXME: This has nasty side effects:
    # - It's fragile and must be updated every time we add new targets to libcudacxx
    # - It oversubs `-dev` presets that configure libcudacxx alongside other CCCL projects
    # - It makes it impossible to just build this target alone since it brings in the world
    # See related issue https://github.com/NVIDIA/cccl/issues/6163.
    DEPENDS
      libcudacxx.test.public_headers
      libcudacxx.test.internal_headers
      libcudacxx.test.public_headers_host_only
      libcudacxx.test.c2h_all
    # gersemi: off
    COMMAND
      "${CMAKE_COMMAND}" -E env "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
        "${libcudacxx_LIT}"
          -vv --no-progress-bar --time-tests
          ${libcudacxx_LIT_FLAGS}
          "-Dexecutor=\"NoopExecutor()\""
          "${libcudacxx_SOURCE_DIR}/test/libcudacxx"
    # gersemi: on
    USES_TERMINAL
  )
endif()

# NOTE(HIP/AMD): the upstream default of 8 was chosen to avoid
# oversubscribing the GPU. On AMD multi-GPU systems (e.g. an 8x
# MI210 box) this is a ~14x slowdown vs the 'ninja check-hipcxx'
# target, which lets lit auto-detect (defaulting to
# multiprocessing.cpu_count()). Match check-hipcxx's behaviour by
# default: 0 tells lit to auto-detect. Override at configure time
# with -Dlibcudacxx_LIT_PARALLEL_LEVEL=N if a particular CI
# machine cannot tolerate the auto-detected level.
set(
  libcudacxx_LIT_PARALLEL_LEVEL
  0
  CACHE STRING
  "Parallelism used to run libcudacxx's lit test suite. 0 = lit auto-detects (cpu_count())."
)

# NOTE(HIP/AMD): only pass '-j N' if the user explicitly set a
# non-zero level. When 0 (the default), omit the flag entirely so
# lit auto-detects -- matches the 'ninja check-hipcxx' invocation
# above which has no -j cap.
set(_libcudacxx_lit_j_flag "")
if(libcudacxx_LIT_PARALLEL_LEVEL GREATER 0)
  set(_libcudacxx_lit_j_flag -j "${libcudacxx_LIT_PARALLEL_LEVEL}")
endif()

add_test(
  NAME libcudacxx.test.lit
  # gersemi: off
  COMMAND
    "${CMAKE_COMMAND}" -E env "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
      "${libcudacxx_LIT}"
        -vv --no-progress-bar --time-tests
        ${libcudacxx_LIT_FLAGS}
        ${_libcudacxx_lit_j_flag}
        "${libcudacxx_SOURCE_DIR}/test/libcudacxx"
    # gersemi: on
    USES_TERMINAL
)

set_tests_properties(
  libcudacxx.test.lit
  PROPERTIES
    # 6hr to match CI timeout
    TIMEOUT 21600
    RUN_SERIAL TRUE
)
