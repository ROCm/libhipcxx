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

if (LIBCUDACXX_ENABLE_CUDA)

  find_package(CUDAToolkit REQUIRED)
  get_target_property(CUDA_INCLUDE_DIR CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)

  set(LIBCUDACXX_HIGHEST_COMPUTE_ARCH 90)

# <<<<<<< OLD CODE from 6def8aee26 (1789a7554e) - COMMENTED OUT
#   foreach (COMPUTE_ARCH ${LIBCUDACXX_COMPUTE_ARCHS})
#     set(_compute_message "${_compute_message} sm_${COMPUTE_ARCH}")
#     set(LIBCUDACXX_COMPUTE_ARCHS_STRING "${LIBCUDACXX_COMPUTE_ARCHS_STRING} ${COMPUTE_ARCH}")
#   endforeach ()
#
#   message(STATUS "Enabled CUDA architectures:${_compute_message}")
#
# elseif(LIBCUDACXX_ENABLE_HIP)
#
#   # If LIBCUDACXX_COMPUTE_ARCHS is not set, use the HIP architectures
#   if (NOT LIBCUDACXX_COMPUTE_ARCHS)
#     set(LIBCUDACXX_COMPUTE_ARCHS ${CMAKE_HIP_ARCHITECTURES})
#   endif()
# =======
foreach (COMPUTE_ARCH ${LIBCUDACXX_COMPUTE_ARCHS})
  set(_compute_message "${_compute_message} sm_${COMPUTE_ARCH}")
  set(LIBCUDACXX_COMPUTE_ARCHS_STRING "${LIBCUDACXX_COMPUTE_ARCHS_STRING} ${COMPUTE_ARCH}")
endforeach ()
# >>>>>>> END NEW CODE (1789a7554e)

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
  set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_BINARY_DIR}/libcudacxx/test/utils/nvidia/nvrtc/nvrtcc")
  set(LIBCUDACXX_CUDA_COMPILER_ARG1 "")
  set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "True")
  # Use the NVRTCC utility to run the built test outputs
  set(LIBCUDACXX_EXECUTOR "PrefixExecutor(['${LIBCUDACXX_CUDA_COMPILER}'], LocalExecutor())")
# <<<<<<< OLD CODE from f1c3a9dfff (6ea2021a22) - COMMENTED OUT
# elseif (LIBHIPCXX_TEST_WITH_HIPRTC)
#   # TODO: Use project properties to get path to binary.
#   # Should also set up dependency on the project when NVRTC is enabled
#   set(CMAKE_CUDA_COMPILER_ID "HIPCC")
#   set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_BINARY_DIR}/test/utils/amd/hiprtc/hiprtcc${CMAKE_EXECUTABLE_SUFFIX}")
#   message(STATUS "LIBCUDACXX_CUDA_COMPILER: ${LIBCUDACXX_CUDA_COMPILER}")
#   set(LIBCUDACXX_CUDA_COMPILER_ARG1 "")
#   set(LIBCUDACXX_HIP_TEST_WITH_HIPRTC "True")
# =======
  # Enable 128-bit types for NVRTC
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -device-int128")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -device-float128")
else() # NOT LIBCUDACXX_TEST_WITH_NVRTC
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
  set(LIBCUDACXX_FORCE_INCLUDE "-include ${libcudacxx_SOURCE_DIR}/test/libcudacxx/force_include.h")
  set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}")
# >>>>>>> END NEW CODE (6ea2021a22)
  set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "False")
# <<<<<<< OLD CODE from 6efaca82a9 (cfe657d109) - COMMENTED OUT
#   set(LIBCUDACXX_FORCE_INCLUDE "")
#   set(LIBCUDACXX_TEST_COMPILER_FLAGS "-I'${HIP_INCLUDE_DIR}'")
#   # NOTE(HIP/AMD): We need to identify clang version to find from the path to clang new header.
#   # The header is used in libhipcxx/include/cuda/std/__memory/construct_at.h.
#   execute_process(
#     COMMAND "${HIP_HIPCC_EXECUTABLE}" --print-resource-dir
#     OUTPUT_VARIABLE _clang_abs_resource_dir
#     OUTPUT_STRIP_TRAILING_WHITESPACE
#     COMMAND_ERROR_IS_FATAL ANY
#   )
#   cmake_path(GET _clang_abs_resource_dir FILENAME _clang_version)
#   set(_clang_include_dir "lib/llvm/lib/clang/${_clang_version}/include")
#   set(_clang_cuda_wrappers_dir "${_clang_include_dir}/cuda_wrappers")
#   string(JOIN " " LIBCUDACXX_TEST_COMPILER_FLAGS
#     -I${HIP_INCLUDE_DIR}
#     -I${HIP_INCLUDE_DIR}/../${_clang_include_dir}
#     -I${HIP_INCLUDE_DIR}/../${_clang_cuda_wrappers_dir}
#   )
#   # Use the NVRTCC utility to run the built test outputs
#   set(LIBCUDACXX_EXECUTOR "PrefixExecutor(['${LIBCUDACXX_CUDA_COMPILER}'], LocalExecutor())")
# else() # NOT LIBCUDACXX_TEST_WITH_NVRTC
#   if (LIBCUDACXX_ENABLE_CUDA)
#     set(LIBCUDACXX_FORCE_INCLUDE "-include ${libhipcxx_SOURCE_DIR}/test/libcudacxx/force_include.h")
#     set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}")
#     set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "False")
#     set(LIBCUDACXX_TEST_COMPILER_FLAGS "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
#   elseif(LIBCUDACXX_ENABLE_HIP)
#     set(LIBCUDACXX_FORCE_INCLUDE "-include ${libhipcxx_SOURCE_DIR}/test/libcudacxx/force_include_hip.h")
#     #temporary workaround until we've found a way to make cmake find hipcc
#     set(LIBCUDACXX_CUDA_COMPILER "${HIP_HIPCC_EXECUTABLE}")
#     set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
#     set(CMAKE_CUDA_COMPILER_ID "HIPCC")
#     set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "False")
#     set(LIBCUDACXX_HIP_TEST_WITH_HIPRTC "False")
#     set(LIBCUDACXX_TEST_COMPILER_FLAGS "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
#   endif()
# =======
# >>>>>>> END NEW CODE (cfe657d109)
endif()

# enable exceptions and assertions in tests
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -DCCCL_ENABLE_ASSERTIONS")

# enable optional<T&>
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -DCCCL_ENABLE_OPTIONAL_REF")

# Disable dialect deprecation
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT")
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -DLIBCUDACXX_IGNORE_DEPRECATED_ABI")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  string(APPEND LIBCUDACXX_TEST_LINKER_FLAGS " -latomic")
endif()

if (NOT MSVC AND NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
  set(LIBCUDACXX_WARNING_LEVEL "--compiler-options=-Wall --compiler-options=-Wextra")
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
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " ${CMAKE_CUDA_FLAGS}"
    " -Xclang -fcuda-allow-variadic-functions"
    " -Xclang -Wno-unused-parameter"
    " -Wno-unknown-cuda-version"
    " ${LIBCUDACXX_FORCE_INCLUDE}"
    " -I${libhipcxx_SOURCE_DIR}/include"
    " ${LIBCUDACXX_WARNING_LEVEL}")

  string(APPEND LIBCUDACXX_TEST_LINKER_FLAGS
    " ${CMAKE_CUDA_FLAGS}"
    " -L${CUDAToolkit_LIBRARY_DIR}"
    " -lcuda"
    " -lcudart")
elseif (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " ${LIBCUDACXX_FORCE_INCLUDE}"
    " ${LIBCUDACXX_WARNING_LEVEL}"
    " -Wno-deprecated-gpu-targets")
elseif (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVHPC")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " -stdpar")
  string(APPEND LIBCUDACXX_TEST_LINKER_FLAGS
    " -stdpar")
endif()

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "AMDCLANG")
  message(STATUS "libhipcxx_SOURCE_DIR: ${libhipcxx_SOURCE_DIR}" )
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
#    "${LIBCUDACXX_TEST_COMPILER_FLAGS} \
    " ${LIBCUDACXX_FORCE_INCLUDE}"
    " -I${CMAKE_SOURCE_DIR}/include"
    " -I${libhipcxx_SOURCE_DIR}/include"
    " -I${libhipcxx_SOURCE_DIR}/include/libhipcxx"
    " -O3"  # hipcc defaults to -O3; amdclang++ invoked directly does not
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

set(LIBCUDACXX_TARGET_INFO "libcudacxx.test.target_info.LocalTI" CACHE STRING
"TargetInfo to use when setting up test environment.")
set(LIBCUDACXX_EXECUTOR "None" CACHE STRING
"Executor to use when running tests.")

set(LIBCUDACXX_TEST_TIMEOUT "200" CACHE STRING "Enable test timeouts (Default = 200, Off = 0)")

set(AUTO_GEN_COMMENT "## Autogenerated by libcudacxx configuration.\n# Do not edit!")

set(lit_site_cfg_path "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg")
configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in"
  "${lit_site_cfg_path}")

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

set(libcudacxx_LIT_FLAGS "" CACHE STRING "Semi-colon separated list of flags passed to the invocation of lit.")
message(STATUS "libcudacxx_LIT_FLAGS: ${libcudacxx_LIT_FLAGS}")

# Restricted to avoid oversubscribing the GPU:
set(libcudacxx_LIT_PARALLEL_LEVEL 8 CACHE STRING
"Parallelism used to run libcudacxx's lit test suite."
)

if (LIBCUDACXX_ENABLE_CUDA)
  if (NOT LIBCUDACXX_TEST_WITH_NVRTC)
    # Build but don't run the tests. Used by CI to pre-seed sccache for the test machines.
    # Only executed if explicitly requested.
    add_custom_target(libcudacxx.test.lit.precompile
      DEPENDS libcudacxx.test.public_headers libcudacxx.test.internal_headers libcudacxx.test.public_headers_host_only
      COMMAND "${CMAKE_COMMAND}" -E env "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
      "${libcudacxx_LIT}" -vv --no-progress-bar --time-tests ${libcudacxx_LIT_FLAGS}
      "-Dexecutor=\"NoopExecutor()\"" "${libhipcxx_SOURCE_DIR}/test/libcudacxx"
    )

  endif()
elseif(LIBCUDACXX_ENABLE_HIP)
  if (NOT LIBHIPCXX_TEST_WITH_HIPRTC)
    # Build but don't run the tests. Used by CI to pre-seed sccache for the test machines.
    # Only executed if explicitly requested.
    add_custom_target(libcudacxx.test.lit.precompile
      DEPENDS libcudacxx.test.public_headers libcudacxx.test.internal_headers libcudacxx.test.public_headers_host_only
      COMMAND "${CMAKE_COMMAND}" -E env "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
      "${libcudacxx_LIT}" -vv --no-progress-bar ${libcudacxx_LIT_FLAGS} -j "${libcudacxx_LIT_PARALLEL_LEVEL}" "-Dexecutor=\"NoopExecutor()\"" "${libhipcxx_SOURCE_DIR}/test/libcudacxx"
    )

  endif()
endif()

add_test(NAME libcudacxx.test.lit COMMAND
  "${CMAKE_COMMAND}" -E env
    "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
  "${libcudacxx_LIT}" -vv --no-progress-bar --time-tests ${libcudacxx_LIT_FLAGS}
    -j "${libcudacxx_LIT_PARALLEL_LEVEL}" "${libhipcxx_SOURCE_DIR}/test/libcudacxx"
)

set_tests_properties(libcudacxx.test.lit PROPERTIES
  TIMEOUT 10800 # 3hr, some CI machines are slow
  RUN_SERIAL TRUE
)

# Add c2h tests (CUDA only - c2h infrastructure is CUDA-specific):
if(LIBCUDACXX_ENABLE_CUDA)
  cccl_get_c2h()

# <<<<<<< OLD CODE from 6b318af440 (8dad8721b7) - COMMENTED OUT
#   file(GLOB_RECURSE test_srcs
#     RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
#     CONFIGURE_DEPENDS
#     *.cu
# =======
set(c2h_all_target "libcudacxx.test.c2h_all")
add_custom_target(${c2h_all_target})

function(libcudacxx_add_test target_name_var source)
  string(REPLACE "/" "." target_name "${source}")
  string(PREPEND target_name "libcudacxx.test.")
  string(REGEX REPLACE "\\.[^.]+$" "" target_name "${target_name}")
  set(${target_name_var} ${target_name} PARENT_SCOPE)

  add_executable(${target_name} "${source}")
  cccl_configure_target(${target_name} DIALECT ${CMAKE_CUDA_STANDARD})
  target_include_directories(${target_name} PRIVATE "${libcudacxx_SOURCE_DIR}/test/libcudacxx/cuda/ccclrt/common")
  target_link_libraries(${target_name} PRIVATE
    libcudacxx::libcudacxx
    cccl.c2h.main
    libcudacxx.compiler_interface
# >>>>>>> END NEW CODE (8dad8721b7)
  )

  set(c2h_all_target "libcudacxx.test.c2h_all")
  add_custom_target(${c2h_all_target})

  function(libcudacxx_add_test target_name_var source)
    string(REPLACE "/" "." target_name "${source}")
    string(PREPEND target_name "libcudacxx.test.")
    string(REGEX REPLACE "\\.[^.]+$" "" target_name "${target_name}")
    set(${target_name_var} ${target_name} PARENT_SCOPE)

    add_executable(${target_name} "${source}")
    cccl_configure_target(${target_name} DIALECT ${CMAKE_CUDA_STANDARD})
    target_include_directories(${target_name} PRIVATE "${libcudacxx_SOURCE_DIR}/test/libcudacxx/cuda/ccclrt/common")
    target_link_libraries(${target_name} PRIVATE
      libcudacxx::libcudacxx
      cccl.c2h.main
      cccl.compiler_interface_cpp${CMAKE_CUDA_STANDARD}
    )

    add_dependencies(${c2h_all_target} ${target_name})

    add_test(NAME ${target_name} COMMAND ${target_name})
  endfunction()

  foreach (test_src IN LISTS test_srcs)
    libcudacxx_add_test(test_target "${test_src}")
  endforeach()
endif()
