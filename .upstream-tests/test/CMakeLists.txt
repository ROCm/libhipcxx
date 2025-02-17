# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

macro(pythonize_bool var)
  if (${var})
    set(${var} True)
  else()
    set(${var} False)
  endif()
endmacro()

  
if (LIBCUDACXX_ENABLE_CUDA)
set(LIBCUDACXX_HIGHEST_COMPUTE_ARCH 90)

foreach (COMPUTE_ARCH ${LIBCUDACXX_COMPUTE_ARCHS})
  set(_compute_message "${_compute_message} sm_${COMPUTE_ARCH}")
  set(LIBCUDACXX_COMPUTE_ARCHS_STRING "${LIBCUDACXX_COMPUTE_ARCHS_STRING} ${COMPUTE_ARCH}")
endforeach ()

message(STATUS "Enabled CUDA architectures:${_compute_message}")

option(LIBCUDACXX_TEST_WITH_NVRTC
  "Test libcu++ with runtime compilation instead of offline compilation. Only runs device side tests."
  OFF)
endif()

if (LIBCUDACXX_TEST_WITH_NVRTC)
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    message(FATAL_ERROR "NVRTC testing not supported on Windows. \
      Set LIBCUDACXX_TEST_WITH_NVRTC to OFF.")
  endif()

  set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/../utils/nvidia/nvrtc/nvrtc.sh")
  set(LIBCUDACXX_CUDA_COMPILER_ARG1 "${CMAKE_CUDA_COMPILER}")
else() # NOT LIBCUDACXX_TEST_WITH_NVRTC
  if (LIBCUDACXX_ENABLE_CUDA)
    set(LIBCUDACXX_FORCE_INCLUDE "-include ${CMAKE_CURRENT_SOURCE_DIR}/force_include_cuda.h")
    set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}")
  elseif(LIBCUDACXX_ENABLE_HIP)
    set(LIBCUDACXX_FORCE_INCLUDE "-include ${CMAKE_CURRENT_SOURCE_DIR}/force_include_hip.h")
     #temporary workaround until we've found a way to make cmake find hipcc
    set(LIBCUDACXX_CUDA_COMPILER ${HIP_HIPCC_EXECUTABLE})
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_COMPILER_ID "HIPCC")
  endif()
endif()

if (NOT MSVC)
  set(LIBCUDACXX_WARNING_LEVEL
    "--compiler-options=-Wall \
     --compiler-options=-Wextra")
endif()

# Intel OneAPI compiler has fast math enabled by default which breaks almost all floating point tests
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
  set(LIBCUDACXX_TEST_COMPILER_FLAGS
    "${LIBCUDACXX_TEST_COMPILER_FLAGS} \
    --compiler-options=-fno-fast-math")
endif()

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
  set(LIBCUDACXX_TEST_COMPILER_FLAGS
    "${LIBCUDACXX_TEST_COMPILER_FLAGS} \
    ${LIBCUDACXX_FORCE_INCLUDE} \
    -I${libhipcxx_SOURCE_DIR}/include \
    ${LIBCUDACXX_WARNING_LEVEL}")
endif()

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVHPC")
  set(LIBCUDACXX_TEST_COMPILER_FLAGS
    "${LIBCUDACXX_TEST_COMPILER_FLAGS} \
    -stdpar")
  set(LIBCUDACXX_TEST_LINKER_FLAGS
    "${LIBCUDACXX_TEST_LINKER_FLAGS} \
    -stdpar")
endif()

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "HIPCC")
  set(LIBCUDACXX_TEST_COMPILER_FLAGS
    "${LIBCUDACXX_TEST_COMPILER_FLAGS} \
    ${LIBCUDACXX_FORCE_INCLUDE} \
    -I${CMAKE_SOURCE_DIR}/include"
    CACHE INTERNAL "Flags for libcxx testing." FORCE)
endif()

set(LIBCUDACXX_COMPUTE_ARCHS_STRING
  "${CMAKE_CUDA_ARCHITECTURES}")

include(AddLLVM)

set(LIBCUDACXX_TARGET_INFO "libcudacxx.test.target_info.LocalTI" CACHE STRING
"TargetInfo to use when setting up test environment.")
set(LIBCUDACXX_EXECUTOR "None" CACHE STRING
"Executor to use when running tests.")

# NVRTC needs extra time until we can get compilation seperated.
if (LIBCUDACXX_TEST_WITH_NVRTC)
  set(LIBCUDACXX_TEST_TIMEOUT "0" CACHE STRING "Enable test timeouts (Default = 200, Off = 0)")
else()
  set(LIBCUDACXX_TEST_TIMEOUT "200" CACHE STRING "Enable test timeouts (Default = 200, Off = 0)")
endif()

set(AUTO_GEN_COMMENT "## Autogenerated by libcudacxx configuration.\n# Do not edit!")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

add_lit_testsuite(check-hipcxx
  "Running libhip++ tests"
  ${CMAKE_CURRENT_BINARY_DIR})

# Add test target for standalone headers
add_subdirectory(host_only)
