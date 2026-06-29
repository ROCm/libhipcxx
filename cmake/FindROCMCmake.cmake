# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

# ###########################
# libhipcxx dependencies
# ###########################

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)

# By default, rocm software stack is expected at /opt/rocm
# set environment variable ROCM_PATH to change location
if(NOT ROCM_PATH)
  set(ROCM_PATH /opt/rocm)
endif()

# Security fix (SEC-00337, SEC-00392): Use secure auto-download with pinned
# commit SHA and hash verification instead of mutable 'master' branch.
#
# The previous implementation downloaded and executed unverified code from a mutable
# Git ref, enabling build-time RCE via:
# - Compromised upstream repository
# - MITM attacks (no TLS_VERIFY enforcement)
# - Tag/branch retargeting
#
# New approach (based on rocthrust):
# - Checks for pre-installed rocm-cmake first
# - Falls back to FetchContent with pinned commit SHA + URL hash verification
# - Can be disabled with -DDEPENDENCIES_FORCE_DOWNLOAD=OFF

if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
  # Try new package name first (ROCm 6.4+), fall back to old name
  find_package(ROCmCMakeBuildTools 0.7.3 CONFIG QUIET PATHS ${ROCM_PATH} /opt/rocm)
  if(NOT ROCmCMakeBuildTools_FOUND)
    find_package(ROCM 0.7.3 CONFIG QUIET PATHS ${ROCM_PATH} /opt/rocm)
    set(ROCmCMakeBuildTools_FOUND ${ROCM_FOUND})
  endif()
endif()

if(NOT ROCmCMakeBuildTools_FOUND)
  message(STATUS "ROCm CMake not found. Fetching from pinned release rocm-7.2.4...")

  # We don't want to consume the build and test targets of ROCm CMake.
  # CMake 3.18+ allows omitting them even though there's a CMakeLists.txt in source root.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set(SOURCE_SUBDIR_ARG SOURCE_SUBDIR "DISABLE ADDING TO BUILD")
  else()
    set(SOURCE_SUBDIR_ARG)
  endif()

  include(FetchContent)
  FetchContent_Declare(
    rocm-cmake
    # Use URL with hash verification for maximum security
    URL https://github.com/ROCm/rocm-cmake/archive/refs/tags/rocm-7.2.4.tar.gz
    URL_HASH SHA256=e7a28cb4baf8afbc21204d37e132dae7e12b2d980a2600948fe35cc4d8ac8087
    # Backup: Git with immutable commit SHA (comment out URL lines to use this)
    # GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
    # GIT_TAG        5f38353474569d57c23732f61b5371505a0ba790  # rocm-7.2.4 commit SHA
    # GIT_SHALLOW    TRUE
    ${SOURCE_SUBDIR_ARG}
  )
  FetchContent_MakeAvailable(rocm-cmake)
  find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
else()
  # Already found via pre-installed version
  if(NOT ROCmCMakeBuildTools_FOUND)
    find_package(ROCM 0.7.3 CONFIG REQUIRED PATHS ${ROCM_PATH} /opt/rocm)
  endif()
endif()
