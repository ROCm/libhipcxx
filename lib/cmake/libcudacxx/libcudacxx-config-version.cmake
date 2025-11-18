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

# Parse version information from version header:
include("${CMAKE_CURRENT_LIST_DIR}/libcudacxx-header-search.cmake")

set(libcudacxx_VERSION_MAJOR 3)
set(libcudacxx_VERSION_MINOR 3)
set(libcudacxx_VERSION_PATCH 0)
set(libcudacxx_VERSION_TWEAK 0)

set(
  libcudacxx_VERSION
  "${libcudacxx_VERSION_MAJOR}.${libcudacxx_VERSION_MINOR}.${libcudacxx_VERSION_PATCH}.${libcudacxx_VERSION_TWEAK}"
)

set(PACKAGE_VERSION ${libcudacxx_VERSION})
set(PACKAGE_VERSION_COMPATIBLE FALSE)
set(PACKAGE_VERSION_EXACT FALSE)
set(PACKAGE_VERSION_UNSUITABLE FALSE)

if (PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION)
  # Semantic version check:
  if (libcudacxx_VERSION_MAJOR VERSION_EQUAL PACKAGE_FIND_VERSION_MAJOR)
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  endif()

  # Special case: Major version 1->2 was bumped to sync with other CCCL
  # libraries. There was no break, and requests for 1 are compatible with 2.
  if (
    PACKAGE_FIND_VERSION VERSION_EQUAL 1
    AND libcudacxx_VERSION_MAJOR VERSION_EQUAL 2
  )
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  endif()

  if (PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
