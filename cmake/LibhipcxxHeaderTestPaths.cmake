# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

# The header tests derive one target and one generated source per header, so a
# name such as `headertest_std___algorithm_lexicographical_compare.h` ends up in
# the object path twice. On Windows that blows past CMAKE_OBJECT_PATH_MAX (250)
# before the build directory is even deep, and CMake only shortens the object
# file name, never the directories leading to it.
#
# Replace those names by a hash of the header when Windows is not configured for
# long paths, and keep the readable names everywhere else.

include_guard(GLOBAL)

set(LIBHIPCXX_HASH_HEADERTEST_PATHS "AUTO" CACHE STRING
    "Shorten header test names so object paths fit on Windows: AUTO, ON or OFF")
set_property(CACHE LIBHIPCXX_HASH_HEADERTEST_PATHS PROPERTY STRINGS AUTO ON OFF)

# LongPathsEnabled is the machine-wide opt-in added in Windows 10 1607. It says
# the OS accepts paths beyond MAX_PATH; it does not promise every tool in the
# chain does, hence the manual override.
set(_libhipcxx_long_paths "n/a")
if(WIN32)
  set(_libhipcxx_long_paths "off")
  execute_process(
    COMMAND reg query "HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" /v LongPathsEnabled
    RESULT_VARIABLE _libhipcxx_reg_result
    OUTPUT_VARIABLE _libhipcxx_reg_output
    ERROR_QUIET)
  if(_libhipcxx_reg_result EQUAL 0 AND
     _libhipcxx_reg_output MATCHES "REG_DWORD[ \t]+0x0*1([^0-9a-fA-F]|$)")
    set(_libhipcxx_long_paths "on")
  endif()
endif()

string(TOUPPER "${LIBHIPCXX_HASH_HEADERTEST_PATHS}" _libhipcxx_hash_setting)
if(_libhipcxx_hash_setting STREQUAL "ON")
  set(_libhipcxx_hash_headertest_names TRUE)
elseif(_libhipcxx_hash_setting STREQUAL "OFF")
  set(_libhipcxx_hash_headertest_names FALSE)
elseif(_libhipcxx_hash_setting STREQUAL "AUTO")
  if(WIN32 AND NOT _libhipcxx_long_paths STREQUAL "on")
    set(_libhipcxx_hash_headertest_names TRUE)
  else()
    set(_libhipcxx_hash_headertest_names FALSE)
  endif()
else()
  message(FATAL_ERROR
          "LIBHIPCXX_HASH_HEADERTEST_PATHS must be AUTO, ON or OFF, got "
          "'${LIBHIPCXX_HASH_HEADERTEST_PATHS}'")
endif()

# Readable names stay, so let CMake emit the long object paths the OS accepts.
if(WIN32 AND NOT _libhipcxx_hash_headertest_names)
  set(CMAKE_OBJECT_PATH_MAX 32767)
endif()

if(_libhipcxx_hash_headertest_names)
  message(STATUS "Header test names: hashed (Windows long paths: ${_libhipcxx_long_paths})")
else()
  message(STATUS "Header test names: readable (Windows long paths: ${_libhipcxx_long_paths})")
endif()

# The map lets a build log point back from a hashed name to its header.
set(_libhipcxx_headertest_name_map "${CMAKE_CURRENT_BINARY_DIR}/header_test_names.txt")
if(_libhipcxx_hash_headertest_names)
  file(WRITE "${_libhipcxx_headertest_name_map}" "# hashed name\theader\n")
endif()

# Return the name to use for the target and generated source of a header test:
# either READABLE as passed in, or a hash of HEADER.
function(libhipcxx_headertest_name out_var header readable)
  if(NOT _libhipcxx_hash_headertest_names)
    set(${out_var} "${readable}" PARENT_SCOPE)
    return()
  endif()

  string(SHA1 hash "${header}")
  string(SUBSTRING "${hash}" 0 12 hash)
  file(APPEND "${_libhipcxx_headertest_name_map}" "${hash}\t${header}\n")
  set(${out_var} "${hash}" PARENT_SCOPE)
endfunction()
