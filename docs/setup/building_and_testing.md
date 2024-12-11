<!-- MIT License
  -- 
  -- Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
  -- 
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  -- 
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  -- 
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
-->

# Building & Testing libhipcxx

## *nix Systems, Native Build/Test

The procedure is demonstrated for hipcc + GCC in C++11 mode on a Debian-like
Linux systems; the same basic steps are required on all other platforms.

### Step 0: Install Build Requirements

In a Bash shell:

```bash
# Install LLVM (needed for LLVM's CMake modules)
apt-get -y install llvm

# Install CMake
apt-get -y install cmake

# Install the LLVM Integrated Tester (`lit`)
apt-get -y install python-pip
pip install lit

# Env vars that should be set, or kept in mind for use later
export LIBHIPCXX_ROOT=/path/to/libhipcxx # Git repo root.
```

### Step 1: Generate the Build Files

In a Bash shell:

```bash
cd ${LIBHIPCXX_ROOT}
cmake \
    -S ./ \
    -B build \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_HIP_COMPILER=hipcc \
    -DLIBHIPCXX_ENABLE_LIBHIPCXX_TESTS=ON \
    -DLIBHIPCXX_ENABLE_LIBCXX_TESTS=OFF
```

### Step 2: Build & Run the Tests

In a Bash shell:

```bash
cd ${LIBHIPCXX_ROOT}/build # build directory of this repo
../utils/amd/linux/perform_tests.bash --skip-libcxx-tests
```
