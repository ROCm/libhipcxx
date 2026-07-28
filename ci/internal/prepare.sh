#!/usr/bin/env bash
# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Install the OS-level packages the libhipcxx CI build needs on top of a base
# image. Only the pipeline calls this; the Docker builder images bake the same
# dependencies into their Dockerfiles.

set -xeu

os_id=$(. /etc/os-release; echo $ID)
os_id_like=$(. /etc/os-release; echo $ID_LIKE)

# note: below may only be required if no conda env is used
if [[ "${os_id}" == *"ubuntu"* ]]; then
  apt install -y --no-install-recommends \
    python3-pip python3-venv python3-dev
elif [[ "${os_id_like}" == *"rhel"* ]]; then
  if [[ "${AUDITWHEEL_PLAT:-}" == "manylinux"* ]]; then
    # note: dev headers already avail
    # note: no venv package in manylinux
    pip install --upgrade pip
    pip install virtualenv
  fi
fi
