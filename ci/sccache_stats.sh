#!/bin/bash

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

# This script prints the sccache hit rate between two calls to sccache --show-stats.
# It should be sourced in your script before and after the operations you want to profile,
# with the 'start' or 'end' argument respectively.
#
# Callers source this file around the lit and hipRTC suites. `exit` would abort
# that caller, so a missing sccache must return from this file only. Fall back
# to `exit` when the file is executed directly.

mode=$1

if [[ "$mode" != "start" && "$mode" != "end" ]]; then
    echo "Invalid mode: $mode"
    echo "Usage: $0 {start|end}"
    return 1 2>/dev/null || exit 1
fi

# Check if sccache is available
if ! command -v sccache &> /dev/null; then
    echo "Notice: sccache is not available. Skipping..."
    return 0 2>/dev/null || exit 0
fi

case $mode in
  start)
    export SCCACHE_START_HITS=$(sccache --show-stats | awk '/^[ \t]*Cache hits[ \t]+[0-9]+/ {print $3}')
    export SCCACHE_START_MISSES=$(sccache --show-stats | awk '/^[ \t]*Cache misses[ \t]+[0-9]+/ {print $3}')
    ;;
  end)
    if [[ -z ${SCCACHE_START_HITS+x} || -z ${SCCACHE_START_MISSES+x} ]]; then
        echo "Notice: sccache start stats were not collected. Skipping..."
        return 0 2>/dev/null || exit 0
    fi

    final_hits=$(sccache --show-stats | awk '/^[ \t]*Cache hits[ \t]+[0-9]+/ {print $3}')
    final_misses=$(sccache --show-stats | awk '/^[ \t]*Cache misses[ \t]+[0-9]+/ {print $3}')
    hits=$((final_hits - SCCACHE_START_HITS))
    misses=$((final_misses - SCCACHE_START_MISSES))
    total=$((hits + misses))

    if (( total > 0 )); then
      hit_rate=$(awk -v hits="$hits" -v total="$total" 'BEGIN { printf "%.2f", (hits / total) * 100 }')
      echo "sccache hits: $hits | misses: $misses | hit rate: $hit_rate%"
    else
      echo "sccache stats: N/A No new compilation requests"
    fi
    unset SCCACHE_START_HITS
    unset SCCACHE_START_MISSES
    ;;
esac
