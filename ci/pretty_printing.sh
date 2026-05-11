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

# Print "ARG=${ARG}" for all args.
function print_var_values() {
    # Iterate through the arguments
    for var_name in "$@"; do
        if [ -z "$var_name" ]; then
            echo "Usage: print_var_values <variable_name1> <variable_name2> ..."
            return 1
        fi

        # Dereference the variable and print the result
        echo "$var_name=${!var_name:-(undefined)}"
    done
}

# begin_group: Start a named section of log output, possibly with color.
# Usage: begin_group "Group Name" [Color]
#   Group Name: A string specifying the name of the group.
#   Color (optional): ANSI color code to set text color. Default is blue (1;34).
function begin_group() {
    # See options for colors here: https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
    local blue="34"
    local name="${1:-}"
    local color="${2:-$blue}"

    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo -e "::group::\e[${color}m${name}\e[0m"
    else
        echo -e "\e[${color}m================== ${name} ======================\e[0m"
    fi
}

# end_group: End a named section of log output and print status based on exit status.
# Usage: end_group "Group Name" [Exit Status]
#   Group Name: A string specifying the name of the group.
#   Exit Status (optional): The exit status of the command run within the group. Default is 0.
function end_group() {
    local name="${1:-}"
    local build_status="${2:-0}"
    local duration="${3:-}"
    local red="31"
    local blue="34"

    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo "::endgroup::"

        if [ "$build_status" -ne 0 ]; then
            echo -e "::error::\e[${red}m ${name} - Failed (⬆️ click above for full log ⬆️)\e[0m"
        fi
    else
        if [ "$build_status" -ne 0 ]; then
            echo -e "\e[${red}m================== End ${name} - Failed${duration:+ - Duration: ${duration}s} ==================\e[0m"
        else
            echo -e "\e[${blue}m================== End ${name} - Success${duration:+ - Duration: ${duration}s} ==================\n\e[0m"
        fi
    fi
}

declare -A command_durations

# Runs a command within a named group, handles the exit status, and prints appropriate messages based on the result.
# Usage: run_command "Group Name" command [arguments...]
#
# NOTE(HIP/AMD): the command is run with 'BASH_ENV=' overridden in the
# per-command env, AND wrapped in a '(set +u; …)' subshell.  This is a
# manylinux-CI quirk: ROCm-DS Jenkins jobs (e.g. libhipcxx/{159,164,165})
# run inside a manylinux container that exports
# 'BASH_ENV=/env/bash.env' globally, and that file does
# 'set -euo pipefail'. Bash sources '$BASH_ENV' on startup of every
# non-interactive shell, so every child bash invocation -- including
# the bash subscripts that 'git submodule update' runs from CMake's
# FetchContent / CPM -- ends up with '-u' enabled. The bundled git
# wrapper scripts (git-sh-i18n, git-sh-setup, ...) however reference
# unset variables ('$GIT_TEXTDOMAINDIR', '$LONG_USAGE', ...) and abort
# with rc=1 under '-u', killing the configure step. That is what was
# failing the Catch2 (and dlpack) git fetch on the air-gapped CI.
#
# Defeating it requires *both* halves:
#   * 'BASH_ENV='  per-command env override prevents the child bash
#     from re-sourcing /env/bash.env on startup.
#   * '(set +u; ...)' subshell ensures the immediate command and any
#     bash subprocesses it spawns *itself* run with '-u' off, even
#     before the BASH_ENV override has any effect.
# Empirically, BASH_ENV= alone is enough for our case (verified against
# a fake /env/bash.env setting -euo pipefail), but keeping the (set +u)
# wrap is a belt-and-braces measure in case the manylinux env is later
# extended to also export 'SHELLOPTS' or other flag-carrying vars.
#
# This affects ALL run_command invocations -- in practice that is the
# 'cmake --preset', 'cmake --build', and 'ctest --preset' calls in
# 'ci/build_common.sh'. Our own shell logic outside the subshell still
# runs with '-u' on, so we keep the strictness benefit for our own code.
function run_command() {
    local group_name="${1:-}"
    shift
    local command=("$@")
    local status

    begin_group "$group_name"
    echo "Working directory: $(pwd)"
    echo "Running command: ${command[*]}"
    set +e
    local start_time=$(date +%s)
    ( set +u; BASH_ENV= "${command[@]}" )
    status=$?
    local end_time=$(date +%s)
    set -e
    local duration=$((end_time - start_time))
    end_group "$group_name" $status $duration
    command_durations["$group_name"]=$duration
    return $status
}

function string_width() {
    local str="$1"
    echo "$str" | awk '{print length}'
}

function print_time_summary() {
    local max_length=0
    local group

    # Find the longest group name for formatting
    for group in "${!command_durations[@]}"; do
        local group_length=$(echo "$group" | awk '{print length}')
        if [ "$group_length" -gt "$max_length" ]; then
            max_length=$group_length
        fi
    done

    if [ "$max_length" -eq 0 ]; then
        return
    fi

    echo "Time Summary:"
    for group in "${!command_durations[@]}"; do
        printf "%-${max_length}s : %s seconds\n" "$group" "${command_durations[$group]}"
    done

    # Clear the array of timing info
    declare -gA command_durations=()
}
