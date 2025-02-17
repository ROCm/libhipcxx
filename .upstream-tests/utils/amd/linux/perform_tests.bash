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

#! /bin/bash

function usage {
  echo "Usage: ${0} [flags...] <tests...>|all"
  echo
  echo "Run <tests> from the libc++ and libhip++ test suites."
  echo "If no tests or \"all\" is specified, all tests are run."
  echo
  echo "-h, -help, --help                : Print this message."
  echo "--dry-run                        : Show what commands would be invoked;"
  echo "                                 : don't actually execute anything."
  echo "--skip-base-tests-build          : Do not build (or run) any tests."
  echo "                                 : Overrides \${LIBCUDACXX_SKIP_BASE_TESTS_BUILD}."
  echo "--skip-tests-runs                : Build tests but do not run them."
  echo "                                 : Overrides \${LIBCUDACXX_SKIP_TESTS_RUN}."
  echo "--skip-libcxx-tests              : Do not build (or run) any libc++ tests."
  echo "                                 : Overrides \${LIBCUDACXX_SKIP_LIBCXX_TESTS}."
  echo "--skip-libhipcxx-tests          : Do not build (or run) any libhip++ tests."
  echo "                                 : Overrides \${LIBCUDACXX_SKIP_LIBCUDACXX_TESTS}."
  echo "--skip-arch-detection            : Do not automatically detect the CDNA architecture"
  echo "                                 : for tests runs."
  echo "                                 : Overrides \${LIBCUDACXX_SKIP_ARCH_DETECTION}."
  echo
  echo "--compute-archs                  : Space-separated list of CDNA architectures"
  echo "                                   (specified as string) to target. If empty,"
  echo "                                   either the architecture is automatically"
  echo "                                   detected (for tests runs if detection isn't"
  echo "                                   disabled) or all known SM architectures are"
  echo "                                   targeted."
  echo "                                 : Overrides \${LIBCUDACXX_COMPUTE_ARCHS}."
  echo
  echo "--libcxx-lit-site-config <file>     : Use <file> as the libc++ lit site config"
  echo "                                    : (default: \${LIBCUDACXX_PATH}/libcxx/build/test/lit.site.cfg)."
  echo "--libhipcxx-lit-site-config <file> : Use <file> as the libhip++ lit site config"
  echo "                                    : (default: \${LIBCUDACXX_PATH}/build/libcxx/test/lit.site.cfg)."
  echo
  echo "--verbose                           : Print SM architecture detection and test results"
  echo "                                    : to stdout in addition to log files."
  echo "--pretty                           : Print each test result individually."
  echo
  echo "\${LIBCUDACXX_SKIP_BASE_TESTS_BUILD}   : If set and non-zero, do not build"
  echo "                                      : (or run) any tests."
  echo "\${LIBCUDACXX_SKIP_TESTS_RUN}          : If set and non-zero, build tests"
  echo "                                      : but do not run them."
  echo "\${LIBCUDACXX_SKIP_LIBCXX_TESTS}       : If set and non-zero, do not build"
  echo "                                      : (or run) any libc++ tests."
  echo "\${LIBCUDACXX_SKIP_LIBCUDACXX_TESTS}   : If set and non-zero, do not build"
  echo "                                      : (or run) any libhip++ tests."
  echo "\${LIBCUDACXX_SKIP_ARCH_DETECTION}     : If set, non-zero, and"
  echo "                                      : \${LIBCUDACXX_COMPUTE_ARCHS} is unset,"
  echo "                                      : do not automatically detect the SM"
  echo "                                      : architecture."
  echo "\${LIBCUDACXX_COMPUTE_ARCHS}           : A space-separated list of SM"
  echo "                                      : architectures (specified as integers)"
  echo "                                      : to target. If empty, either the"
  echo "                                      : architecture is automatically"
  echo "                                      : detected (for tests runs if detection"
  echo "                                      : isn't disabled) or all known SM"
  echo "                                      : architectures are targeted."

  exit -3
}

function section_separator {
  for i in {0..79}
  do
    echo -n "#"
  done
  echo
}

LIBCXX_LOG=$(mktemp)
LIBCUDACXX_LOG=$(mktemp)

KNOWN_COMPUTE_ARCHS="gfx940 gfx941 gfx942 gfx90a gfx908 gfx1100"

function report_and_exit {
  # If any of the lines searched for below aren't present in the log files, the
  # grep commands will return nothing, and the variables will be empty. Bash
  # treats empty variables as zero for the purposes of arithmetic, which is what
  # we want anyways, so we don't need to do anything else.

  # Example parse target:
  # |Testing Time: 19.78s
  # |  Unsupported        : 1
  # |  Passed             : 9
  # |  Failed             : 2
  # |  Unexpectedly Passed: 1

  if [ -e "${LIBCXX_LOG}" ]
  then
    # White space is dynamic, capture first space on output lines to prevent
    # grabbing lines like `Unexpectedly Passed Tests (1)`
    LIBCXX_UNSUPPORTED_TESTS=$(  egrep '^ \s*Unsupported'         ${LIBCXX_LOG} | sed 's/^\s*Unsupported\s*:\s*\([0-9]\+\)/\1/')
    LIBCXX_EXPECTED_PASSES=$(    egrep '^ \s*Passed'              ${LIBCXX_LOG} | sed 's/^\s*Passed\s*:\s*\([0-9]\+\)/\1/')
    LIBCXX_UNEXPECTED_FAILURES=$(egrep '^ \s*Failed'              ${LIBCXX_LOG} | sed 's/^\s*Failed\s*:\s*\([0-9]\+\)/\1/')
    LIBCXX_UNEXPECTED_PASSES=$(  egrep '^ \s*Unexpectedly Passed' ${LIBCXX_LOG} | sed 's/^\s*Unexpectedly Passed\s*:\s*\([0-9]\+\)/\1/')
  fi

  if [ -e "${LIBCUDACXX_LOG}" ]
  then
    # White space is dynamic, capture first space on output lines to prevent
    # grabbing lines like `Unexpectedly Passed Tests (1)`
    LIBCUDACXX_UNSUPPORTED_TESTS=$(  egrep '^ \s*Unsupported'         ${LIBCUDACXX_LOG} | sed 's/^\s*Unsupported\s*:\s*\([0-9]\+\)/\1/')
    LIBCUDACXX_EXPECTED_PASSES=$(    egrep '^ \s*Passed'              ${LIBCUDACXX_LOG} | sed 's/^\s*Passed\s*:\s*\([0-9]\+\)/\1/')
    LIBCUDACXX_UNEXPECTED_FAILURES=$(egrep '^ \s*Failed'              ${LIBCUDACXX_LOG} | sed 's/^\s*Failed\s*:\s*\([0-9]\+\)/\1/')
    LIBCUDACXX_UNEXPECTED_PASSES=$(  egrep '^ \s*Unexpectedly Passed' ${LIBCUDACXX_LOG} | sed 's/^\s*Unexpectedly Passed\s*:\s*\([0-9]\+\)/\1/')
  fi

  LIBCXX_PASSES=$((  LIBCXX_EXPECTED_PASSES))
  LIBCXX_FAILURES=$((LIBCXX_UNEXPECTED_PASSES + LIBCXX_UNEXPECTED_FAILURES))
  LIBCXX_TOTAL=$((   LIBCXX_PASSES            + LIBCXX_FAILURES))

  LIBCUDACXX_PASSES=$((  LIBCUDACXX_EXPECTED_PASSES))
  LIBCUDACXX_FAILURES=$((LIBCUDACXX_UNEXPECTED_PASSES + LIBCUDACXX_UNEXPECTED_FAILURES))
  LIBCUDACXX_TOTAL=$((   LIBCUDACXX_PASSES            + LIBCUDACXX_FAILURES))

  OVERALL_PASSES=$((  LIBCXX_PASSES   + LIBCUDACXX_PASSES))
  OVERALL_FAILURES=$((LIBCXX_FAILURES + LIBCUDACXX_FAILURES))
  OVERALL_TOTAL=$((   LIBCXX_TOTAL    + LIBCUDACXX_TOTAL))

  section_separator

  if [ "${OVERALL_TOTAL:-0}" != "0" ]
  then
    printf "Score: %.2f%%\n" "$((10000 * ${OVERALL_PASSES} / ${OVERALL_TOTAL}))e-2"
  else
    echo "Score: 0.00%"
  fi

  if (( 1 <= ${#} ))
  then
    exit ${1}
  else
    if (( ${OVERALL_PASSES} == ${OVERALL_TOTAL} ))
    then
      exit 0
    else
      exit -1
    fi
  fi
}

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

LIBCUDACXX_PATH=$(realpath ${SCRIPT_PATH}/../../../../)

################################################################################
# Command Line Processing.

LIT_PREFIX="time"

LIBCXX_LIT_SITE_CONFIG=${LIBCUDACXX_PATH}/libcxx/build/libcxx/test/lit.site.cfg
LIBCUDACXX_LIT_SITE_CONFIG=${LIBCUDACXX_PATH}/build/test/lit.site.cfg

RAW_TEST_TARGETS=""

while test ${#} != 0
do
  case "${1}" in
  -h) usage ;;
  -help) usage ;;
  --help) usage ;;
  --dry-run) LIT_PREFIX="echo" ;;
  --skip-base-tests-build)    LIBCUDACXX_SKIP_BASE_TESTS_BUILD=1 ;;
  --skip-tests-runs)          LIBCUDACXX_SKIP_TESTS_RUN=1 ;;
  --skip-libcxx-tests)        LIBCUDACXX_SKIP_LIBCXX_TESTS=1 ;;
  --skip-libhipcxx-tests)    LIBCUDACXX_SKIP_LIBCUDACXX_TESTS=1 ;;
  --skip-arch-detection)      LIBCUDACXX_SKIP_ARCH_DETECTION=1 ;;
  --compute-archs)
    shift # The next argument is the list of archs.
    LIBCUDACXX_COMPUTE_ARCHS=${1}
    ;;
  --libcxx-lit-site-config)
    shift # The next argument is the file.
    LIBCXX_LIT_SITE_CONFIG=${1}
    ;;
  --libhipcxx-lit-site-config)
    shift # The next argument is the file.
    LIBCUDACXX_LIT_SITE_CONFIG=${1}
    ;;
  --verbose) VERBOSE=1 ;;
  --pretty) PRETTY=1 ;;
  *)
    RAW_TEST_TARGETS="${RAW_TEST_TARGETS:+${RAW_TEST_TARGETS} }${1}"
    ;;
  esac
  shift
done

LIBCXX_TEST_TARGETS="${LIBCUDACXX_PATH}/libcxx/test"
LIBCUDACXX_TEST_TARGETS="${LIBCUDACXX_PATH}/test"

if [ "${RAW_TEST_TARGETS:-all}" != "all" ]
then
  LIBCXX_TEST_TARGETS=""
  LIBCUDACXX_TEST_TARGETS=""
  for test in ${RAW_TEST_TARGETS}
  do
    LIBCXX_TEST_TARGETS="${LIBCXX_TEST_TARGETS:+${LIBCXX_TEST_TARGETS} }${LIBCUDACXX_PATH}/libcxx/test/${test}"
    LIBCUDACXX_TEST_TARGETS="${LIBCUDACXX_TEST_TARGETS:+${LIBCUDACXX_TEST_TARGETS} }${LIBCUDACXX_PATH}/test/${test}"
  done
fi

################################################################################
# Variable Processing

if [ "${LIBCUDACXX_SKIP_BASE_TESTS_BUILD:-0}" != "0" ] || \
   [[ "${LIBCUDACXX_SKIP_LIBCXX_TESTS:-0}" != "0" && "${LIBCUDACXX_SKIP_LIBCUDACXX_TESTS:-0}" != "0" ]]
then
  echo "# TEST libc++  : Skipped"
  echo "# TEST libhip++ : Skipped"
  section_separator
  echo "Score: 100.00%"
  exit 0
fi

if [ "${VERBOSE:-0}" != "0" ]
then
  LIT_FLAGS="-vv -a"
elif [ "${PRETTY:-0}" != "0" ]
then
  LIT_FLAGS="--show-pass --show-skipped"
else
  LIT_FLAGS="-sv --no-progress-bar"
fi

if [ "${LIBCUDACXX_SKIP_TESTS_RUN:-0}" != "0" ]
then
  LIT_FLAGS="${LIT_FLAGS:+${LIT_FLAGS} }-Dexecutor=\"NoopExecutor()\""
fi

JSON_OUTPUT_TARGET="0"

if [ "${JSON_OUTPUT_PATH:-0}" != "0" ]
then
  JSON_OUTPUT_TARGET="${JSON_OUTPUT_PATH}/testlog_$(basename $(mktemp))"
  mkdir -p "${JSON_OUTPUT_TARGET}"
fi

################################################################################
# CDNA Architecture Detection

if [ "${LIBCUDACXX_SKIP_ARCH_DETECTION:-0}" == "0" ] && \
   [ "${LIBCUDACXX_SKIP_TESTS_RUN:-0}" == "0" ] && \
   [ ! -n "${LIBCUDACXX_COMPUTE_ARCHS}" ]
then
  section_separator

  echo "# TEST CDNA Architecture Detection"

  ARCH_DETECTION_LOG=$(mktemp)
  DETECTION_LIT_FLAGS="-vv -a"
  if [ "${JSON_OUTPUT_TARGET}" != "0" ]
  then
    DETECTION_LIT_FLAGS="${DETECTION_LIT_FLAGS} -o ${JSON_OUTPUT_TARGET}/detect_gfx.log"
  fi


  LIBCUDACXX_SITE_CONFIG=${LIBCUDACXX_LIT_SITE_CONFIG} \
  bash -c "lit ${DETECTION_LIT_FLAGS} ${LIBCUDACXX_PATH}/test/nothing_to_do.pass.cpp -Dcompute_archs=\"${KNOWN_COMPUTE_ARCHS}\"" \
    > ${ARCH_DETECTION_LOG} 2>&1

  if [ "${PIPESTATUS[0]}" != "0" ]
  then
    cat ${ARCH_DETECTION_LOG}
    report_and_exit 1
  fi

  DEVICE_0_COMPUTE_ARCH=$(egrep '^Device 0:' ${ARCH_DETECTION_LOG} | sed 's/^Device 0: ".*", Selected, CDNA \(gfx[0-9a-f]\{3\}\).*/\1/')

  rm -f ${ARCH_DETECTION_LOG}

  echo "# DETECTION CDNA Architecture : Device 0, ${DEVICE_0_COMPUTE_ARCH}"
  LIBCUDACXX_COMPUTE_ARCHS=${DEVICE_0_COMPUTE_ARCH}
fi

if [ -n "${LIBCUDACXX_COMPUTE_ARCHS}" ]
then
  LIT_COMPUTE_ARCHS_FLAG="-Dcompute_archs=\""
  LIT_COMPUTE_ARCHS_SUFFIX="\""
fi

################################################################################
# Dump Variables

VARIABLES="
  PATH
  PWD
  SCRIPT_PATH
  LIBCUDACXX_PATH
  VERBOSE
  PRETTY
  LIBCUDACXX_SKIP_BASE_TESTS_BUILD
  LIBCUDACXX_SKIP_TESTS_RUN
  LIBCUDACXX_SKIP_LIBCXX_TESTS
  LIBCUDACXX_SKIP_LIBCUDACXX_TESTS
  LIBCUDACXX_SKIP_ARCH_DETECTION
  LIBCXX_LIT_SITE_CONFIG
  LIBCUDACXX_LIT_SITE_CONFIG
  LIBCXX_LOG
  LIBCUDACXX_LOG
  LIBCXX_TEST_TARGETS
  LIBCUDACXX_TEST_TARGETS
  LIT_COMPUTE_ARCHS_FLAG
  LIT_COMPUTE_ARCHS_SUFFIX
  LIT_FLAGS
  LIT_PREFIX
  RAW_TEST_TARGETS
  LIBCUDACXX_COMPUTE_ARCHS
  DEVICE_0_COMPUTE_ARCH
  JSON_OUTPUT_TARGET
"

section_separator

for VARIABLE in ${VARIABLES}
do
  printf "# VARIABLE %s%q\n" "${VARIABLE}=" "${!VARIABLE}"
done

################################################################################
# Build/Run libc++ & libhip++ Tests

section_separator

OUTPUT_STREAM_FLAG=""

if [ "${JSON_OUTPUT_TARGET}" != "0" ]
then
  LIT_FLAGS="${LIT_FLAGS} -o ${JSON_OUTPUT_TARGET}/test_results.log"
  OUTPUT_STREAM_FLAG="> /dev/null"
fi

if [ "${LIBCUDACXX_SKIP_LIBCXX_TESTS:-0}" == "0" ]
then
  echo "# TEST libc++"
  TIMEFORMAT="# WALLTIME libc++ : %R [sec]" \
  LIBCXX_SITE_CONFIG=${LIBCXX_LIT_SITE_CONFIG} \
  bash -c "${LIT_PREFIX} lit ${LIT_FLAGS} ${LIBCXX_TEST_TARGETS} ${OUTPUT_STREAM_FLAG}" \
  2>&1 | tee "${LIBCXX_LOG}"
  if [ "${PIPESTATUS[0]}" != "0" ]; then report_and_exit 1; fi
else
  echo "# TEST libc++ : Skipped"
fi

section_separator

if [ "${LIBCUDACXX_SKIP_LIBCUDACXX_TESTS:-0}" == "0" ]
then
  par=$(nproc)
  if [[ $? -eq 0 ]]
  then
    if [[ $par -gt 8 ]]
    then
      par=8
    fi
    LIT_FLAGS="${LIT_FLAGS} -j${par}"
  fi

  echo "# TEST libhip++"
  TIMEFORMAT="# WALLTIME libhip++: %R [sec]" \
  LIBCUDACXX_SITE_CONFIG=${LIBCUDACXX_LIT_SITE_CONFIG} \
  bash -c "${LIT_PREFIX} lit ${LIT_FLAGS} ${LIT_COMPUTE_ARCHS_FLAG}${LIBCUDACXX_COMPUTE_ARCHS}${LIT_COMPUTE_ARCHS_SUFFIX} ${LIBCUDACXX_TEST_TARGETS} ${OUTPUT_STREAM_FLAG}" \
  2>&1 | tee "${LIBCUDACXX_LOG}"
  if [ "${PIPESTATUS[0]}" != "0" ]; then report_and_exit 1; fi
else
  echo "# TEST libhip++ : Skipped"
fi

################################################################################

report_and_exit

