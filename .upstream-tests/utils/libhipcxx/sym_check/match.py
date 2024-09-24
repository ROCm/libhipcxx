# -*- Python -*- vim: set syntax=python tabstop=4 expandtab cc=80:
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
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

"""
match - A set of functions for matching symbols in a list to a list of regexs
"""

import re


def find_and_report_matching(symbol_list, regex_list):
    report = ''
    found_count = 0
    for regex_str in regex_list:
        report += 'Matching regex "%s":\n' % regex_str
        matching_list = find_matching_symbols(symbol_list, regex_str)
        if not matching_list:
            report += '    No matches found\n\n'
            continue
        # else
        found_count += len(matching_list)
        for m in matching_list:
            report += '    MATCHES: %s\n' % m['name']
        report += '\n'
    return found_count, report


def find_matching_symbols(symbol_list, regex_str):
    regex = re.compile(regex_str)
    matching_list = []
    for s in symbol_list:
        if regex.match(s['name']):
            matching_list += [s]
    return matching_list
