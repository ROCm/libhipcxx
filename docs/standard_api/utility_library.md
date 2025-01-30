<!-- MIT License
  -- 
  -- Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

## Utility Library

Any Standard C++ header not listed below is omitted.

*: Some of the Standard C++ facilities in this header are omitted, see the
libhipcxx Specifics for details.

| API    | Description |
| -------------------------------------------------------------------------- | ---------------------------------- |
| [`<cuda/std/type_traits>`] | Compile-time type introspection.                                                                                                |
| [`<cuda/std/tuple>`]*      | Fixed-sized heterogeneous container (see also: [libhipcxx Specifics](./utility_library/tuple.md)).          |
| [`<cuda/std/functional>`]* | Function objects and function wrappers (see also: [libhipcxx Specifics](./utility_library/functional.md)).  |
| [`<cuda/std/utility>`]*    | Various utility components (see also: [libhipcxx Specifics](./utility_library/utility.md)).                 |
| [`<cuda/std/version>`]     | Compile-time version information (see also: [libhipcxx Specifics](./utility_library/version.md)).           |


[`<cuda/std/type_traits>`]: https://en.cppreference.com/w/cpp/header/type_traits
[`<cuda/std/tuple>`]: https://en.cppreference.com/w/cpp/header/tuple
[`<cuda/std/functional>`]: https://en.cppreference.com/w/cpp/header/functional
[`<cuda/std/utility>`]: https://en.cppreference.com/w/cpp/header/utility
[`<cuda/std/version>`]: https://en.cppreference.com/w/cpp/header/version
