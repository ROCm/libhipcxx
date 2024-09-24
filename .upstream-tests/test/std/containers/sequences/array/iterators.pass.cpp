//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// <cuda/std/array>

// iterator, const_iterator

#include <hip/std/array>
#include <hip/std/iterator>
#include <hip/std/cassert>

#include "test_macros.h"

// hip::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main(int, char**)
{
    {
    typedef hip::std::array<int, 5> C;
    C c;
    C::iterator i;
    i = c.begin();
    C::const_iterator j;
    j = c.cbegin();
    assert(i == j);
    }
    {
    typedef hip::std::array<int, 0> C;
    C c;
    C::iterator i;
    i = c.begin();
    C::const_iterator j;
    j = c.cbegin();
    assert(i == j);
    }

#if TEST_STD_VER > 11
    { // N3644 testing
        {
        typedef hip::std::array<int, 5> C;
        C::iterator ii1{}, ii2{};
        C::iterator ii4 = ii1;
        C::const_iterator cii{};
        assert ( ii1 == ii2 );
        assert ( ii1 == ii4 );
        assert ( ii1 == cii );

        assert ( !(ii1 != ii2 ));
        assert ( !(ii1 != cii ));

        C c;
        assert ( c.begin()   == hip::std::begin(c));
        assert ( c.cbegin()  == hip::std::cbegin(c));
        assert ( c.rbegin()  == hip::std::rbegin(c));
        assert ( c.crbegin() == hip::std::crbegin(c));
        assert ( c.end()     == hip::std::end(c));
        assert ( c.cend()    == hip::std::cend(c));
        assert ( c.rend()    == hip::std::rend(c));
        assert ( c.crend()   == hip::std::crend(c));

        assert ( hip::std::begin(c)   != hip::std::end(c));
        assert ( hip::std::rbegin(c)  != hip::std::rend(c));
        assert ( hip::std::cbegin(c)  != hip::std::cend(c));
        assert ( hip::std::crbegin(c) != hip::std::crend(c));
        }
        {
        typedef hip::std::array<int, 0> C;
        C::iterator ii1{}, ii2{};
        C::iterator ii4 = ii1;
        C::const_iterator cii{};
        assert ( ii1 == ii2 );
        assert ( ii1 == ii4 );

        assert (!(ii1 != ii2 ));

        assert ( (ii1 == cii ));
        assert ( (cii == ii1 ));
        assert (!(ii1 != cii ));
        assert (!(cii != ii1 ));
        assert (!(ii1 <  cii ));
        assert (!(cii <  ii1 ));
        assert ( (ii1 <= cii ));
        assert ( (cii <= ii1 ));
        assert (!(ii1 >  cii ));
        assert (!(cii >  ii1 ));
        assert ( (ii1 >= cii ));
        assert ( (cii >= ii1 ));
        assert (cii - ii1 == 0);
        assert (ii1 - cii == 0);

        C c;
        assert ( c.begin()   == hip::std::begin(c));
        assert ( c.cbegin()  == hip::std::cbegin(c));
        assert ( c.rbegin()  == hip::std::rbegin(c));
        assert ( c.crbegin() == hip::std::crbegin(c));
        assert ( c.end()     == hip::std::end(c));
        assert ( c.cend()    == hip::std::cend(c));
        assert ( c.rend()    == hip::std::rend(c));
        assert ( c.crend()   == hip::std::crend(c));

        assert ( hip::std::begin(c)   == hip::std::end(c));
        assert ( hip::std::rbegin(c)  == hip::std::rend(c));
        assert ( hip::std::cbegin(c)  == hip::std::cend(c));
        assert ( hip::std::crbegin(c) == hip::std::crend(c));
        }
    }
#endif
#if TEST_STD_VER > 14
    {
        typedef hip::std::array<int, 5> C;
        constexpr C c{0,1,2,3,4};

        static_assert ( c.begin()   == hip::std::begin(c), "");
        static_assert ( c.cbegin()  == hip::std::cbegin(c), "");
        static_assert ( c.end()     == hip::std::end(c), "");
        static_assert ( c.cend()    == hip::std::cend(c), "");

        static_assert ( c.rbegin()  == hip::std::rbegin(c), "");
        static_assert ( c.crbegin() == hip::std::crbegin(c), "");
        static_assert ( c.rend()    == hip::std::rend(c), "");
        static_assert ( c.crend()   == hip::std::crend(c), "");

        static_assert ( hip::std::begin(c)   != hip::std::end(c), "");
        static_assert ( hip::std::rbegin(c)  != hip::std::rend(c), "");
        static_assert ( hip::std::cbegin(c)  != hip::std::cend(c), "");
        static_assert ( hip::std::crbegin(c) != hip::std::crend(c), "");

        static_assert ( *c.begin()  == 0, "");
        static_assert ( *c.rbegin()  == 4, "");

        static_assert ( *hip::std::begin(c)   == 0, "" );
        static_assert ( *hip::std::cbegin(c)  == 0, "" );
        static_assert ( *hip::std::rbegin(c)  == 4, "" );
        static_assert ( *hip::std::crbegin(c) == 4, "" );
    }
#endif

  return 0;
}
