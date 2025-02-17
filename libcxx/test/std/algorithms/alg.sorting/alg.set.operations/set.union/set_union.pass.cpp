//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter1, InputIterator InIter2, typename OutIter>
//   requires OutputIterator<OutIter, InIter1::reference>
//         && OutputIterator<OutIter, InIter2::reference>
//         && HasLess<InIter2::value_type, InIter1::value_type>
//         && HasLess<InIter1::value_type, InIter2::value_type>
//   OutIter
//   set_union(InIter1 first1, InIter1 last1,
//             InIter2 first2, InIter2 last2, OutIter result);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter1, class Iter2, class OutIter>
void
test()
{
    int ia[] = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    const int sa = sizeof(ia)/sizeof(ia[0]);
    int ib[] = {2, 4, 4, 6};
    const int sb = sizeof(ib)/sizeof(ib[0]);
    int ic[20];
    int ir[] = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 6};
    const int sr = sizeof(ir)/sizeof(ir[0]);
    OutIter ce = std::set_union(Iter1(ia), Iter1(ia+sa),
                                Iter2(ib), Iter2(ib+sb), OutIter(ic));
    assert(base(ce) - ic == sr);
    assert(std::lexicographical_compare(ic, base(ce), ir, ir+sr) == 0);
    ce = std::set_union(Iter1(ib), Iter1(ib+sb),
                        Iter2(ia), Iter2(ia+sa), OutIter(ic));
    assert(base(ce) - ic == sr);
    assert(std::lexicographical_compare(ic, base(ce), ir, ir+sr) == 0);
}

int main(int, char**)
{
    test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, forward_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, random_access_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, int*>();

    test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, int*>();

    test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, int*>();

    test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, int*>();

    test<cpp17_input_iterator<const int*>, const int*, cpp17_output_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, const int*, forward_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, const int*, bidirectional_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, const int*, random_access_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, const int*, int*>();

    test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, forward_iterator<int*> >();
    test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, random_access_iterator<int*> >();
    test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, int*>();

    test<forward_iterator<const int*>, forward_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<forward_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*> >();
    test<forward_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test<forward_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*> >();
    test<forward_iterator<const int*>, forward_iterator<const int*>, int*>();

    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, int*>();

    test<forward_iterator<const int*>, random_access_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<forward_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*> >();
    test<forward_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test<forward_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*> >();
    test<forward_iterator<const int*>, random_access_iterator<const int*>, int*>();

    test<forward_iterator<const int*>, const int*, cpp17_output_iterator<int*> >();
    test<forward_iterator<const int*>, const int*, forward_iterator<int*> >();
    test<forward_iterator<const int*>, const int*, bidirectional_iterator<int*> >();
    test<forward_iterator<const int*>, const int*, random_access_iterator<int*> >();
    test<forward_iterator<const int*>, const int*, int*>();

    test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, random_access_iterator<int*> >();
    test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, int*>();

    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, int*>();

    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, int*>();

    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, int*>();

    test<bidirectional_iterator<const int*>, const int*, cpp17_output_iterator<int*> >();
    test<bidirectional_iterator<const int*>, const int*, forward_iterator<int*> >();
    test<bidirectional_iterator<const int*>, const int*, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<const int*>, const int*, random_access_iterator<int*> >();
    test<bidirectional_iterator<const int*>, const int*, int*>();

    test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, random_access_iterator<int*> >();
    test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, int*>();

    test<random_access_iterator<const int*>, forward_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<random_access_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*> >();
    test<random_access_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test<random_access_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*> >();
    test<random_access_iterator<const int*>, forward_iterator<const int*>, int*>();

    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, int*>();

    test<random_access_iterator<const int*>, random_access_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<const int*>, int*>();

    test<random_access_iterator<const int*>, const int*, cpp17_output_iterator<int*> >();
    test<random_access_iterator<const int*>, const int*, forward_iterator<int*> >();
    test<random_access_iterator<const int*>, const int*, bidirectional_iterator<int*> >();
    test<random_access_iterator<const int*>, const int*, random_access_iterator<int*> >();
    test<random_access_iterator<const int*>, const int*, int*>();

    test<const int*, cpp17_input_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<const int*, cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<const int*, cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<const int*, cpp17_input_iterator<const int*>, random_access_iterator<int*> >();
    test<const int*, cpp17_input_iterator<const int*>, int*>();

    test<const int*, forward_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<const int*, forward_iterator<const int*>, forward_iterator<int*> >();
    test<const int*, forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test<const int*, forward_iterator<const int*>, random_access_iterator<int*> >();
    test<const int*, forward_iterator<const int*>, int*>();

    test<const int*, bidirectional_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<const int*, bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test<const int*, bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test<const int*, bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test<const int*, bidirectional_iterator<const int*>, int*>();

    test<const int*, random_access_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<const int*, random_access_iterator<const int*>, forward_iterator<int*> >();
    test<const int*, random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test<const int*, random_access_iterator<const int*>, random_access_iterator<int*> >();
    test<const int*, random_access_iterator<const int*>, int*>();

    test<const int*, const int*, cpp17_output_iterator<int*> >();
    test<const int*, const int*, forward_iterator<int*> >();
    test<const int*, const int*, bidirectional_iterator<int*> >();
    test<const int*, const int*, random_access_iterator<int*> >();
    test<const int*, const int*, int*>();

  return 0;
}
