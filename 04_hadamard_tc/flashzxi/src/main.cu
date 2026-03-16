//
// Created by core_dump on 3/14/26.
//
#include "hadacore.hpp"
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(make_shape(Int<4>{}, Int<256>{}),
                              make_stride(Int<256>{}, Int<1>{}));

    auto B = make_layout(Shape<_16, _16>{}, Stride<_16, _1>{});
    auto new_layout = composition(layout, B);
    print_layout(new_layout);   // 直接打印二维“坐标 -> index”表
}
// int main()
// {
//     // hadacore::test_small();
//     printf("\n========================================\n\n");
//     hadacore::test_large();
//     return 0;
// }