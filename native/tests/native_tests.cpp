#include "shadow/model.hpp"
#include <array>
#include <cassert>
#include <iostream>

int main() {
    const auto ids = shadowrt::parse_token_list("2 8 42 9");
    assert((ids == std::vector<std::uint32_t>{2, 8, 42, 9}));
    std::array<std::uint8_t, 64> a{}, b{}; b[0] = 0xff; b[63] = 3;
    assert(shadowrt::popcount_xor(a, b) == 10);
    std::cout << "native tests passed\n";
}
