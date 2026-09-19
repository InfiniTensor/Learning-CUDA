#include "row_policy.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace hadamard;
    unsigned checks = 0;
    auto expect = [&](RowChoice actual, bool packed, int threads) {
        ++checks;
        if (actual.packed != packed || actual.threads != threads)
            throw std::runtime_error("row layout policy regression");
    };
    const auto model = row_device("NVIDIA GeForce RTX 4090 D");
    if (model != RowDevice::RTX4090D ||
        row_device("NVIDIA GeForce RTX 4090") != RowDevice::RTX4090 ||
        row_device("unverified 4090 D") != RowDevice::Unknown)
        throw std::runtime_error("device identities must match exactly");
    for (int threads : {128, 256}) {
        for (bool fused : {false, true}) {
            for (int dim : {1, 2, 4, 8, 16}) {
                for (std::size_t rows : {4096u, 4097u, 32767u, 65536u})
                    expect(choose_rows(RowLayout::Auto, model, rows, dim, fused, threads), true, 256);
                for (std::size_t rows : {0u, 17u, 4095u, 65537u, 1000000u})
                    expect(choose_rows(RowLayout::Auto, model, rows, dim, fused, threads), false, threads);
                expect(choose_rows(RowLayout::Original, model, 4096, dim, fused, threads), false, threads);
                expect(choose_rows(RowLayout::Packed, model, 17, dim, fused, threads), true, threads);
                expect(choose_rows(RowLayout::Auto, RowDevice::Unknown, 4096, dim, fused, threads), false, threads);
            }
            for (int dim : {0, 3, 32, 64, 128, 256})
                expect(choose_rows(RowLayout::Auto, model, 4096, dim, fused, threads), false, threads);
        }
    }
    // Representative existing-device boundaries remain part of the contract.
    expect(choose_rows(RowLayout::Auto, RowDevice::RTX4090, 63, 8, true, 128), false, 128);
    expect(choose_rows(RowLayout::Auto, RowDevice::RTX4090, 64, 8, true, 128), true, 256);
    expect(choose_rows(RowLayout::Auto, RowDevice::A100, 255, 2, false, 128), false, 128);
    expect(choose_rows(RowLayout::Auto, RowDevice::A100, 256, 2, false, 128), true, 256);
    expect(choose_rows(RowLayout::Auto, RowDevice::A800, 4096, 16, false, 128), true, 256);
    expect(choose_rows(RowLayout::Auto, RowDevice::A800, 4096, 16, true, 128), false, 128);
    std::cout << "PASS: " << checks << " row-policy checks\n";
}
