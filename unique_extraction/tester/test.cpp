#include <omp.h>
#include <iostream>
#include <vector>

int main() {
    int n = 100, cnt = 0;
    std::vector<int> arr(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int now;
        #pragma omp atomic capture
            now = cnt ++;
        arr[i] = now;
    }

    for (auto x : arr) {
        std::cout << x << " ";
    } std::cout << "\n";
}