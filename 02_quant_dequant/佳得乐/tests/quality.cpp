#include "project.hpp"
#include <iomanip>
#include <iostream>
using namespace lp;
int main() {
    try {
        std::cout << "format,input_dtype,output_dtype,mode,round,distribution,max_abs,mae,mse,"
                     "payload_ratio,file_ratio\n"
                  << std::setprecision(12);
        for (int f = 0; f < 3; ++f)
            for (int dt = 0; dt < 2; ++dt)
                for (int out = 0; out < 3; ++out)
                    for (int mode = 0; mode < 2; ++mode)
                        for (int round = 0; round < 2; ++round)
                            for (std::string dist : {"uniform", "normal", "outliers"}) {
                                Config c;
                                c.format = f;
                                c.block = f == 2 ? 16 : 32;
                                c.out = out;
                                c.mode = mode;
                                c.round = round;
                                auto x = generate(257, 259, dt, dist, 42);
                                auto p = quant_cpu(x, c);
                                auto y = dequant_cpu(p);
                                auto e = error(x, y);
                                std::cout
                                    << f << ',' << dt << ',' << out << ',' << mode << ',' << round
                                    << ',' << dist << ',' << e.max << ',' << e.mae << ',' << e.mse
                                    << ','
                                    << double(x.data.size()) /
                                           (p.data.size() + p.scales.size() + (f == 2 ? 4 : 0))
                                    << ','
                                    << double(x.data.size() + 48) /
                                           (p.data.size() + p.scales.size() + 112)
                                    << '\n';
                            }
    } catch (std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
