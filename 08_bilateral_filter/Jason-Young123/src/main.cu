#include <bilateral.h>
#include <myopencv.h>

int main(){

#ifndef HAS_CV
    std::cout << "\n" << BOLD << RED 
              << "******************************************************************\n"
              << " [WARNING] OpenCV is NOT detected on this server!\n"
              << " [NOTICE ] CPU Reference will be SKIPPED!\n"
              << "******************************************************************" 
              << RESET << std::endl;
#endif

    //genTester();
    //return 0;
    int warmup_round = 3;
    int test_round = 15;
    //bool only4K = false;
    //runTester("tester", warmup_round, test_round, only4K);
    
    runAll("tester", warmup_round, test_round);

    return 0;
}