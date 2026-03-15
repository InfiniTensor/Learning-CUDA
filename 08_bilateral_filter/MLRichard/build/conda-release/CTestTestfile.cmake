# CMake generated Testfile for 
# Source directory: /home/richard/2026/Learning-CUDA/08_bilateral_filter/MLRichard
# Build directory: /home/richard/2026/Learning-CUDA/08_bilateral_filter/MLRichard/build/conda-release
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[correctness]=] "/home/richard/2026/Learning-CUDA/08_bilateral_filter/MLRichard/build/conda-release/test_correctness")
set_tests_properties([=[correctness]=] PROPERTIES  LABELS "correctness" _BACKTRACE_TRIPLES "/home/richard/2026/Learning-CUDA/08_bilateral_filter/MLRichard/CMakeLists.txt;48;add_test;/home/richard/2026/Learning-CUDA/08_bilateral_filter/MLRichard/CMakeLists.txt;0;")
add_test([=[performance]=] "/home/richard/2026/Learning-CUDA/08_bilateral_filter/MLRichard/build/conda-release/test_performance")
set_tests_properties([=[performance]=] PROPERTIES  LABELS "performance;benchmark" _BACKTRACE_TRIPLES "/home/richard/2026/Learning-CUDA/08_bilateral_filter/MLRichard/CMakeLists.txt;51;add_test;/home/richard/2026/Learning-CUDA/08_bilateral_filter/MLRichard/CMakeLists.txt;0;")
