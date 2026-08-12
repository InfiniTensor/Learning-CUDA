root@autodl-container-mamykqv3ku-9260f267:~/autodl-tmp/Learning-CUDA# SKIP_ATTENTION=1 make
=== Running tests (output from src/kernels.o) ===
=== Verbose mode: Enabled (using '--verbose') ===
./test_kernels  
Testing on device: NVIDIA GeForce RTX 3080 Ti

=== rmsNorm Tests ===
Test # 1: float  | Verification: Passed
Test # 1: half   | Verification: Passed
Test # 2: float  | Verification: Passed
Test # 2: half   | Verification: Passed
Test # 3: float  | Verification: Passed
Test # 3: half   | Verification: Passed
Test # 4: float  | Verification: Passed
Test # 4: half   | Verification: Passed
Test # 5: float  | Verification: Passed
Test # 5: half   | Verification: Passed
Test # 6: float  | Verification: Passed
Test # 6: half   | Verification: Passed
Test # 7: float  | Verification: Passed
Test # 7: half   | Verification: Passed
Test # 8: float  | Verification: Passed
Test # 8: half   | Verification: Passed
Test # 9: float  | Verification: Passed
Test # 9: half   | Verification: Passed
Test #10: float  | Verification: Passed
Test #10: half   | Verification: Passed
Test #11: float  | Verification: Passed
Test #11: half   | Verification: Passed
Test #12: float  | Verification: Passed
Test #12: half   | Verification: Passed
Test #13: float  | Verification: Passed
Test #13: half   | Verification: Passed

root@autodl-container-mamykqv3ku-9260f267:~/autodl-tmp/Learning-CUDA# SKIP_RMS_NORM=1 make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
=== Linking executable (student code + test logic) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -o test_kernels src/kernels.o tester/tester_nv.o 
=== Running tests (output from src/kernels.o) ===
=== Verbose mode: Enabled (using '--verbose') ===
./test_kernels  
Testing on device: NVIDIA GeForce RTX 3080 Ti

=== Attention Tests ===
Test # 1: float  | Verification: Passed
Test # 1: half   | Verification: Passed
Test # 2: float  | Verification: Passed
Test # 2: half   | Verification: Passed
Test # 3: float  | Verification: Passed
Test # 3: half   | Verification: Passed
Test # 4: float  | Verification: Passed
Test # 4: half   | Verification: Passed
Test # 5: float  | Verification: Passed
Test # 5: half   | Verification: Passed
Test # 6: float  | Verification: Passed
Test # 6: half   | Verification: Passed
Test # 7: float  | Verification: Passed
Test # 7: half   | Verification: Passed
Test # 8: float  | Verification: Passed
Test # 8: half   | Verification: Passed
Test # 9: float  | Verification: Passed
Test # 9: half   | Verification: Passed
Test #10: float  | Verification: Passed
Test #10: half   | Verification: Passed
Test #11: float  | Verification: Passed
Test #11: half   | Verification: Passed
Test #12: float  | Verification: Passed
Test #12: half   | Verification: Passed
Test #13: float  | Verification: Passed
Test #13: half   | Verification: Passed
Test #14: float  | Verification: Passed
Test #14: half   | Verification: Passed
