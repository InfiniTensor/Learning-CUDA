#include <stdio.h>
#include <musa_runtime.h>

// musa kernel vector Add, C = A + B
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < numElements) {
C[i] = A[i] + B[i];
}
}

int main(void) {
musaError_t err = musaSuccess;
int numElements = 50000;
size_t size = numElements * sizeof(float);
float *h_A = (float *)malloc(size);
float *h_B = (float *)malloc(size);
float *h_C = (float *)malloc(size);
if (h_A == NULL || h_B == NULL || h_C == NULL) {
fprintf(stderr, "Failed to allocate host vectors!\n");
exit(EXIT_FAILURE);
}
for (int i = 0; i < numElements; ++i) {
h_A[i] = rand()/(float)RAND_MAX;
h_B[i] = rand()/(float)RAND_MAX;
}

// Allocate the device input vector A
float *d_A = NULL;
err = musaMalloc((void **)&d_A, size);
if (err != musaSuccess) {
fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", musaGetErrorString(err));
exit(EXIT_FAILURE);
}
// Allocate the device input vector B
float *d_B = NULL;
err = musaMalloc((void **)&d_B, size);
if (err != musaSuccess) {
fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", musaGetErrorString(err));
exit(EXIT_FAILURE);
}
// Allocate the device output vector C
float *d_C = NULL;
err = musaMalloc((void **)&d_C, size);
if (err != musaSuccess) {
fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", musaGetErrorString(err));
exit(EXIT_FAILURE);
}

// Copy the host input vectors A and B in host memory to the device input vectors in
// device memory
printf("Copy input data from the host memory to the MUSA device\n");
err = musaMemcpy(d_A, h_A, size, musaMemcpyHostToDevice);
if (err != musaSuccess) {
fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", musaGetErrorString(err));
exit(EXIT_FAILURE);
}
err = musaMemcpy(d_B, h_B, size, musaMemcpyHostToDevice);
if (err != musaSuccess) {
fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", musaGetErrorString(err));
exit(EXIT_FAILURE);
}
// Launch the Vector Add MUSA Kernel
int threadsPerBlock = 256;
int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
printf("MUSA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
err = musaGetLastError();
if (err != musaSuccess) {
fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", musaGetErrorString(err));
exit(EXIT_FAILURE);
}
// Copy the device result vector in device memory to the host result vector
// in host memory.
printf("Copy output data from the MUSA device to the host memory\n");
err = musaMemcpy(h_C, d_C, size, musaMemcpyDeviceToHost);
if (err != musaSuccess) {
fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", musaGetErrorString(err));
exit(EXIT_FAILURE);
}
// Verify that the result vector is correct
for (int i = 0; i < numElements; ++i) {
if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
fprintf(stderr, "Result verification failed at element %d!\n", i);
exit(EXIT_FAILURE);
}
}
printf("Test PASSED\n");
// Free device global memory
err = musaFree(d_A);
if (err != musaSuccess) {
fprintf(stderr, "Failed to free device vector A (error code %s)!\n", musaGetErrorString(err));
exit(EXIT_FAILURE);
}
err = musaFree(d_B);
if (err != musaSuccess) {
fprintf(stderr, "Failed to free device vector B (error code %s)!\n", musaGetErrorString(err));
exit(EXIT_FAILURE);
}
err = musaFree(d_C);
if (err != musaSuccess) {
fprintf(stderr, "Failed to free device vector C (error code %s)!\n", musaGetErrorString(err));
exit(EXIT_FAILURE);
}
// Free host memory
free(h_A);
free(h_B);
free(h_C);
printf("Done\n");
return 0;
}