#include <stdio.h>
#include <stdlib.h>
#include<iostream>

#define _USE_MATH_DEFINES
#define PI 3.14159265358979323846
#define IDX2R(i,j,N) (((i)*(N))+(j))
#include <math.h>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

// CPU functions

void read_complex_matrix_from_text(const char* file_name, double** real_part, double** imag_part, int m) {
    FILE *file = fopen(file_name, "r");
    if (!file) {
        fprintf(stderr, "Error opening file \"%s\"!\n", file_name);
        exit(1);
    }

    *real_part = (double*)malloc(sizeof(double) * m * m);
    *imag_part = (double*)malloc(sizeof(double) * m * m);
    if (!(*real_part) || !(*imag_part)) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }

    for (int i = 0; i < m * m; i++) {
        fscanf(file, "%lf %lf", &(*real_part)[i], &(*imag_part)[i]);
    }

    fclose(file);
}

void save_complex_matrix_to_text(const char* file_name, cufftDoubleComplex *data, int m) {
    FILE *file = fopen(file_name, "w");
    if (!file) {
        fprintf(stderr, "Error creating file \"%s\"!\n", file_name);
        return;
    }
    for (int i = 0; i < m * m; i++) {
        fprintf(file, "%f %f\n", data[i].x, data[i].y);
    }
    fclose(file);
}

void generate_coords(double* coordsx, double* coordsy, int M, double pixel) {
    int half_M = M / 2;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            coordsx[i * M + j] = (j - half_M) * pixel;
            coordsy[i * M + j] = (i - half_M) * pixel;
        }
    }
}

void create_h_matrix(cufftDoubleComplex* h, double* coordsx, double* coordsy, int M, double lambda, double z) {
    double k = 2 * PI / lambda;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int index = i * M + j;
            double rSq = coordsx[index] * coordsx[index] + coordsy[index] * coordsy[index] + z * z;
            h[index].x = z * cos(k * sqrt(rSq)) / (lambda * rSq);
            h[index].y = z * sin(k * sqrt(rSq)) / (lambda * rSq);
        }
    }
}

// GPU functions

void CUDA_CHECK(const char *label) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%s]: %s\n", label, cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void init_fft_u(cufftDoubleComplex *u0, double *u0_real_part, double *u0_imag_part, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < m) {
        int index = i * m + j;
        u0[index].x = u0_real_part[index];
        u0[index].y = u0_imag_part[index];
    }
}

__global__ void elementwise_multiply(cufftDoubleComplex *C, cufftDoubleComplex *A, cufftDoubleComplex *B, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        C[i].x = A[i].x * B[i].x - A[i].y * B[i].y;
        C[i].y = A[i].x * B[i].y + A[i].y * B[i].x;
    }
}

__global__ void fftshift(double2 *data, int N1, int N2)
{
int i = threadIdx.y + blockDim.y * blockIdx.y;
int j = threadIdx.x + blockDim.x * blockIdx.x;

if (i < N1 && j < N2) 
    {
       double a = 1-2*((i+j)&1);
       data[IDX2R(i,j,N2)].x *= a;
       data[IDX2R(i,j,N2)].y *= a;
   }
}

__global__ void normalize(cufftDoubleComplex *data, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index].x /= size;
        data[index].y /= size;
    }
}

void propagate(cufftDoubleComplex *u0, cufftDoubleComplex *h, cufftDoubleComplex *C, int m) {
    cufftHandle plan_fwd, plan_bwd;
    cufftPlan2d(&plan_fwd, m, m, CUFFT_Z2Z); CUDA_CHECK("cufftPlan2d (plan_fwd)");
    cufftPlan2d(&plan_bwd, m, m, CUFFT_Z2Z); CUDA_CHECK("cufftPlan2d (plan_bwd)");

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (m + threadsPerBlock.y - 1) / threadsPerBlock.y); CUDA_CHECK("kernel");

    fftshift<<<numBlocks, threadsPerBlock>>>(u0, m, m); CUDA_CHECK("fftshift");
    fftshift<<<numBlocks, threadsPerBlock>>>(h, m, m); CUDA_CHECK("fftshift");
   
    cufftExecZ2Z(plan_fwd, u0, u0, CUFFT_FORWARD); CUDA_CHECK("cufftExec(plan u)");
    cufftExecZ2Z(plan_fwd, h, h, CUFFT_FORWARD); CUDA_CHECK("cufftExec(plan h)");

    fftshift<<<numBlocks, threadsPerBlock>>>(u0, m, m); CUDA_CHECK("fftshift (u0)");
    fftshift<<<numBlocks, threadsPerBlock>>>(h, m, m); CUDA_CHECK("fftshift (h)");

    elementwise_multiply<<<(m * m + 255) / 256, 256>>>(C, u0, h, m * m); CUDA_CHECK("multiply");

    fftshift<<<numBlocks, threadsPerBlock>>>(C, m, m); CUDA_CHECK("fftshift (C)");

    cufftExecZ2Z(plan_bwd, C, C, CUFFT_INVERSE); CUDA_CHECK("cufftExec(plan_bwd)");
    fftshift<<<numBlocks, threadsPerBlock>>>(C, m, m); CUDA_CHECK("fftshift");
    
    dim3 threadsPerBlock_norm(256); 
    dim3 numBlocks_norm( (m + threadsPerBlock_norm.x - 1) / threadsPerBlock_norm.x, 
                    (m + threadsPerBlock_norm.y - 1) / threadsPerBlock_norm.y);
    normalize<<< numBlocks_norm,threadsPerBlock_norm>>>(C, m); CUDA_CHECK("normalize");

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_bwd);
    printf(">> Finished propagation\n");
}

int main() {

    const char* file_name = "source_500.txt";
    const char* slm_name = "spatial_modulation_500_ones.txt";
    const char* output_file = "out_cuda_15000_test.txt";

    double lambda = 0.6e-6;
    double z = 1.0;
    double L1 = 8e-3;
    int M = 500;
    double pixel = L1 / M;

    double *u_real, *u_imag;
    double *sp_real, *sp_imag;
    
    read_complex_matrix_from_text(file_name, &u_real, &u_imag, M);
    read_complex_matrix_from_text(slm_name, &sp_real, &sp_imag, M);
    cufftDoubleComplex* result = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * M * M);
    if (result == NULL) {
        fprintf(stderr, "Failed to allocate host memory for result.\n");
        return EXIT_FAILURE;
    }

    cufftDoubleComplex *d_u0, *d_h, *d_spatial, *d_C, *d_C1, *d_C2;
    double *d_u_real, *d_u_imag, *d_sp_real, *d_sp_imag;
    double* coordsx = (double*)malloc(sizeof(double) * M * M);
    double* coordsy = (double*)malloc(sizeof(double) * M * M);
    generate_coords(coordsx, coordsy, M, pixel);
    cufftDoubleComplex* h = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * M * M);
    create_h_matrix(h, coordsx, coordsy, M, lambda, z);

    /****
     * Allocation on the gpu
     */
    if( cudaMalloc( ( void** ) & d_u0, sizeof(cufftDoubleComplex) * M * M) != cudaSuccess ||
        cudaMalloc( ( void** ) & d_h,  sizeof(cufftDoubleComplex) * M * M) != cudaSuccess  ||
        cudaMalloc( ( void** ) & d_spatial, sizeof(cufftDoubleComplex) * M * M) != cudaSuccess  ||
        cudaMalloc( ( void** ) & d_C, sizeof(cufftDoubleComplex) * M * M) != cudaSuccess  ||
        cudaMalloc( ( void** ) & d_C1, sizeof(cufftDoubleComplex) * M * M) != cudaSuccess  ||
        cudaMalloc( ( void**) & d_C2, sizeof(cufftDoubleComplex) * M * M) != cudaSuccess 
        )
    {
        std::cerr << "Unable to allocate cufftDoubleComplex vectors on the device." << std::endl;
        return EXIT_FAILURE;
    }

    if( cudaMalloc( ( void** ) & d_u_real, sizeof(double) * M * M) != cudaSuccess  ||
        cudaMalloc( ( void** ) & d_u_imag, sizeof(double) * M * M) != cudaSuccess  ||
        cudaMalloc( ( void** ) & d_sp_real, sizeof(double) * M * M) != cudaSuccess  ||
        cudaMalloc( ( void** ) & d_sp_imag, sizeof(double) * M * M) != cudaSuccess    
        )
    {
        std::cerr << "Unable to allocate double-type vectors on the device." << std::endl;
        return EXIT_FAILURE;
    }

    /****
     * Transfer to device
     */
    if(  cudaMemcpy(d_u_real, u_real, sizeof(double) * M * M, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_u_imag, u_imag, sizeof(double) * M * M, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_sp_real, sp_real, sizeof(double) * M * M, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_sp_imag, sp_imag, sizeof(double) * M * M, cudaMemcpyHostToDevice) != cudaSuccess || 
        cudaMemcpy(d_h, h, sizeof(cufftDoubleComplex) * M * M, cudaMemcpyHostToDevice)  != cudaSuccess
        )
    {
        std::cerr << "Unable to copy data from the host to the device." << std::endl;
        return EXIT_FAILURE;
    }


      /****
     * Run the CUDA kernel
     */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Assuming device 0
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max block dimensions: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid dimensions: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    dim3 threadsPerBlock(256); 
    dim3 numBlocks( (M + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    init_fft_u<<<numBlocks, threadsPerBlock>>>(d_u0, d_u_real, d_u_imag, M);
    init_fft_u<<<numBlocks, threadsPerBlock>>>(d_spatial, d_sp_real, d_sp_imag, M);

    propagate(d_u0, d_h, d_C, M);

    //phase modulation
    elementwise_multiply<<<(M * M + 255) / 256, 256>>>(d_C1, d_C, d_spatial, M * M);
   
    propagate(d_C1, d_h, d_C2, M);


     /****
     * Copy data back to the host
     */
     if( cudaMemcpy(result, d_C2, sizeof(cufftDoubleComplex) * M * M, cudaMemcpyDeviceToHost)!= cudaSuccess )
    {
        std::cerr << "Unable to copy data back from the GPU." << std::endl;
        return EXIT_FAILURE;
    }
    
    save_complex_matrix_to_text(output_file, result, M);
    printf("Completed, saved.\n");
    
    /****
     * Freeing allocated memory
     */
    cudaFree(d_u0);
    cudaFree(d_h);
    cudaFree(d_spatial);
    cudaFree(d_C);
    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_u_real);
    cudaFree(d_u_imag);
    cudaFree(d_sp_real);
    cudaFree(d_sp_imag);
    free(coordsx);
    free(coordsy);
    free(u_real);
    free(u_imag);
    free(sp_real);
    free(sp_imag);
    free(d_h);

    return 0;
}
