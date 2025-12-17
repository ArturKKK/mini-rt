#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Используем float как стандарт
using DType = float;

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Макрос для проверки ошибок cuBLAS
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

__global__ void matrixMulNaive(const DType* A, const DType* B, DType* C, int N) {
    // Вычисляем глобальные индексы потока
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        DType sum = 0.0f;
        // Скалярное произведение строки A и столбца B
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Заполнение случайными числами
void fillMatrix(std::vector<DType>& mat) {
    for (auto& val : mat) {
        val = static_cast<DType>(rand()) / RAND_MAX;
    }
}

// Проверка корректности (Сравниваем Naive и cuBLAS)
void verifyResult(const std::vector<DType>& ref, const std::vector<DType>& res, int N) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double a = (double)ref[i];
        double b = (double)res[i];
        double diff = std::fabs(a - b);
        max_abs = std::max(max_abs, diff);
        double denom = std::max(1.0, std::fabs(a));
        max_rel = std::max(max_rel, diff / denom);
    }
    std::cout << "  Verification: MaxAbs=" << max_abs
              << " MaxRel=" << max_rel;

    if (max_rel < 1e-2) std::cout << " [OK]\n";
    else                std::cout << " [FAIL]\n";
}


int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 2048;
    std::cout << "Matrix Size: " << N << "x" << N << std::endl;

    size_t bytes = N * N * sizeof(DType);

    // Хост память
    std::vector<DType> h_A(N * N);
    std::vector<DType> h_B(N * N);
    std::vector<DType> h_C_naive(N * N);
    std::vector<DType> h_C_cublas(N * N);

    fillMatrix(h_A);
    fillMatrix(h_B);

    // Девайс память
    DType *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Копирование Host to Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // Настройка сетки потоков
    // Block 32x32 = 1024 потока (максимум для CUDA-блока)
    // dim3 threadsPerBlock(32, 32);
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // События для тайминга
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    // Warmup (прогрев) - запускаем, но не замеряем
    matrixMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Замер
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    double time_naive = milliseconds / 1000.0;
    std::cout << "  Naive GPU Time:  " << std::fixed << std::setprecision(4) << time_naive << " sec" << std::endl;

    // Забираем результат Naive
    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // cuBLAS Library
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    DType alpha = 1.0f;
    DType beta = 0.0f;

    // Warmup cuBLAS
    // Трюк: меняем местами A и B, чтобы получить Row-Major результат в C
    // C = alpha * (B * A) + beta * C  (для cuBLAS это A^T * B^T)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Замер
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    double time_cublas = milliseconds / 1000.0;
    std::cout << "  cuBLAS GPU Time: " << std::fixed << std::setprecision(4) << time_cublas << " sec" << std::endl;

    // Забираем результат cuBLAS
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << "  Speedup: " << std::setprecision(2) << time_naive / time_cublas << "x" << std::endl;

    // Сравнение результатов
    verifyResult(h_C_naive, h_C_cublas, N);

    // Очистка
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
