#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
// Подключаем заголовки CUDA
#include <cuda_runtime.h> // Базовые функции (cudaMalloc, cudaMemcpy...)
#include <cublas_v2.h>    // Библиотека линейной алгебры

// Используем float, так как видеокарты лучше всего работают с одинарной точностью
using DType = float;

// МАКРОСЫ ДЛЯ ОБРАБОТКИ ОШИБОК
// CUDA функции возвращают код ошибки. Если он не cudaSuccess, программа падает с сообщением.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// __global__ означает, что функция вызывается с CPU (Host), но выполняется на GPU (Device).
// Каждый поток выполняет эту функцию независимо.
__global__ void matrixMulNaive(const DType* A, const DType* B, DType* C, int N) {
    // ВЫЧИСЛЕНИЕ КООРДИНАТ
    // У нас 2D сетка блоков. Нам нужно знать, какой пиксель считает этот поток.
    // blockIdx.y * blockDim.y -> смещение блока по вертикали
    // threadIdx.y             -> смещение потока внутри блока
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // ПРОВЕРКА ГРАНИЦ
    if (row < N && col < N) {
        DType sum = 0.0f;
        // ВЫЧИСЛЕНИЯ
        // Обычное умножение строки A на столбец B.
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        // Записываем результат в глобальную память видеокарты
        C[row * N + col] = sum;
    }
}

// Вспомогательная функция для заполнения матрицы на CPU
void fillMatrix(std::vector<DType>& mat) {
    for (auto& val : mat) {
        val = static_cast<DType>(rand()) / RAND_MAX;
    }
}

// Вспомогательная функция для проверки (сравниваем наш результат с cuBLAS)
void verifyResult(const std::vector<DType>& ref, const std::vector<DType>& res, int N) {
    double max_rel = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double a = (double)ref[i];
        double b = (double)res[i];
        double diff = std::fabs(a - b);
        double denom = std::max(1.0, std::fabs(a));
        max_rel = std::max(max_rel, diff / denom);
    }
    // Если относительная ошибка меньше 1%, считаем, что все ок.
    // Различия возможны из-за разного порядка округления float.
    if (max_rel < 1e-2) std::cout << "Verification: [OK]\n";
    else                std::cout << "Verification: [FAIL] MaxRel=" << max_rel << "\n";
}


int main(int argc, char** argv) {
    // Читаем размер матрицы из аргументов
    int N = (argc > 1) ? std::atoi(argv[1]) : 2048;
    std::cout << "Matrix Size: " << N << "x" << N << std::endl;

    size_t bytes = N * N * sizeof(DType);

    // ВЫДЕЛЕНИЕ ПАМЯТИ НА HOST (CPU)
    std::vector<DType> h_A(N * N);
    std::vector<DType> h_B(N * N);
    std::vector<DType> h_C_naive(N * N);  // Сюда запишем результат нашего ядра
    std::vector<DType> h_C_cublas(N * N); // Сюда результат библиотеки

    // Заполняем случайными числами
    fillMatrix(h_A);
    fillMatrix(h_B);

    // ВЫДЕЛЕНИЕ ПАМЯТИ НА DEVICE (GPU)
    // Указатели d_A, d_B, d_C хранят адреса в видеопамяти
    // Мы не можем читать их напрямую с CPU (без cudaMemcpy).
    DType *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // КОПИРОВАНИЕ ДАННЫХ
    // Пересылаем матрицы из ОЗУ в видеопамять
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // НАСТРОЙКА СЕТКИ ЗАПУСКА
    // threadsPerBlock: Размер одного блока потоков. 16x16 = 256 потоков.
    dim3 threadsPerBlock(16, 16);
    
    // blocksPerGrid: Количество блоков, чтобы накрыть всю матрицу N.
    // Формула (N + block - 1) / block — это деление с округлением вверх.
    // Если N=100, а блок=16, нам нужно ceil(100/16) = 7 блоков.
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Подготовка событий для точного замера времени на GPU
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    // ЗАПУСК 1: NAIVE IMPLEMENTATION
    
    // Warmup Первый запуск ядра всегда медленный.
    // Запускаем один раз вхолостую, чтобы замеры были честными.
    matrixMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize()); // Ждем завершения

    // Настоящий замер
    CUDA_CHECK(cudaEventRecord(start)); // Засекаем время
    // <<<Grid, Block>>> - спец. синтаксис запуска ядра
    matrixMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));  // Останавливаем время
    
    CUDA_CHECK(cudaEventSynchronize(stop)); // Ждем, пока GPU закончит
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop)); // Считаем разницу
    
    double time_naive = milliseconds / 1000.0;
    std::cout << "  Naive GPU Time:  " << std::fixed << std::setprecision(4) << time_naive << " sec" << std::endl;

    // Забираем результат с видеокарты обратно в ОЗУ (D2H: Device -> Host)
    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // ЗАПУСК 2: cuBLAS (БИБЛИОТЕКА)
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle)); // Инициализация контекста cuBLAS
    DType alpha = 1.0f;
    DType beta = 0.0f;

    // Warmup для cuBLAS
    // ТРЮК С ПОРЯДКОМ МАТРИЦ:
    // C++ хранит матрицы по строкам (Row-Major).
    // cuBLAS (и Fortran) хранит по столбцам (Column-Major).
    // Математически: (A * B)^T = B^T * A^T.
    // Если мы скажем cuBLAS, что наши матрицы лежат в памяти "как есть", он будет думать, 
    // что это транспонированные матрицы. Поэтому мы меняем местами A и B в аргументах.
    // Считаем: C = B * A (для cuBLAS это выглядит как A * B в нормальном виде).
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Замер cuBLAS
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    double time_cublas = milliseconds / 1000.0;
    std::cout << "  cuBLAS GPU Time: " << std::fixed << std::setprecision(4) << time_cublas << " sec" << std::endl;

    // Забираем результат
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Считаем ускорение (во сколько раз библиотека быстрее нас)
    std::cout << "  Speedup: " << std::setprecision(2) << time_naive / time_cublas << "x" << std::endl;

    // Проверяем, что результаты совпали
    verifyResult(h_C_naive, h_C_cublas, N);

    // ОЧИСТКА ПАМЯТИ
    // Освобождаем видеопамять (иначе будет утечка до перезагрузки драйвера)
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}