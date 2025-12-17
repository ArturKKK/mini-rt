cmake_minimum_required(VERSION 3.10)
project(MatMulCUDA LANGUAGES CXX CUDA)

cmake_policy(SET CMP0104 NEW)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# V100 = sm_70
set(CMAKE_CUDA_ARCHITECTURES 70)

find_package(CUDAToolkit REQUIRED)

add_executable(matmul_cuda matmul.cu)
target_link_libraries(matmul_cuda PRIVATE CUDA::cublas)

target_compile_options(matmul_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)
