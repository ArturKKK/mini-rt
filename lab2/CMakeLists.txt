cmake_minimum_required(VERSION 3.10)
project(PoissonHybrid CXX)

set(CMAKE_CXX_STANDARD 17)

# Подключаем MPI и OpenMP
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)

add_executable(poisson_hybrid poisson_mpi.cpp)

# Линкуем библиотеки
target_link_libraries(poisson_hybrid PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Оптимизации
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
    target_compile_options(poisson_hybrid PRIVATE -O3 -march=native)
endif()