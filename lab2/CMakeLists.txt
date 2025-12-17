cmake_minimum_required(VERSION 3.10)
project(PoissonMPI CXX)

# Ищем MPI с компонентами C++
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(poisson_mpi poisson_mpi.cpp)

# Линкуем
target_link_libraries(poisson_mpi PRIVATE MPI::MPI_CXX)

# Включаем оптимизации для скорости
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
    target_compile_options(poisson_mpi PRIVATE -O3 -march=native)
endif()
