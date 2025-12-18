#include <mpi.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

// --- ПАРАМЕТРЫ ЗАДАЧИ ---
const int GRID_SIZE = 2048; 
const double EPS = 0.01;    
const int MAX_ITER = 1000;  

int main(int argc, char** argv) {
    int provided;
    // Инициализируем MPI с поддержкой многопоточности 
    // MPI_THREAD_FUNNELED означает, что MPI вызовы делает только главный поток, 
    // но вычисления могут идти в нескольких потоках.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Топология. Представляем процессы квадратом, чтобы легко находить соседей сверху/низу/слева/справа.
    int dims[2] = {0, 0};
    // MPI_Dims_create: MPI сам посчитает, как лучше разбить N процессов на сетку X*Y.
    MPI_Dims_create(size, 2, dims);
    
    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;
     // Создаем саму решетку процессов
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int cart_rank;
    int coords[2]; // Наши координаты в решетке
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    // Находим соседей. MPI_Cart_shift сам скажет, кто твой сосед.
    // Если соседа нет (на краю), он вернет MPI_PROC_NULL.
    int top, bottom, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &top, &bottom);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    // Декомпозиция
    int rows_per_proc = GRID_SIZE / dims[0];
    int cols_per_proc = GRID_SIZE / dims[1];

    // Добавляем теневые грани
    // Это рамка толщиной в 1 клетку вокруг наших данных, куда мы запишем данные соседей.
    int local_rows = rows_per_proc + 2;
    int local_cols = cols_per_proc + 2;

    std::vector<double> A(local_rows * local_cols, 0.0);
    std::vector<double> A_new(local_rows * local_cols, 0.0);

    // Граничные условия
    auto init_bc = [&](std::vector<double>& M) {
        if (coords[0] == 0) { 
            #pragma omp parallel for
            for (int j = 1; j <= cols_per_proc; ++j) M[1 * local_cols + j] = 100.0;
        }
        if (coords[0] == dims[0] - 1) { 
            #pragma omp parallel for
            for (int j = 1; j <= cols_per_proc; ++j) M[rows_per_proc * local_cols + j] = 0.0;
        }
        if (coords[1] == 0) { 
            #pragma omp parallel for
            for (int i = 1; i <= rows_per_proc; ++i) M[i * local_cols + 1] = 0.0;
        }
        if (coords[1] == dims[1] - 1) { 
            #pragma omp parallel for
            for (int i = 1; i <= rows_per_proc; ++i) M[i * local_cols + cols_per_proc] = 0.0;
        }
    };

    init_bc(A);
    A_new = A;

    // Буферы для обмена боковыми границами
    std::vector<double> send_left(rows_per_proc), recv_left(rows_per_proc);
    std::vector<double> send_right(rows_per_proc), recv_right(rows_per_proc);

    // Определяем зону вычислений.
    int i_start = 1, i_end = rows_per_proc;
    int j_start = 1, j_end = cols_per_proc;

    // Если мы на самом краю глобальной сетки, то крайнюю линию тоже не считаем
    if (coords[0] == 0)           i_start = 2;
    if (coords[0] == dims[0] - 1) i_end   = rows_per_proc-1;
    if (coords[1] == 0)           j_start = 2;
    if (coords[1] == dims[1] - 1) j_end   = cols_per_proc-1;

    MPI_Barrier(cart_comm); // Ждем всех перед стартом таймера
    double start_time = MPI_Wtime();

    double global_diff = 0.0;
    int iter = 0;

    for (iter = 0; iter < MAX_ITER; ++iter) {
        
        // ОБМЕНЫ (MPI)
        // Массив запросов для асинхронных операций
        std::vector<MPI_Request> reqs;
        
        // Верх/Низ
        if (top != MPI_PROC_NULL) {
            MPI_Request r1, r2;
            // Отправляем свою 1-ю рабочую строку наверх
            MPI_Isend(&A[1 * local_cols + 1], cols_per_proc, MPI_DOUBLE, top, 0, cart_comm, &r1);
            // Ждем данные сверху в свою 0-ю (теневую) строку
            MPI_Irecv(&A[0 * local_cols + 1], cols_per_proc, MPI_DOUBLE, top, 0, cart_comm, &r2);
            reqs.push_back(r1); reqs.push_back(r2);
        }
        if (bottom != MPI_PROC_NULL) {
            MPI_Request r1, r2;
            MPI_Isend(&A[rows_per_proc * local_cols + 1], cols_per_proc, MPI_DOUBLE, bottom, 0, cart_comm, &r1);
            MPI_Irecv(&A[(rows_per_proc + 1) * local_cols + 1], cols_per_proc, MPI_DOUBLE, bottom, 0, cart_comm, &r2);
            reqs.push_back(r1); reqs.push_back(r2);
        }

        // Лево/Право
        if (left != MPI_PROC_NULL) {
            for (int i = 0; i < rows_per_proc; ++i) send_left[i] = A[(i + 1) * local_cols + 1];
            MPI_Request r1, r2;
            MPI_Isend(send_left.data(), rows_per_proc, MPI_DOUBLE, left, 0, cart_comm, &r1);
            MPI_Irecv(recv_left.data(), rows_per_proc, MPI_DOUBLE, left, 0, cart_comm, &r2);
            reqs.push_back(r1); reqs.push_back(r2);
        }
        if (right != MPI_PROC_NULL) {
            for (int i = 0; i < rows_per_proc; ++i) send_right[i] = A[(i + 1) * local_cols + cols_per_proc];
            MPI_Request r1, r2;
            MPI_Isend(send_right.data(), rows_per_proc, MPI_DOUBLE, right, 0, cart_comm, &r1);
            MPI_Irecv(recv_right.data(), rows_per_proc, MPI_DOUBLE, right, 0, cart_comm, &r2);
            reqs.push_back(r1); reqs.push_back(r2);
        }

        // Ждем, пока ВСЕ обмены (отправки и приемы) завершатся
        if (!reqs.empty()) {
            MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        }

        // Распаковка. Принятые боковые данные перекладываем из буфера в теневые ячейки
        if (left != MPI_PROC_NULL) {
            for (int i = 0; i < rows_per_proc; ++i) A[(i + 1) * local_cols + 0] = recv_left[i];
        }
        if (right != MPI_PROC_NULL) {
            for (int i = 0; i < rows_per_proc; ++i) A[(i + 1) * local_cols + cols_per_proc + 1] = recv_right[i];
        }

        // ВЫЧИСЛЕНИЯ
        double max_diff = 0.0;
        
        // collapse(2) объединяет вложенные циклы для лучшей нагрузки потоков
        #pragma omp parallel for collapse(2) reduction(max:max_diff)
        for (int i = i_start; i <= i_end; ++i) {
            for (int j = j_start; j <= j_end; ++j) {
                // Метод Якоби: новое значение = среднее арифметическое 4-х соседей
                double val = 0.25 * (A[(i - 1) * local_cols + j] + 
                                     A[(i + 1) * local_cols + j] + 
                                     A[i * local_cols + (j - 1)] + 
                                     A[i * local_cols + (j + 1)]);
                
                A_new[i * local_cols + j] = val;
                double diff = std::abs(val - A[i * local_cols + j]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        std::swap(A, A_new);

        // Проверка сходимости
        if (iter % 10 == 0) {
            MPI_Allreduce(&max_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
            if (global_diff < EPS) break;
        }
    }

    MPI_Barrier(cart_comm);
    double end_time = MPI_Wtime();
    double max_time = 0.0;
    double local_time = end_time - start_time;
    // Ищем самое долгое время среди всех процессов
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (cart_rank == 0) {
        // Вывод: MPI_Procs OMP_Threads Time
        std::cout << size << " " << omp_get_max_threads() << " " << max_time << std::endl;
    }

    MPI_Finalize();
    return 0;
}