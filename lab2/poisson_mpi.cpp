#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>

// --- ПАРАМЕТРЫ ЗАДАЧИ ---
const int GRID_SIZE = 2048; // Размер глобальной сетки (2048x2048)
const double EPS = 0.01;    // Порог сходимости
const int MAX_ITER = 1000;  // Лимит итераций

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Создание виртуальной топологии (2D решетка)
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims); // Авто-расчет (например, 4 -> 2x2)
    
    int periods[2] = {0, 0};
    int reorder = 1; // Разрешаем MPI менять ранги для оптимизации "железа"
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    // Получаем координаты и ранк внутри новой топологии
    int cart_rank;
    int coords[2];
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    // Соседи (сдвиг по координатам)
    int top, bottom, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &top, &bottom); // По оси Y (строки)
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right); // По оси X (столбцы)

    // 2. Декомпозиция данных
    int rows_per_proc = GRID_SIZE / dims[0];
    int cols_per_proc = GRID_SIZE / dims[1];

    // Локальные размеры + гало-ячейки (границы)
    int local_rows = rows_per_proc + 2;
    int local_cols = cols_per_proc + 2;

    std::vector<double> A(local_rows * local_cols, 0.0);
    std::vector<double> A_new(local_rows * local_cols, 0.0);

    // Функция инициализации граничных условий (Dirichlet)
    // Верх = 100 градусов, остальные = 0
    auto init_bc = [&](std::vector<double>& M) {
        if (coords[0] == 0) { // Глобальный верх
            for (int j = 1; j <= cols_per_proc; ++j) M[1 * local_cols + j] = 100.0;
        }
        if (coords[0] == dims[0] - 1) { // Глобальный низ
            for (int j = 1; j <= cols_per_proc; ++j) M[rows_per_proc * local_cols + j] = 0.0;
        }
        if (coords[1] == 0) { // Глобальный левый край
            for (int i = 1; i <= rows_per_proc; ++i) M[i * local_cols + 1] = 0.0;
        }
        if (coords[1] == dims[1] - 1) { // Глобальный правый край
            for (int i = 1; i <= rows_per_proc; ++i) M[i * local_cols + cols_per_proc] = 0.0;
        }
    };

    // Применяем начальные условия
    init_bc(A);
    A_new = A;

    // Буферы для упаковки данных (лево/право - данные не подряд в памяти)
    std::vector<double> send_left(rows_per_proc), recv_left(rows_per_proc);
    std::vector<double> send_right(rows_per_proc), recv_right(rows_per_proc);

    // --- Определение диапазона вычислений ---
    // Важно: мы не должны обновлять ячейки, лежащие на глобальной границе,
    // так как там заданы константные условия (100 или 0).
    int i_start = 1, i_end = rows_per_proc;
    int j_start = 1, j_end = cols_per_proc;

    if (coords[0] == 0)           i_start = 2;               // Пропускаем 1-ю строку (там 100.0)
    if (coords[0] == dims[0] - 1) i_end   = rows_per_proc-1; // Пропускаем последнюю строку
    if (coords[1] == 0)           j_start = 2;               // Пропускаем левый столбец
    if (coords[1] == dims[1] - 1) j_end   = cols_per_proc-1; // Пропускаем правый столбец

    // Синхронизация перед стартом таймера
    MPI_Barrier(cart_comm);
    double start_time = MPI_Wtime();

    double global_diff = 0.0;
    int iter = 0;

    for (iter = 0; iter < MAX_ITER; ++iter) {
        
        // --- 3. Обмены границами (Halo Exchange) ---
        std::vector<MPI_Request> reqs;
        
        // ВЕРХ / НИЗ (данные лежат подряд, шлем напрямую из массива)
        if (top != MPI_PROC_NULL) {
            MPI_Request r1, r2;
            MPI_Isend(&A[1 * local_cols + 1], cols_per_proc, MPI_DOUBLE, top, 0, cart_comm, &r1);
            MPI_Irecv(&A[0 * local_cols + 1], cols_per_proc, MPI_DOUBLE, top, 0, cart_comm, &r2);
            reqs.push_back(r1); reqs.push_back(r2);
        }
        if (bottom != MPI_PROC_NULL) {
            MPI_Request r1, r2;
            MPI_Isend(&A[rows_per_proc * local_cols + 1], cols_per_proc, MPI_DOUBLE, bottom, 0, cart_comm, &r1);
            MPI_Irecv(&A[(rows_per_proc + 1) * local_cols + 1], cols_per_proc, MPI_DOUBLE, bottom, 0, cart_comm, &r2);
            reqs.push_back(r1); reqs.push_back(r2);
        }

        // ЛЕВО / ПРАВО (упаковка в буфер)
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

        // Ждем завершения всех обменов
        if (!reqs.empty()) {
            MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        }

        // Распаковка боковых границ
        if (left != MPI_PROC_NULL) {
            for (int i = 0; i < rows_per_proc; ++i) A[(i + 1) * local_cols + 0] = recv_left[i];
        }
        if (right != MPI_PROC_NULL) {
            for (int i = 0; i < rows_per_proc; ++i) A[(i + 1) * local_cols + cols_per_proc + 1] = recv_right[i];
        }

        // --- 4. Вычисления (Ядро Якоби) ---
        // Итерируемся только по внутренним точкам, НЕ трогая глобальные границы
        double max_diff = 0.0;
        
        for (int i = i_start; i <= i_end; ++i) {
            for (int j = j_start; j <= j_end; ++j) {
                double val = 0.25 * (A[(i - 1) * local_cols + j] + 
                                     A[(i + 1) * local_cols + j] + 
                                     A[i * local_cols + (j - 1)] + 
                                     A[i * local_cols + (j + 1)]);
                
                A_new[i * local_cols + j] = val;
                max_diff = std::max(max_diff, std::abs(val - A[i * local_cols + j]));
            }
        }

        // Меняем буферы (границы в A_new тоже корректны, т.к. мы их не трогали, а init был общим)
        std::swap(A, A_new);

        // --- 5. Проверка сходимости ---
        // Проверяем раз в 10 шагов (оптимизация)
        if (iter % 10 == 0) {
            MPI_Allreduce(&max_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
            if (global_diff < EPS) break;
        }
    }

    MPI_Barrier(cart_comm);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    double max_time = 0.0;

    // Собираем МАКСИМАЛЬНОЕ время выполнения (чтобы никто не отставал)
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    // Вывод статистики (только 0-й ранк в декартовой топологии)
    if (cart_rank == 0) {
        std::cout << "Grid: " << GRID_SIZE << "x" << GRID_SIZE << "\n";
        std::cout << "Processes: " << size << " (" << dims[0] << "x" << dims[1] << ")\n";
        std::cout << "Iterations: " << iter << "\n";
        std::cout << "Time: " << max_time << " sec" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
