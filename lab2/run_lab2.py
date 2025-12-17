import os
import subprocess
import shutil

build_dir = "build"
executable = f"./{build_dir}/poisson_hybrid"
results_csv = "lab2_hybrid_results.csv"

TOTAL_CORES = int(os.getenv("LAB2_CORES", "8"))

# Генерируем конфигурации (MPI x OMP), которые в сумме дают 8
# (8,1), (4,2), (2,4), (1,8)
configs = []
p = TOTAL_CORES
while p >= 1:
    t = TOTAL_CORES // p
    configs.append((p, t))
    p //= 2

print(f">>> Running Hybrid MPI+OpenMP Benchmarks on {TOTAL_CORES} cores")
print(f">>> Configurations (MPI processes x OMP threads): {configs}")

# 1. Очистка и сборка
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)
os.makedirs(build_dir)

print("\n>>> Building...")
# Добавим флаги компиляции для уверенности
env = os.environ.copy()
subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir, check=True, env=env)
subprocess.run(["cmake", "--build", ".", "-j", str(TOTAL_CORES)], cwd=build_dir, check=True, env=env)

print(f"\n{'MPI':<5} | {'OMP':<5} | {'Time (s)':<10} | {'Status':<10}")
print("-" * 45)

with open(results_csv, "w") as f:
    f.write("MPI,OMP,Time\n")
    
    for mpi_procs, omp_threads in configs:
        # Устанавливаем переменные окружения для OpenMP
        current_env = os.environ.copy()
        current_env["OMP_NUM_THREADS"] = str(omp_threads)
        current_env["OMP_PLACES"] = "cores"
        current_env["OMP_PROC_BIND"] = "close"

        # --- ВАЖНО: --bind-to none ---
        # Это предотвращает привязку MPI-процесса к одному ядру.
        # Если MPI привяжет процесс, все OpenMP потоки будут толпиться на одном ядре.
        cmd = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",
        "--bind-to", "none",
        "-np", str(mpi_procs),
        executable
        ]
        
        try:
            res = subprocess.run(cmd, env=current_env, capture_output=True, text=True)
            output = res.stdout.strip()
            
            # Ищем строку с результатами (последняя строка вывода)
            # Формат вывода C++: "MPI OMP TIME"
            time_sec = 0.0
            status = "FAIL"
            
            if res.returncode == 0 and output:
                lines = output.split('\n')
                # Ищем строку, которая начинается с числа процессов
                for line in reversed(lines):
                    parts = line.split()
                    if len(parts) >= 3 and parts[0] == str(mpi_procs):
                        try:
                            time_sec = float(parts[2])
                            status = "OK"
                            break
                        except ValueError:
                            continue
            
            if status == "OK":
                print(f"{mpi_procs:<5} | {omp_threads:<5} | {time_sec:<10.4f} | {status}")
                f.write(f"{mpi_procs},{omp_threads},{time_sec}\n")
            else:
                print(f"{mpi_procs:<5} | {omp_threads:<5} | {'N/A':<10} | {status}")
                print("Stderr:", res.stderr)

        except Exception as e:
            print(f"Error running {mpi_procs}x{omp_threads}: {e}")

print(f"\nDone! Results saved to {results_csv}")