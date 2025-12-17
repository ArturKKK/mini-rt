import os
import subprocess
import re
import shutil

# --- НАСТРОЙКИ ---
build_dir = "build"
executable = f"./{build_dir}/poisson_mpi"
# Список процессов для теста. 
# Если ядер на машине 8, то 16 запускать не стоит (будет тормозить).
procs_list = [1, 2, 4, 8] 
results_csv = "lab2_results.csv"

# 1. Очистка и Сборка
print(">>> Building project with CMake...")

if os.path.exists(build_dir):
    shutil.rmtree(build_dir)
os.makedirs(build_dir)

env = os.environ.copy()

try:
    # Конфигурация и сборка
    subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir, check=True, env=env)
    # -j включает многопоточную компиляцию
    subprocess.run(["cmake", "--build", ".", "-j", str(os.cpu_count())], cwd=build_dir, check=True, env=env)
except subprocess.CalledProcessError as e:
    print(f"BUILD FAILED: {e}")
    exit(1)

if not os.path.exists(executable):
    print(f"ERROR: Executable {executable} not found!")
    exit(1)

print("\n>>> Starting Benchmarks...")
print(f"{'Procs':<8} | {'Dims':<8} | {'Time (sec)':<12} | {'Speedup':<8}")
print("-" * 45)

base_time = None

with open(results_csv, "w") as f:
    f.write("Processes,Time,Speedup\n")

    for p in procs_list:
        # Запуск mpirun. --oversubscribe убрали для чистоты.
        cmd = ["mpirun", "-np", str(p), executable]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            output = res.stdout
            
            # Ищем время и топологию
            time_match = re.search(r"Time:\s+([0-9.]+)", output)
            dims_match = re.search(r"Processes:\s+\d+\s+\((.+)\)", output)
            
            if time_match:
                time_sec = float(time_match.group(1))
                dims_str = dims_match.group(1) if dims_match else "?x?"
                
                if p == 1:
                    base_time = time_sec
                    speedup = 1.0
                else:
                    speedup = base_time / time_sec if base_time else 0.0
                
                print(f"{p:<8} | {dims_str:<8} | {time_sec:<12.4f} | {speedup:<8.2f}")
                f.write(f"{p},{time_sec},{speedup}\n")
            else:
                print(f"{p:<8} | FAILED TO PARSE OUTPUT")
                print(res.stderr)

        except Exception as e:
            print(f"Error running {p} procs: {e}")

print(f"\nDone! Results saved to {results_csv}")
