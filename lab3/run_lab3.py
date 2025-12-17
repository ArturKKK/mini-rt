import os
import subprocess
import shutil
import re

# --- НАСТРОЙКИ ---
build_dir = "build"
executable = f"./{build_dir}/matmul_cuda"
sizes = [1024, 2048, 4096, 8192] # 8192 - это ~256MB под матрицы, V100 съест легко
results_csv = "lab3_results.csv"

print(">>> Configuring and Building with CMake (CUDA)...")

if os.path.exists(build_dir):
    shutil.rmtree(build_dir)
os.makedirs(build_dir)

env = os.environ.copy()

try:
    # CMake для CUDA сам найдет nvcc, если он есть в PATH
    subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir, check=True, env=env)
    subprocess.run(["cmake", "--build", ".", "-j", str(os.cpu_count())], cwd=build_dir, check=True, env=env)
except subprocess.CalledProcessError as e:
    print(f"BUILD FAILED: {e}")
    print("Hint: Make sure 'nvcc' is installed (conda install -c nvidia cuda-nvcc)")
    exit(1)

if not os.path.exists(executable):
    print(f"ERROR: {executable} not found!")
    exit(1)

# Хак для Conda: добавляем путь к библиотекам в LD_LIBRARY_PATH перед запуском
# Иначе программа может не найти libcublas.so
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    lib_path = os.path.join(conda_prefix, "lib")
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld}"

print("\n>>> Starting Benchmarks...")
print(f"{'Size':<8} | {'Naive (s)':<10} | {'cuBLAS (s)':<10} | {'Speedup':<8} | {'Status':<8}")
print("-" * 55)

with open(results_csv, "w") as f:
    f.write("Size,Naive_Time,cuBLAS_Time,Speedup\n")

    for N in sizes:
        try:
            # Запуск
            res = subprocess.run([executable, str(N)], env=env, capture_output=True, text=True)
            output = res.stdout
            
            # Парсинг
            naive_m = re.search(r"Naive GPU Time:\s+([0-9.]+)", output)
            cublas_m = re.search(r"cuBLAS GPU Time:\s+([0-9.]+)", output)
            verify_m = re.search(r"Verification:.*\[(OK|FAIL)\]", output)
            
            if naive_m and cublas_m:
                t_naive = float(naive_m.group(1))
                t_cublas = float(cublas_m.group(1))
                speedup = t_naive / t_cublas
                status = verify_m.group(1) if verify_m else "?"
                
                print(f"{N:<8} | {t_naive:<10.4f} | {t_cublas:<10.4f} | {speedup:<8.2f} | {status:<8}")
                f.write(f"{N},{t_naive},{t_cublas},{speedup}\n")
            else:
                print(f"{N:<8} | FAILED TO PARSE")
                print(res.stderr)

        except Exception as e:
            print(f"Error N={N}: {e}")

print(f"\nDone! Results -> {results_csv}")
