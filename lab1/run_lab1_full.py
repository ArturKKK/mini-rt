import os
import subprocess
import re
import statistics
from datetime import datetime

executable = "../build/test/minirt-test"
args = ["2000", "2000", "1"]

schedules = ["static", "dynamic,1", "dynamic,10", "guided", "guided,10"]
threads_list = [1, 2, 4, 8, 16]
runs_per_config = 3

if not os.path.exists(executable):
    raise SystemExit(f"ОШИБКА: {executable} не найден. Собери проект в build/")

results_csv = "lab1_final_results.csv"
log_file = "lab1_runs.log"

print(f"{'Schedule':<15} | {'Threads':<8} | {'Median Time':<12} | {'Speedup':<8}")
print("-" * 55)

base_time = None

with open(results_csv, "w") as fcsv, open(log_file, "a") as flog:
    fcsv.write("Schedule,Threads,MedianTime,Speedup\n")

    for sched in schedules:
        for t in threads_list:
            times = []

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(t)
            env["OMP_SCHEDULE"] = sched
            # опционально уменьшает шум:
            env.setdefault("OMP_PROC_BIND", "true")
            env.setdefault("OMP_PLACES", "cores")

            for i in range(runs_per_config):
                ts = datetime.now().isoformat(timespec="seconds")
                res = subprocess.run([executable] + args, env=env, capture_output=True, text=True)

                flog.write(f"\n[{ts}] sched={sched} threads={t} run={i+1}\n")
                flog.write(res.stdout)
                flog.write(res.stderr)

                m = re.search(r"Rendering Time:\s*([0-9.]+)", res.stdout)
                if m:
                    times.append(float(m.group(1)))

            if not times:
                print(f"{sched:<15} | {t:<8} | FAILED")
                continue

            median_time = statistics.median(times)

            # База: 1 поток + static (важно, чтобы schedules начинались со static, threads — с 1)
            if base_time is None:
                base_time = median_time

            speedup = base_time / median_time
            print(f"{sched:<15} | {t:<8} | {median_time:<12.4f} | {speedup:<8.2f}")
            fcsv.write(f"{sched},{t},{median_time},{speedup}\n")

print(f"\nГотово! Результаты: {results_csv}, лог: {log_file}")
