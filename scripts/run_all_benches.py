import os
import subprocess
import re

benches_dir = "benches"
iris_exe = r"target\release\iris.exe"

if not os.path.exists(iris_exe):
    # Try cargo run --release as fallback
    iris_cmd = ["cargo", "run", "--release", "--"]
else:
    iris_cmd = [iris_exe]

# Get all bench files sorted
bench_files = sorted([f for f in os.listdir(benches_dir) if f.endswith("_bench.iris")])

print(f"Found {len(bench_files)} benchmark files.")

results = {}

for bf in bench_files:
    name = bf.replace("_bench.iris", "")
    bf_path = os.path.join(benches_dir, bf)
    print(f"Running benchmark for {name}...")
    
    cmd = iris_cmd + ["bench", bf_path]
    # Run the benchmark
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        output = res.stderr # iris bench prints output to stderr
        if not output or "Phase" not in output:
            output = res.stdout
        
        # Parse the statistics
        # Format is:
        #   Phase          Min          Mean         Median       Max          StdDev
        #   Parse          22.0µs       35.1µs       36.5µs       49.0µs       8.2µs
        #   Compile        151.0µs      235.2µs      249.5µs      292.0µs      48.6µs
        #   Eval           27.383s      28.541s      28.148s      32.368s      1.408s
        #   Total          27.383s      28.541s      28.149s      32.368s      1.408s
        #   throughput: 0 iterations/sec
        
        phase_data = {}
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'\s+', line)
            if parts[0] in ["Parse", "Compile", "Eval", "Total"]:
                # Parse, Min, Mean, Median, Max, StdDev
                phase_data[parts[0]] = {
                    "min": parts[1],
                    "mean": parts[2],
                    "median": parts[3],
                    "max": parts[4],
                    "stddev": parts[5]
                }
            elif "throughput:" in line:
                m = re.search(r'throughput:\s+([0-9\.]+)\s+iterations/sec', line)
                if m:
                    phase_data["throughput"] = m.group(1)
        
        results[name] = phase_data
    except Exception as e:
        print(f"Failed to run {name}: {e}")

# Generate markdown table output
print("\n### Performance Matrix")
print("| Benchmark | Parse (Mean) | Compile (Mean) | Execution (Mean) | Total Time (Mean) | Throughput (it/s) |")
print("|---|---|---|---|---|---|")
for name in sorted(results.keys()):
    r = results[name]
    parse = r.get("Parse", {}).get("mean", "N/A")
    compile_t = r.get("Compile", {}).get("mean", "N/A")
    exec_t = r.get("Eval", {}).get("mean", "N/A")
    total = r.get("Total", {}).get("mean", "N/A")
    tp = r.get("throughput", "0")
    print(f"| **{name}** | {parse} | {compile_t} | {exec_t} | {total} | {tp} |")

print("\n### Phase Times Breakdown")
print("| Benchmark | Min (Total) | Median (Total) | Max (Total) | StdDev (Total) |")
print("|---|---|---|---|---|")
for name in sorted(results.keys()):
    r = results[name]
    total_min = r.get("Total", {}).get("min", "N/A")
    total_med = r.get("Total", {}).get("median", "N/A")
    total_max = r.get("Total", {}).get("max", "N/A")
    total_std = r.get("Total", {}).get("stddev", "N/A")
    print(f"| **{name}** | {total_min} | {total_med} | {total_max} | {total_std} |")
