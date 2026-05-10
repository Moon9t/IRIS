#!/usr/bin/env python3
"""Performance gate: run benchmarks, record medians, compare to baselines."""
import argparse
import json
import os
import re
import subprocess
import sys
import statistics
from pathlib import Path

def run_cmd(cmd, timeout=120):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return p.returncode, p.stdout

def bench_once(iris, benchfile):
    cmd = [str(iris), 'bench', benchfile]
    rc, out = run_cmd(cmd, timeout=120)
    # Save raw output for debugging
    try:
        Path('target_verify').mkdir(parents=True, exist_ok=True)
        logf = Path('target_verify') / (Path(benchfile).stem + '.bench.log')
        logf.write_text(out)
    except Exception:
        pass

    # Expect bench to print a numeric result somewhere in the output; try to extract a float.
    float_re = re.compile(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
    lines = out.strip().splitlines()
    for line in reversed(lines):
        m = float_re.search(line)
        if m:
            try:
                return rc, float(m.group(1))
            except Exception:
                continue
    return rc, None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('bench', help='bench .iris file to run')
    p.add_argument('--iris', default=os.environ.get('IRIS_BIN','target\\debug\\iris.exe'))
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--baseline', default='engineering/v1/bench_baselines.json')
    p.add_argument('--threshold-pct', type=float, default=10.0, help='allowed percent regression')
    p.add_argument('--allow-record', action='store_true', help='allow recording a missing baseline')
    args = p.parse_args()

    iris = Path(args.iris)
    if not iris.exists():
        print(f"iris binary not found at {iris}")
        sys.exit(2)

    samples = []
    for i in range(args.runs):
        rc, val = bench_once(iris, args.bench)
        if rc != 0 or val is None:
            print(f"bench run {i} failed or returned no numeric value")
            sys.exit(3)
        samples.append(val)
        print(f"run {i}: {val}")

    median = statistics.median(samples)
    print(f"median: {median}")

    basefile = Path(args.baseline)
    baselines = {}
    if basefile.exists():
        baselines = json.loads(basefile.read_text())

    key = args.bench
    if key not in baselines:
        if args.allow_record:
            print(f"No baseline for {key}; recording {median}")
            baselines[key] = median
            basefile.parent.mkdir(parents=True, exist_ok=True)
            basefile.write_text(json.dumps(baselines, indent=2))
            print("Baseline saved. Rerun to enforce gate.")
            sys.exit(0)
        else:
            print(f"ERROR: no baseline for {key}; run with --allow-record to create one")
            sys.exit(2)

    baseline = baselines[key]
    # regression means median slower (larger) than baseline
    pct = (median - baseline) / baseline * 100.0
    print(f"baseline={baseline} median={median} change={pct:.2f}%")
    if pct > args.threshold_pct:
        print(f"PERF REGRESSION: {pct:.2f}% > {args.threshold_pct}%")
        sys.exit(1)
    else:
        print("Performance within threshold")
        # update baseline if faster
        if median < baseline:
            baselines[key] = median
            basefile.write_text(json.dumps(baselines, indent=2))
            print("Baseline updated to faster value")
        sys.exit(0)

if __name__ == '__main__':
    main()
