Tools for v1 automation:

- `iris_diff_runner.py` — run a set of `.iris` files under interpreter and native backends and report diffs.
- `iris_perf_gate.py` — run a benchmark repeatedly, record median baseline, and fail on regressions.

Usage examples:

Windows PowerShell:
```powershell
python tools\iris_diff_runner.py examples\hello.iris --iris target\debug\iris.exe
python tools\iris_perf_gate.py benches\binary_search_bench.iris --iris target\debug\iris.exe --runs 5
```

Set `IRIS_BIN` to override the default `target\debug\iris.exe`.
