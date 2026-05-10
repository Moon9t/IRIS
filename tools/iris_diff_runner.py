#!/usr/bin/env python3
"""Differential runner: run examples under interpreter and native backends and diff outputs."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, cwd=None, timeout=60):
    try:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return 124, "<timeout>"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+', help='.iris files to test')
    p.add_argument('--iris', default=os.environ.get('IRIS_BIN','target\\debug\\iris.exe'), help='path to iris binary')
    p.add_argument('--workdir', default='target_verify', help='directory to store outputs')
    args = p.parse_args()

    iris = Path(args.iris)
    if not iris.exists():
        print(f"iris binary not found at {iris}. Build with `cargo build` or set IRIS_BIN.")
        sys.exit(2)

    outdir = Path(args.workdir)
    outdir.mkdir(parents=True, exist_ok=True)

    failures = 0

    for f in args.files:
        name = Path(f).stem
        interp_cmd = [str(iris), '--emit', 'eval', f]
        native_cmd = [str(iris), 'run', f]

        print(f"Running interpreter: {' '.join(interp_cmd)}")
        rc_i, out_i = run_cmd(interp_cmd, timeout=30)
        (outdir / f"{name}.interp.out").write_text(out_i)

        print(f"Running native: {' '.join(native_cmd)}")
        rc_n, out_n = run_cmd(native_cmd, timeout=60)
        (outdir / f"{name}.native.out").write_text(out_n)

        # Normalize outputs: ignore build log lines like 'wrote binary:'.
        def normalize(s):
            lines = [l for l in s.splitlines() if not l.strip().lower().startswith('wrote binary:')]
            # strip trailing numeric-only lines (common interpreter exit prints)
            while lines and lines[-1].strip().isdigit():
                lines.pop()
            return '\n'.join(lines).strip()

        n_i = normalize(out_i)
        n_n = normalize(out_n)

        if rc_i != 0 or rc_n != 0:
            print(f"[WARN] non-zero exit: interp={rc_i} native={rc_n} for {f}")

        if n_i != n_n:
            print(f"[DIFF] {f}: outputs differ; see {outdir}/{name}.interp.out and {outdir}/{name}.native.out")
            failures += 1
        else:
            print(f"[OK] {f}: outputs match")

    if failures:
        print(f"Differential test failed: {failures} files differ")
        sys.exit(1)
    print("All differential tests passed")

if __name__ == '__main__':
    main()
