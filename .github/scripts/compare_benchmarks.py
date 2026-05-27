#!/usr/bin/env python3
import json
import sys

def load(path):
    with open(path) as f:
        return {b['name']: b['mean_ms'] for b in json.load(f).get('benchmarks', [])}

def main():
    if len(sys.argv) != 3:
        print('Usage: compare_benchmarks.py <baseline.json> <current.json>')
        return 2
    baseline_path, current_path = sys.argv[1], sys.argv[2]
    baseline = load(baseline_path)
    current = load(current_path)
    regression = False
    for name, cur in sorted(current.items()):
        base = baseline.get(name)
        if base and base > 0:
            pct = ((cur - base) / base) * 100
            status = 'REGRESSION' if pct > 15 else ('improved' if pct < -5 else 'ok')
            if status == 'REGRESSION':
                regression = True
            print(f'{name:30s} base={base:8.2f}ms now={cur:8.2f}ms {pct:+.1f}% [{status}]')
        else:
            print(f'{name:30s} new benchmark now={cur:.2f}ms')

    if regression:
        print('\nWARNING: Performance regression detected (>15% slower)')
        print('Review the benchmarks above before merging.')
        return 1
    else:
        print('\nAll benchmarks within acceptable range.')
        return 0

if __name__ == '__main__':
    sys.exit(main())
