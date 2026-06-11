# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Compare two bench_bindings.py JSON outputs.

pixi run python tools/binding-benchmark/compare.py pybind11.json nanobind.json
"""

from __future__ import annotations

import argparse
import json


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', help='JSON from the reference build')
    parser.add_argument('candidate', help='JSON from the new build')
    args = parser.parse_args()

    base = _load(args.baseline)
    cand = _load(args.candidate)

    b_name = base['meta'].get('binding_framework', 'baseline')
    c_name = cand['meta'].get('binding_framework', 'candidate')

    print(f'baseline:  {b_name}  (scipp {base["meta"]["scipp_version"]})')
    print(f'candidate: {c_name}  (scipp {cand["meta"]["scipp_version"]})')
    print()

    header = (
        f'{"benchmark":<32} {b_name + " ns":>14} {c_name + " ns":>14} {"speedup":>9}'
    )
    print(header)
    print('-' * len(header))
    speedups = []
    for name, b in base['results'].items():
        c = cand['results'].get(name)
        if c is None:
            print(f'{name:<32} {b["ns_min"]:>14.0f} {"missing":>14}')
            continue
        ratio = b['ns_min'] / c['ns_min'] if c['ns_min'] else float('inf')
        speedups.append(ratio)
        flag = '  *' if ratio < 0.95 else ''
        print(
            f'{name:<32} {b["ns_min"]:>14.0f} {c["ns_min"]:>14.0f} {ratio:>8.2f}x{flag}'
        )

    if speedups:
        speedups.sort()
        geomean = 1.0
        for s in speedups:
            geomean *= s
        geomean **= 1.0 / len(speedups)
        print('-' * len(header))
        print(
            f'{"geometric mean":<32} {"":>14} {"":>14} {geomean:>8.2f}x   '
            f'(median {speedups[len(speedups) // 2]:.2f}x, * = regression >5%)'
        )

    print()
    print('import time (ms):')
    for key in base.get('import_time', {}):
        b_ms = base['import_time'][key]
        c_ms = cand.get('import_time', {}).get(key)
        if c_ms is not None:
            print(f'  {key:<24} {b_ms:>10.1f} {c_ms:>10.1f} {b_ms / c_ms:>8.2f}x')

    print()
    print('binary sizes (MB):')
    for key in base.get('binary_sizes', {}):
        b_sz = base['binary_sizes'][key] / 1e6
        c_sz = cand.get('binary_sizes', {}).get(key)
        if c_sz is not None:
            ratio = b_sz / (c_sz / 1e6)
            print(f'  {key:<24} {b_sz:>10.2f} {c_sz / 1e6:>10.2f} {ratio:>8.2f}x')


if __name__ == '__main__':
    main()
