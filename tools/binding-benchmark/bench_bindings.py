# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Microbenchmarks for Python binding overhead.

Measures operations whose cost is dominated by the Python<->C++ binding
layer (argument conversion, object wrapping, overload dispatch) rather
than by C++ kernels. Run the same script against a pybind11 build and a
nanobind build, then diff the JSON outputs with compare.py:

    pixi run python tools/binding-benchmark/bench_bindings.py -o pybind11.json
    # ... switch to nanobind build ...
    pixi run python tools/binding-benchmark/bench_bindings.py -o nanobind.json
    pixi run python tools/binding-benchmark/compare.py pybind11.json nanobind.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import timeit
from collections.abc import Callable

import numpy as np

import scipp as sc

REPEAT = 9


def _make_benchmarks() -> dict[str, Callable[[], object]]:
    small = sc.array(dims=['x'], values=np.arange(4.0), unit='m')
    small2 = small.copy()
    np_small = np.arange(4.0)
    np_large = np.arange(1_000_000.0)
    large = sc.array(dims=['x'], values=np_large, unit='m')
    scalar = sc.scalar(3.14, unit='m')
    da = sc.DataArray(small, coords={'x': small2.copy()})
    ds = sc.Dataset({'a': da.copy(), 'b': da.copy()})
    unit_m = sc.Unit('m')
    var_2d = sc.array(dims=['x', 'y'], values=np.arange(16.0).reshape(4, 4))

    unit_m2 = sc.Unit('m')
    str_var = sc.array(dims=['x'], values=['a', 'b', 'c'])

    def dataset_iter() -> list[object]:
        return list(ds.items())

    return {
        # --- construction ---
        'scalar_create': lambda: sc.scalar(1.0),
        'scalar_create_with_unit': lambda: sc.scalar(1.0, unit='m'),
        'scalar_create_kwargs': lambda: sc.scalar(1.0, unit='m', dtype='float64'),
        'variable_create_direct': lambda: sc.Variable(dims=['x'], values=np_small),
        'scalar_create_unit_obj': lambda: sc.scalar(1.0, unit=unit_m),
        'unit_create_from_str': lambda: sc.Unit('m/s'),
        'index_create': lambda: sc.index(5),
        'array_create_from_numpy_small': lambda: sc.array(dims=['x'], values=np_small),
        'array_create_from_list': lambda: sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
        'zeros_small': lambda: sc.zeros(dims=['x'], shape=[4]),
        'dataarray_create': lambda: sc.DataArray(small, coords={'x': small2}),
        'dataset_create': lambda: sc.Dataset({'a': da}),
        # --- attribute / property access ---
        'attr_unit': lambda: scalar.unit,
        'attr_dims': lambda: small.dims,
        'attr_shape': lambda: small.shape,
        'attr_sizes': lambda: small.sizes,
        'attr_ndim': lambda: small.ndim,
        'attr_dtype': lambda: small.dtype,
        'attr_value_scalar': lambda: scalar.value,
        'attr_values_small': lambda: small.values,
        'attr_values_large': lambda: large.values,
        'dataarray_data': lambda: da.data,
        'dataarray_coords_getitem': lambda: da.coords['x'],
        'dataset_getitem': lambda: ds['a'],
        'dataset_keys': lambda: list(ds.keys()),
        'dataset_iterate': dataset_iter,
        'iterate_str_variable': lambda: list(str_var.values),
        'unit_eq': lambda: unit_m == unit_m2,
        # --- indexing / slicing ---
        'getitem_label_int': lambda: small['x', 0],
        'getitem_label_slice': lambda: small['x', 0:2],
        'getitem_positional_int': lambda: small[0],
        'getitem_2d': lambda: var_2d['x', 0],
        'getitem_ellipsis': lambda: small[...],
        # --- small-data operations (dispatch dominated) ---
        'add_small': lambda: small + small2,
        'add_scalar_broadcast': lambda: small + scalar,
        'multiply_small': lambda: small * small2,
        'compare_eq_small': lambda: small == small2,
        'identical_small': lambda: sc.identical(small, small2),
        'sum_small': lambda: small.sum(),
        'copy_shallow': lambda: small.copy(deep=False),
        'copy_deep_small': lambda: small.copy(),
        'unit_conversion_scalar': lambda: scalar.to(unit='mm'),
        # --- numpy interop ---
        'asarray_small': lambda: np.asarray(small.values),
        'values_roundtrip_small': lambda: sc.array(dims=['x'], values=small.values),
        # --- formatting / misc ---
        'str_scalar': lambda: str(scalar),
        'sizeof': lambda: small.underlying_size(),
    }


def _make_inplace_benchmarks() -> dict[str, tuple[Callable[[], object], str]]:
    """Benchmarks with side effects: (callable, description)."""
    small = sc.array(dims=['x'], values=np.arange(4.0), unit='m')
    small2 = small.copy()
    np_small = np.arange(4.0)
    large = sc.array(dims=['x'], values=np.arange(1_000_000.0))
    np_large = np.arange(1_000_000.0)

    def iadd() -> None:
        nonlocal small
        small += small2

    def assign_values_small() -> None:
        small.values = np_small

    def assign_values_large() -> None:
        large.values = np_large

    return {
        'iadd_small': (iadd, 'in-place add, 4 elements'),
        'assign_values_small': (assign_values_small, 'values setter, 4 elements'),
        'assign_values_large': (assign_values_large, 'values setter, 1e6 elements'),
    }


def _time_one(func: Callable[[], object]) -> dict[str, float]:
    timer = timeit.Timer(func)
    n_loops, _ = timer.autorange()
    raw = timer.repeat(repeat=REPEAT, number=n_loops)
    per_op = sorted(t / n_loops for t in raw)
    return {
        'ns_min': per_op[0] * 1e9,
        'ns_median': per_op[len(per_op) // 2] * 1e9,
        'n_loops': n_loops,
        'repeats': REPEAT,
    }


def _binding_framework() -> str:
    """Detect which binding library built _scipp from its symbols."""
    from scipp import _scipp

    so_path = _scipp.__file__
    try:
        symbols = subprocess.run(  # noqa: S603
            ['nm', '-gU' if sys.platform == 'darwin' else '-gD', so_path],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if 'nanobind' in symbols or 'nb::' in symbols:
            return 'nanobind'
        if 'pybind11' in symbols:
            return 'pybind11'
        with open(so_path, 'rb') as f:
            blob = f.read()
        if b'nanobind' in blob:
            return 'nanobind'
        if b'pybind11' in blob:
            return 'pybind11'
    except OSError:
        pass
    return 'unknown'


def _import_time_ms() -> dict[str, float]:
    """Self-import time of the _scipp extension and total scipp import."""
    proc = subprocess.run(  # noqa: S603
        [sys.executable, '-X', 'importtime', '-c', 'import scipp'],
        capture_output=True,
        text=True,
        check=True,
    )
    ext_self_us = 0.0
    total_us = 0.0
    for line in proc.stderr.splitlines():
        # format: "import time: <self> | <cumulative> | <name>"
        parts = line.split('|')
        if len(parts) != 3:
            continue
        name = parts[2].strip()
        if name == 'scipp._scipp':
            ext_self_us = float(parts[0].split(':')[1])
        if name == 'scipp':
            total_us = float(parts[1])
    return {'ext_self_ms': ext_self_us / 1000, 'scipp_total_ms': total_us / 1000}


def _binary_sizes() -> dict[str, int]:
    from scipp import _scipp

    so_path = os.path.realpath(_scipp.__file__)
    sizes = {'_scipp.so': os.path.getsize(so_path)}
    lib_dir = os.path.join(os.path.dirname(so_path), '..', 'lib')
    if os.path.isdir(lib_dir):
        for name in sorted(os.listdir(lib_dir)):
            if 'scipp' in name and (name.endswith('.dylib') or name.endswith('.so')):
                sizes[name] = os.path.getsize(os.path.join(lib_dir, name))
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', default=None, help='write JSON here')
    parser.add_argument(
        '--quick', action='store_true', help='fewer repeats (smoke test)'
    )
    args = parser.parse_args()

    global REPEAT
    if args.quick:
        REPEAT = 3

    results: dict[str, dict[str, float]] = {}
    benchmarks = _make_benchmarks()
    inplace = _make_inplace_benchmarks()
    total = len(benchmarks) + len(inplace)
    for i, (name, func) in enumerate(benchmarks.items()):
        func()  # warm up / fail fast
        results[name] = _time_one(func)
        print(
            f'[{i + 1}/{total}] {name}: {results[name]["ns_min"]:.0f} ns',
            file=sys.stderr,
        )
    for j, (name, (func, _desc)) in enumerate(inplace.items()):
        func()
        results[name] = _time_one(func)
        print(
            f'[{len(benchmarks) + j + 1}/{total}] {name}: '
            f'{results[name]["ns_min"]:.0f} ns',
            file=sys.stderr,
        )

    report = {
        'meta': {
            'scipp_version': sc.__version__,
            'binding_framework': _binding_framework(),
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'platform': platform.platform(),
            'machine': platform.machine(),
        },
        'import_time': _import_time_ms(),
        'binary_sizes': _binary_sizes(),
        'results': results,
    }

    out = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(out + '\n')
        print(f'wrote {args.output}', file=sys.stderr)
    else:
        print(out)


if __name__ == '__main__':
    main()
