# Binding-overhead benchmarks

Measures the cost of the Python↔C++ binding layer of scipp, to evaluate the
pybind11 → nanobind migration. All operations here are dominated by argument
conversion, overload dispatch, and object wrapping — not by C++ kernels — so
they isolate what a binding-framework change can actually affect.

## Metrics

1. **Call overhead** (`bench_bindings.py`): ~49 microbenchmarks covering
   construction (`sc.scalar`, `sc.array`, `sc.Unit`, `sc.DataArray`),
   property access (`.unit`, `.dims`, `.values`, `.value`), slicing,
   small-array arithmetic, dict-like `Dataset`/coords access, and numpy
   round-trips. `assign_values_large` is included as a memcpy-bound
   control that should NOT change. Note `attr_values_large` is *not* a
   control: like `attr_values_small` it is dominated by nanobind's
   ndarray-to-numpy export, so both regress (~0.66x) regardless of array
   size — the data is never copied, only re-wrapped.
2. **Import time**: self-import time of the `_scipp` extension and total
   `import scipp` (captured automatically in the JSON).
3. **Binary size**: `_scipp.so` and the libscipp-* shared libraries
   (captured automatically; only `_scipp.so` contains binding code).
4. **Compile time**: clean rebuild of only the bindings target with ccache
   disabled (manual, see below).

## Protocol

On the baseline (pybind11) build:

```sh
pixi run python tools/binding-benchmark/bench_bindings.py \
    -o tools/binding-benchmark/results/pybind11-baseline.json

touch lib/python/*.cpp lib/python/*.h
CCACHE_DISABLE=1 /usr/bin/time -p pixi run ninja -C build _scipp
```

After switching the build to nanobind (same machine, same Python, release
config), repeat with `-o .../nanobind.json`, then:

```sh
pixi run python tools/binding-benchmark/compare.py \
    tools/binding-benchmark/results/pybind11-baseline.json \
    tools/binding-benchmark/results/nanobind.json
```

`compare.py` prints per-benchmark speedups, the geometric mean, and the
import-time/binary-size deltas. Benchmarks flagged `*` regressed by more
than 5 %.

The binding framework is auto-detected from symbols in `_scipp.so`, so the
JSON files are self-describing.

## Notes

- Use `--quick` for a fast smoke run (3 repeats instead of 9).
- Numbers are min-of-repeats per op; medians are also stored in the JSON.
- Quiet the machine (close browsers, disable background indexing) before
  recording numbers used in a decision.
- The existing `benchmarks/` (asv) suite measures end-to-end and
  kernel-bound performance across commits; this directory deliberately
  measures only the binding layer within a single working tree.
