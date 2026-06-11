# Migrating after the switch to nanobind

## Overview

Scipp's Python bindings were re-implemented using [nanobind](https://nanobind.readthedocs.io/) (previously [pybind11](https://pybind11.readthedocs.io/)).
This is an internal change and the vast majority of code is unaffected.
However, nanobind is stricter about a few argument conversions that pybind11 performed automatically, so a small number of calls that previously worked may now raise a `TypeError`.

This page lists the user-visible changes and how to update your code.

## NumPy booleans are no longer accepted where a Python `bool` is expected

Where a function expects a Python `bool`, a NumPy boolean such as `np.True_` (i.e. `np.bool_`) is no longer accepted; nanobind's `bool` converter requires an exact Python `bool`.

```python
# No longer works:
sc.array(dims=['x'], values=[1, 2], unit='m').sum(np.True_)

# Do this instead:
... .sum(True)
```

If a NumPy boolean reaches such an argument, pass `bool(value)` (or use the Python literals `True`/`False`).

## `bytes` are no longer decoded to `str`

`bytes` objects are no longer automatically decoded to `str`.
This affects places that expect a string, such as dictionary keys, dimension names, and unit strings.

```python
# No longer works:
sc.array(dims=[b'x'], values=[1, 2])
sc.scalar(1, unit=b'm')

# Do this instead:
sc.array(dims=['x'], values=[1, 2])
sc.scalar(1, unit='m')
```

Decode explicitly with `value.decode()` if you are starting from `bytes`.

## Arithmetic with NumPy scalar operands

Arithmetic between a Scipp object and a NumPy scalar now follows NumPy's dtype rules consistently.
In particular, an integer Scipp variable combined with a NumPy integer scalar keeps its integer dtype:

```python
var = sc.array(dims=['x'], values=[1, 2], dtype='int64')
(var + np.int64(1)).dtype  # int64
```

This matches the behavior from before the migration (an intermediate version incorrectly promoted such results to `float64`; that has been fixed).

Note that, consistent with NumPy ≥ 2, a NumPy boolean operand such as `np.True_` resolves to `float64` rather than to an integer, because NumPy removed `np.bool_.__index__`.
Use a Python `int` or an explicit `np.int64` if you need an integer result.
