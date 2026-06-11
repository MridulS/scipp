# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest

import scipp as sc


def test_operation_with_python_float() -> None:
    a = sc.scalar(1.23456789)
    b = 9.87654321
    assert (a / b).value == (a / sc.scalar(b)).value


def test_inplace_operation_with_python_float() -> None:
    a = sc.scalar(1.23456789)
    b = 9.87654321
    expected = a.value / b
    a /= b
    assert a.value == expected


def test_operation_with_python_int32() -> None:
    a = sc.scalar(3, dtype='int32')
    b = 2
    assert sc.identical(a + b, a + sc.scalar(b))


def test_reverse_operation_with_python_int32() -> None:
    a = sc.scalar(3, dtype='int32')
    b = 2
    assert sc.identical(b + a, sc.scalar(b) + a)
    assert sc.identical(b - a, sc.scalar(b) - a)
    assert sc.identical(b * a, sc.scalar(b) * a)
    assert sc.identical(b / a, sc.scalar(b) / a)


def test_inplace_operation_with_python_int32() -> None:
    a = sc.scalar(3, dtype='int32')
    b = 2
    expected = a.value + b
    a += b
    assert a.value == expected


def test_int_variable_plus_python_int_stays_int64() -> None:
    a = sc.scalar(1, dtype='int64')
    assert (a + 1).dtype == sc.DType.int64


def test_int_variable_plus_numpy_int_stays_int64() -> None:
    # The headline regression: nanobind's exact-match dispatch pass rejects
    # numpy scalars, so without explicit handling np.int64 would fall through
    # to the float64 operator overload.
    a = sc.scalar(1, dtype='int64')
    assert (a + np.int64(1)).dtype == sc.DType.int64
    assert (a + np.int32(1)).dtype == sc.DType.int64


def test_int_variable_plus_python_bool_stays_int64() -> None:
    a = sc.scalar(1, dtype='int64')
    assert (a + True).dtype == sc.DType.int64


def test_int_variable_plus_float_is_float64() -> None:
    a = sc.scalar(1, dtype='int64')
    assert (a + 1.5).dtype == sc.DType.float64
    assert (a + np.float64(1)).dtype == sc.DType.float64


def test_int_variable_plus_numpy_float32_is_float64() -> None:
    # np.float32 is not a Python float subclass and has no __index__, so it
    # falls to the float64 fallback (no float32 scalar overload exists).
    a = sc.scalar(1, dtype='int64')
    assert (a + np.float32(1)).dtype == sc.DType.float64


def test_int_variable_plus_oversized_int_is_float64() -> None:
    # Too large for int64: falls through to the float64 path, as before.
    a = sc.scalar(1, dtype='int64')
    assert (a + 2**70).dtype == sc.DType.float64


def test_reverse_and_inplace_scalar_ops_preserve_dtype() -> None:
    a = sc.scalar(2, dtype='int64')
    assert (np.int64(1) + a).dtype == sc.DType.int64
    assert (1 - a).dtype == sc.DType.int64
    a += np.int64(3)
    assert a.dtype == sc.DType.int64
    assert a.value == 5


def test_comparison_with_numpy_int_scalar() -> None:
    a = sc.array(dims=['x'], values=[1, 2], dtype='int64')
    result = a == np.int64(2)
    assert result.dtype == sc.DType.bool
    assert list(result.values) == [False, True]


def test_scalar_op_with_non_number_returns_notimplemented() -> None:
    # Returning NotImplemented (rather than raising) lets Python try the
    # reflected operation and ultimately raise a clean TypeError.
    assert sc.scalar(1).__add__(object()) is NotImplemented
    with pytest.raises(TypeError):
        _ = sc.scalar(1) + object()  # type: ignore[operator]
