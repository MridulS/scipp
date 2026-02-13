# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest

import scipp as sc


@pytest.fixture
def var1():
    return sc.array(dims=['x'], values=[1, 2, 3, 4, 5])


@pytest.fixture
def var2():
    return sc.array(dims=['x'], values=[1, 2, 3, 4, 5])


def test_shallow_copy(benchmark, var1):
    benchmark(lambda: var1.copy(deep=False))


def test_deep_copy(benchmark, var1):
    benchmark(lambda: var1.copy())


def test_variable_inplace_operation(benchmark, var1, var2):
    def operation():
        v = var1.copy()
        v += var2
        return v

    benchmark(operation)


def test_variable_non_inplace_operation(benchmark, var1, var2):
    benchmark(lambda: var1 + var2)


@pytest.mark.parametrize("size", [10**5, 10**6, 10**7])
def test_assign_from_numpy(benchmark, size):
    array = np.arange(size, dtype=np.float64)
    var = sc.Variable(dims=['x'], values=array)

    def assign():
        var.values = array

    benchmark(assign)
