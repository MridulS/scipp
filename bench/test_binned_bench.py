# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest

import scipp as sc
from scipp.binning import make_binned


@pytest.mark.parametrize("nevent,nbin", [
    (1, 1),
    (10, 1),
    (100, 1),
    (1000, 10),
    (10000, 100),
    (100000, 1000),
])
def test_bins_constituents(benchmark, nevent, nbin):
    da = sc.data.binned_x(nevent, nbin)
    benchmark(lambda: da.bins.constituents)


@pytest.mark.parametrize("nevent,nbin", [
    (1, 1),
    (10, 1),
    (100, 1),
    (1000, 10),
    (10000, 100),
    (100000, 1000),
])
def test_bins_size(benchmark, nevent, nbin):
    da = sc.data.binned_x(nevent, nbin)
    benchmark(da.bins.size)


@pytest.mark.parametrize("nevent,nbin", [
    (1, 1),
    (10, 1),
    (100, 1),
    (1000, 10),
    (10000, 100),
    (100000, 1000),
])
def test_bins_sum(benchmark, nevent, nbin):
    da = sc.data.binned_x(nevent, nbin)
    benchmark(da.bins.sum)


@pytest.mark.parametrize("nevent,nbin", [
    (1, 1),
    (10, 1),
    (100, 1),
    (1000, 10),
    (10000, 100),
    (100000, 1000),
])
def test_bins_mean(benchmark, nevent, nbin):
    da = sc.data.binned_x(nevent, nbin)
    benchmark(da.bins.mean)


@pytest.mark.parametrize("nevent,nbin", [
    (1, 1),
    (10, 1),
    (100, 1),
    (1000, 10),
    (10000, 100),
    (100000, 1000),
])
def test_bins_concat(benchmark, nevent, nbin):
    da = sc.data.binned_x(nevent, nbin)
    benchmark(da.bins.concat, 'x')


@pytest.mark.parametrize("nbin", [1, 2, 4, 8, 16, 32, 64, 128])
def test_binned2d_concat(benchmark, nbin):
    nx = 100000
    binned = sc.data.binned_x(nevent=2 * nx, nbin=nx)
    y = sc.linspace(dim='y', start=0, stop=1, num=nbin + 1, unit='m')
    da = make_binned(binned, edges=[y])

    benchmark(da.bins.concat, 'x')


@pytest.mark.parametrize("nbin", [1024, 2048, 4096, 8192])
def test_binned2d_concat_inner(benchmark, nbin):
    binned = sc.data.binned_x(nevent=2 * nbin, nbin=nbin)
    y = sc.linspace(dim='y', start=0, stop=1, num=2, unit='m')
    da = make_binned(binned, edges=[y])

    benchmark(da.bins.concat, 'y')


def test_lookup_create_bool(benchmark):
    x = sc.linspace(dim='x', start=0.0, stop=1.0, num=1_000_001, unit='m')
    groups = sc.arange(dim='x', start=0, stop=1_000_000) // 1000 % 5
    hist_bool = sc.DataArray(data=groups.astype('bool'), coords={'x': x})

    def create():
        hist_bool.coords['x'].values[-1] *= 1.1
        return sc.lookup(hist_bool, 'x')

    benchmark(create)


def test_lookup_create_float64(benchmark):
    x = sc.linspace(dim='x', start=0.0, stop=1.0, num=1_000_001, unit='m')
    groups = sc.arange(dim='x', start=0, stop=1_000_000) // 1000 % 5
    hist_float = sc.DataArray(data=groups.astype('float64'), coords={'x': x})

    def create():
        hist_float.coords['x'].values[-1] *= 1.1
        return sc.lookup(hist_float, 'x')

    benchmark(create)


def test_lookup_create_int64(benchmark):
    x = sc.linspace(dim='x', start=0.0, stop=1.0, num=1_000_001, unit='m')
    groups = sc.arange(dim='x', start=0, stop=1_000_000) // 1000 % 5
    hist_int = sc.DataArray(data=groups, coords={'x': x})

    def create():
        hist_int.coords['x'].values[-1] *= 1.1
        return sc.lookup(hist_int, 'x')

    benchmark(create)


def test_lookup_map_bool(benchmark):
    binned_x = sc.data.binned_x(100_000_000, 10000).bins.coords['x']
    x = sc.linspace(dim='x', start=0.0, stop=1.0, num=1_000_001, unit='m')
    groups = sc.arange(dim='x', start=0, stop=1_000_000) // 1000 % 5
    hist_bool = sc.DataArray(data=groups.astype('bool'), coords={'x': x})

    def map_lookup():
        hist_bool.coords['x'].values[-1] *= 1.1
        return sc.lookup(hist_bool, 'x')[binned_x]

    benchmark(map_lookup)


def test_lookup_map_float64(benchmark):
    binned_x = sc.data.binned_x(100_000_000, 10000).bins.coords['x']
    x = sc.linspace(dim='x', start=0.0, stop=1.0, num=1_000_001, unit='m')
    groups = sc.arange(dim='x', start=0, stop=1_000_000) // 1000 % 5
    hist_float = sc.DataArray(data=groups.astype('float64'), coords={'x': x})

    def map_lookup():
        hist_float.coords['x'].values[-1] *= 1.1
        return sc.lookup(hist_float, 'x')[binned_x]

    benchmark(map_lookup)


def test_lookup_map_int64(benchmark):
    binned_x = sc.data.binned_x(100_000_000, 10000).bins.coords['x']
    x = sc.linspace(dim='x', start=0.0, stop=1.0, num=1_000_001, unit='m')
    groups = sc.arange(dim='x', start=0, stop=1_000_000) // 1000 % 5
    hist_int = sc.DataArray(data=groups, coords={'x': x})

    def map_lookup():
        hist_int.coords['x'].values[-1] *= 1.1
        return sc.lookup(hist_int, 'x')[binned_x]

    benchmark(map_lookup)
