# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest

import scipp as sc
from scipp.binning import make_binned


@pytest.mark.parametrize("nbin", [1024, 2048, 4096, 8192])
def test_bin2d_rebin_outer(benchmark, nbin):
    binned = sc.data.binned_x(nevent=2 * nbin, nbin=nbin)
    y = sc.linspace(dim='y', start=0, stop=1, num=2, unit='m')
    da = make_binned(binned, edges=[y])
    x = sc.linspace(dim='x', start=0, stop=1, num=nbin - 1, unit='m')

    benchmark(make_binned, da, edges=[x])


@pytest.mark.parametrize("nbin", [1024, 2048, 4096, 8192])
def test_bin2d_rebin_outer_transposed(benchmark, nbin):
    binned = sc.data.binned_x(nevent=2 * nbin, nbin=nbin)
    y = sc.linspace(dim='y', start=0, stop=1, num=2, unit='m')
    da = make_binned(binned, edges=[y])
    x = sc.linspace(dim='x', start=0, stop=1, num=nbin - 1, unit='m')

    benchmark(make_binned, da.transpose(), edges=[x])


@pytest.mark.parametrize("nbin", [1024, 2048, 4096, 8192])
def test_bin2d_rebin_outer_transposed_copied(benchmark, nbin):
    binned = sc.data.binned_x(nevent=2 * nbin, nbin=nbin)
    y = sc.linspace(dim='y', start=0, stop=1, num=2, unit='m')
    da = make_binned(binned, edges=[y])
    da_transposed = da.transpose().copy()
    x = sc.linspace(dim='x', start=0, stop=1, num=nbin - 1, unit='m')

    benchmark(make_binned, da_transposed, edges=[x])


@pytest.mark.parametrize("nbin", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_bin1d_table(benchmark, nbin):
    table = sc.data.table_xyz(50_000_000)
    x = sc.linspace(dim='x', start=0, stop=1, num=nbin + 1, unit='m')

    benchmark(make_binned, table, edges=[x])


@pytest.mark.parametrize("nevent,ngroup", [
    (1_000, 4),
    (1_000, 64),
    (1_000, 1024),
    (1_000_000, 4),
    (1_000_000, 64),
    (1_000_000, 1024),
    (10_000_000, 4),
    (10_000_000, 64),
])
def test_group_contiguous(benchmark, nevent, ngroup):
    table = sc.data.table_xyz(nevent)
    table.coords['group'] = (ngroup * table.coords['x']).to(dtype='int64')
    del table.coords['x']
    contiguous_groups = sc.arange('group', ngroup, unit='m')

    benchmark(table.group, contiguous_groups)


@pytest.mark.parametrize("nevent,ngroup", [
    (1_000, 4),
    (1_000, 64),
    (1_000, 1024),
    (1_000_000, 4),
    (1_000_000, 64),
    (1_000_000, 1024),
    (10_000_000, 4),
    (10_000_000, 64),
])
def test_group(benchmark, nevent, ngroup):
    table = sc.data.table_xyz(nevent)
    table.coords['group'] = (ngroup * table.coords['x']).to(dtype='int64')
    del table.coords['x']
    groups = sc.arange('group', ngroup, unit='m')
    groups.values[0] = 1
    groups.values[1] = 0

    benchmark(table.group, groups)
