# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc


class Binned1d:
    """
    Benchmark different aspects of binned data
    """
    params = (list(10**np.arange(8)), list(10**np.arange(0, 7)))
    param_names = ['nevent', 'nbin']

    def setup(self, nevent, nbin):
        self.da = sc.data.binned_x(nevent, nbin)

    def time_bins_constituents(self, nevent, nbin):
        self.da.bins.constituents

    def time_bins_size(self, nevent, nbin):
        self.da.bins.size()

    def time_bins_sum(self, nevent, nbin):
        self.da.bins.sum()

    def time_bins_mean(self, nevent, nbin):
        self.da.bins.sum()

    def time_bins_concat(self, nevent, nbin):
        self.da.bins.concat('x')


class Lookup:
    """
    Benchmark map operations using sc.lookup()
    """
    def setup(self):
        self.data = sc.data.binned_x(100_000_000, 10000)
        x = sc.linspace(dim='x', start=0.0, stop=1.0, num=1_000_001, unit='m')
        groups = sc.arange(dim='x', start=0, stop=1_000_000) // 1000 % 5
        self.hist_bool = sc.DataArray(data=groups.astype('bool'), coords={'x': x})
        self.hist_int = sc.DataArray(data=groups, coords={'x': x})
        self.hist_float = sc.DataArray(data=groups.astype('float64'), coords={'x': x})

    def time_create_bool(self):
        self.hist_bool.coords['x'].values[-1] *= 1.1
        sc.lookup(self.hist_bool, 'x')

    def time_create_float64(self):
        self.hist_float.coords['x'].values[-1] *= 1.1
        sc.lookup(self.hist_float, 'x')

    def time_create_int64(self):
        self.hist_int.coords['x'].values[-1] *= 1.1
        sc.lookup(self.hist_int, 'x')

    def time_map_bool(self):
        self.hist_float.coords['x'].values[-1] *= 1.1
        sc.lookup(self.hist_bool, 'x')[self.data.bins.coords['x']]

    def time_map_float64(self):
        self.hist_float.coords['x'].values[-1] *= 1.1
        sc.lookup(self.hist_float, 'x')[self.data.bins.coords['x']]

    def time_map_int64(self):
        self.hist_int.coords['x'].values[-1] *= 1.1
        sc.lookup(self.hist_int, 'x')[self.data.bins.coords['x']]

    def time_map_linspace_bool(self):
        sc.lookup(self.hist_bool, 'x')[self.data.bins.coords['x']]

    def time_map_linspace_int64(self):
        sc.lookup(self.hist_int, 'x')[self.data.bins.coords['x']]

    def time_map_linspace_float64(self):
        sc.lookup(self.hist_float, 'x')[self.data.bins.coords['x']]
