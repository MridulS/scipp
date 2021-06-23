# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @file
# @author Neil Vaytet, Simon Heybrock

from .. import config, transpose
from .model import PlotModel
from .resampling_model import resampling_model


class PlotModel2d(PlotModel):
    """
    Model class for 2 dimensional plots.
    """
    def __init__(self, *args, resolution=None, **kwargs):
        self._model = None
        super().__init__(*args, **kwargs)
        if resolution is None:
            self._resolution = [config.plot.height, config.plot.width]
        elif isinstance(resolution, int):
            self._resolution = [resolution, resolution]
        else:
            self._resolution = [resolution[ax] for ax in 'yx']
        self.dims = next(iter(self.data_arrays.values())).dims[-2:]

    def _dims_updated(self):
        self._model.bounds = {}
        self._model.resolution = {}
        for dim, resolution in zip(self.dims, self._resolution):
            self._model.resolution[dim] = resolution
            self._model.bounds[dim] = None

    def update(self):
        """
        Create or update the internal resampling model.
        """
        if self._model is None:
            self._model = resampling_model(self.data_arrays[self.name])
        else:
            self._model.update_array(self.data_arrays[self.name])

    def _update_image(self):
        """
        Resample 2d images to a fixed resolution to handle very large images.
        """
        self.dslice = self._model.data
        return transpose(self.dslice, dims=self._dims)

    def update_data(self, slices):
        """
        Slice the data along dimension sliders that are not disabled for all
        entries in the dict of data arrays.
        Then perform dynamic image resampling based on current viewport.
        """
        for dim, bounds in slices.items():
            self._model.bounds[dim] = bounds
        return self._update_image()

    def update_viewport(self, limits):
        """
        When an update to the viewport is requested on a zoom event, set new
        rebin edges and call for a resample of the image.
        """
        for dim, lims in limits.items():
            self._model.bounds[dim] = (lims[0], lims[1])
        return self._update_image()

    def reset_resampling_model(self):
        self._model.reset()
