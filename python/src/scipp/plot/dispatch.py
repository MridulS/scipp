# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

from .._scipp import core as sc
from .._bins import histogram
from .._utils import is_variable
import numpy as np


def dispatch(scipp_obj_dict,
             ndim=0,
             name=None,
             bins=None,
             projection=None,
             mpl_line_params=None,
             **kwargs):
    """
    Function to automatically dispatch the dict of DataArrays to the
    appropriate plotting function, depending on the number of dimensions.
    """

    if ndim < 1:
        raise RuntimeError("Invalid number of dimensions for "
                           "plotting: {}".format(ndim))

    # If the input is binned data, we histogram first
    for key, array in scipp_obj_dict.items():
        if sc.is_bins(array):
            if bins is not None:
                if is_variable(bins):
                    array = histogram(array.bins, bins)
                elif isinstance(bins, dict):
                    for dim, binning in bins.items():
                        coord = array.coords[dim]
                        if is_variable(binning):
                            edges = binning
                        elif isinstance(binning, int):
                            edges = sc.Variable([dim],
                                                values=np.linspace(
                                                    sc.nanmin(coord).value,
                                                    sc.nanmax(coord).value,
                                                    binning + 1),
                                                unit=coord.unit)
                        elif isinstance(binning, np.ndarray):
                            edges = sc.Variable([dim],
                                                values=binning,
                                                unit=coord.unit)
                        else:
                            raise RuntimeError(
                                "Unknown bins type: {}".format(binning))
                        array = histogram(array.bins, edges)
                else:
                    raise RuntimeError(
                        "bins must be either a Variable or a dict.")
                scipp_obj_dict[key] = array
            else:
                scipp_obj_dict[key] = array.bins.sum()

    # Dispatch the input to the different plotting functions
    if projection is None:
        if ndim < 3:
            projection = "{}d".format(ndim)
        else:
            projection = "2d"
    projection = projection.lower()

    if projection == "1d":
        from .plot1d import plot1d
        return plot1d(scipp_obj_dict,
                      mpl_line_params=mpl_line_params,
                      **kwargs)
    elif projection == "2d":
        from .plot2d import plot2d
        return plot2d(scipp_obj_dict, **kwargs)
    elif projection == "3d":
        from .plot3d import plot3d
        return plot3d(scipp_obj_dict, **kwargs)
    else:
        raise RuntimeError("Wrong projection type. Expected either '1d', "
                           "'2d', or '3d', got {}.".format(projection))
