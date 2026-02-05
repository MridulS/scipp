# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Internal utilities for numpy array lifetime management."""

import numpy as np


class _ArrayWithBase(np.ndarray):
    """NumPy array subclass that holds a reference to keep the data owner alive."""

    def __new__(cls, input_array, owner=None):
        # Create a view of the input array as our subclass
        obj = np.asarray(input_array).view(cls)
        obj._scipp_owner = owner
        return obj

    def __array_finalize__(self, obj):
        # Called when array is created from template (slicing, etc.)
        # Preserve the owner reference through views
        if obj is None:
            return
        self._scipp_owner = getattr(obj, '_scipp_owner', None)

    def __reduce__(self):
        # For pickling, convert back to regular ndarray
        return (np.asarray, (np.asarray(self),))


def set_array_base(arr, owner):
    """
    Create a numpy array that keeps the owner alive.

    Parameters
    ----------
    arr : numpy.ndarray
        The array whose lifetime should depend on owner.
    owner : object
        The object that owns the underlying data buffer.

    Returns
    -------
    numpy.ndarray
        An array subclass that keeps owner alive.
    """
    return _ArrayWithBase(arr, owner)
