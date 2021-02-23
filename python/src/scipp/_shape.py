# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Andrew
from ._scipp import core as _cpp
from ._cpp_wrapper_util import call_func as _call_cpp_func
from typing import Sequence


def broadcast(x, dims, shape):
    """Broadcast a variable.

    Note that scipp operations broadcast automatically, so using this function
    directly is rarely required.

    :param x: Variable to broadcast.
    :param dims: List of new dimensions.
    :param shape: New extents in each dimension.
    :type x: Variable
    :type dims: list[str]
    :type shape: list[int]
    :return: New variable with requested dimension labels and shape.
    """
    return _call_cpp_func(_cpp.broadcast, x, dims, shape)


def concatenate(x, y, dim):
    """Concatenate input data array along the given dimension.

    Concatenation can happen in two ways:

    - Along an existing dimension, yielding a new dimension extent
      given by the sum of the input's extents.
    - Along a new dimension that is not contained in either of the inputs,
      yielding an output with one extra dimensions.

    In the case of a data array or dataset, the coords, and masks are also
    concatenated.
    Coords, and masks for any but the given dimension are required to match
    and are copied to the output without changes.

    :param x: Left hand side input.
    :param y: Right hand side input.
    :param dim: Dimension along which to concatenate.
    :type x: Dataset, DataArray or Variable. Must be same type as y
    :type y: Dataset, DataArray or Variable. Must be same type as x
    :type dim: str
    :raises: If the dtype or unit does not match, or if the
             dimensions and shapes are incompatible.
    :return: The absolute values of the input.
    """
    return _call_cpp_func(_cpp.concatenate, x, y, dim)


def reshape(x, dims, to_dims=None):
    """Reshape a variable, data array or dataset.

    :param x: Container to reshape.
    :param dims: List of new dimensions.
    :param shape: New extents in each dimension.
    :type x: Dataset, DataArray or Variable
    :type dims: list[str]
    :type shape: list[int]
    :raises: If the volume of the old shape is not equal to the
             volume of the new shape.
    :return: New Dataset, DataArray or Variable with requested dimension labels
             and shape.
    """
    if to_dims is not None:
        return _call_cpp_func(_cpp.reshape, x, dims, to_dims)
    else:
        return _call_cpp_func(_cpp.reshape, x, dims)


def transpose(x, dims: Sequence[str]):
    """Transpose dimensions of a variable.

    :param x: Variable to transpose.
    :param dims: List of dimensions in desired order. If default,
                        reverses existing order.
    :type x: Variable
    :type dims: list[str]
    :raises: If the dtype or unit does not match, or if the
             dimensions and shapes are incompatible.
    :return: The absolute values of the input.
    """
    return _call_cpp_func(_cpp.transpose, x, dims)


# def stack(x, dim, to_dims):
#     """Reshape a variable, data array or dataset.

#     :param x: Container to reshape.
#     :param dims: List of new dimensions.
#     :param shape: New extents in each dimension.
#     :type x: Dataset, DataArray or Variable
#     :type dims: list[str]
#     :type shape: list[int]
#     :raises: If the volume of the old shape is not equal to the
#              volume of the new shape.
#     :return: New Dataset, DataArray or Variable with requested dimension labels
#              and shape.
#     """
#     return _call_cpp_func(_cpp.stack, x, dim, to_dims)

# def unstack(x, dims, to_dim):
#     """Reshape a variable, data array or dataset.

#     :param x: Container to reshape.
#     :param dims: List of new dimensions.
#     :param shape: New extents in each dimension.
#     :type x: Dataset, DataArray or Variable
#     :type dims: list[str]
#     :type shape: list[int]
#     :raises: If the volume of the old shape is not equal to the
#              volume of the new shape.
#     :return: New Dataset, DataArray or Variable with requested dimension labels
#              and shape.
#     """
#     return _call_cpp_func(_cpp.unstack, x, dims, to_dim)
