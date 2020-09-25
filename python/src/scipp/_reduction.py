# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from ._scipp import core as _cpp
from ._cpp_wrapper_util import call_func as _call_cpp_func


def flatten(x, dim):
    """Flatten the specified dimension into event lists equivalent to
    summing dense data.

    :param x: Input data.
    :param dim: Dimension to flatten over.
    :raises: If the dimension does not exist, or if x does not contain event
             lists.
    :return: The flattened data.
    :seealso: :py:func:`scipp.sum` for regular dense data.
    """
    return _call_cpp_func(_cpp.flatten, x, dim)


def mean(x, dim, out=None):
    """Element-wise mean over the specified dimension, if variances are
    present, the new variance is computated as standard-deviation of the mean.

    If the input has variances, the variances stored in the ouput are based on
    the "standard deviation of the mean", i.e.,
    :math:`\\sigma_{mean} = \\sigma / \\sqrt{N}`.
    :math:`N` is the length of the input dimension.
    :math:`sigma` is estimated as the average of the standard deviations of
    the input elements along that dimension.

    :param x: Input data.
    :param dim: Dimension along which to calculate the mean.
    :param out: Optional output buffer.
    :raises: If the dimension does not exist, or the dtype cannot be summed,
             e.g., if it is a string.
    :return: The mean of the input values.
    """
    return _call_cpp_func(_cpp.mean, x, dim, out=out)


def sum(x, dim, out=None):
    """Element-wise sum over the specified dimension.

    :param x: Input data.
    :param dim: Dimension along which to calculate the sum.
    :param out: Optional output buffer.
    :raises: If the dimension does not exist, or the dtype cannot be summed,
             e.g., if it is a string.
    :return: The sum of the input values.
    """
    return _call_cpp_func(_cpp.sum, x, dim, out=out)


def min(x, dim=None, out=None):
    """Element-wise min over the specified dimension or all dimensions if not
    provided.

    :param x: Input data.
    :param dim: Optional dimension along which to calculate the min. If not
                given, the min over all dimensions is calculated.
    :param out: Optional output buffer.
    :raises: If the dimension does not exist, or the dtype cannot be summed,
             e.g., if it is a string.
    :return: The min of the input values.
    """
    if dim is None:
        return _call_cpp_func(_cpp.min, x, out=out)
    else:
        return _call_cpp_func(_cpp.min, x, dim=dim, out=out)


def max(x, dim=None, out=None):
    """Element-wise max over the specified dimension or all dimensions if not
    provided.

    :param x: Input data.
    :param dim: Optional dimension along which to calculate the max. If not
                given, the max over all dimensions is calculated.
    :param out: Optional output buffer.
    :raises: If the dimension does not exist, or the dtype cannot be summed,
             e.g., if it is a string.
    :return: The max of the input values.
    """
    if dim is None:
        return _call_cpp_func(_cpp.max, x, out=out)
    else:
        return _call_cpp_func(_cpp.max, x, dim=dim, out=out)


def nanmin(x, dim=None, out=None):
    """Element-wise min ignoring not at number values over the specified
    dimension or all dimensions if not provided.

    :param x: Input data.
    :param dim: Optional dimension along which to calculate the min. If not
                given, the min over all dimensions is calculated.
    :param out: Optional output buffer.
    :raises: If the dimension does not exist, or the dtype cannot be summed,
             e.g., if it is a string.
    :return: The min of the input values.
    """
    if dim is None:
        return _call_cpp_func(_cpp.nanmin, x, out=out)
    else:
        return _call_cpp_func(_cpp.nanmin, x, dim=dim, out=out)


def nanmax(x, dim=None, out=None):
    """Element-wise max ignoring not a number values over the specified
    dimension or all dimensions if not provided.

    :param x: Input data.
    :param dim: Optional dimension along which to calculate the max. If not
                given, the max over all dimensions is calculated.
    :param out: Optional output buffer.
    :raises: If the dimension does not exist, or the dtype cannot be summed,
             e.g., if it is a string.
    :return: The max of the input values.
    """
    if dim is None:
        return _call_cpp_func(_cpp.nanmax, x, out=out)
    else:
        return _call_cpp_func(_cpp.nanmax, x, dim=dim, out=out)


def all(x, dim=None, out=None):
    """Element-wise AND over the specified dimension or all dimensions if not
    provided.

    :param x: Input data.
    :param dim: Optional dimension along which to calculate the AND. If not
                given, the AND over all dimensions is calculated.
    :param out: Optional output buffer.
    :raises: If the dimension does not exist, or the dtype cannot be summed,
             e.g., if it is a string.
    :return: The AND of the input values.
    """
    if dim is None:
        return _call_cpp_func(_cpp.all, x, out=out)
    else:
        return _call_cpp_func(_cpp.all, x, dim=dim, out=out)


def any(x, dim=None, out=None):
    """Element-wise OR over the specified dimension or all dimensions if not
    provided.

    :param x: Input data.
    :param dim: Optional dimension along which to calculate the OR. If not
                given, the OR over all dimensions is calculated.
    :param out: Optional output buffer.
    :raises: If the dimension does not exist, or the dtype cannot be summed,
             e.g., if it is a string.
    :return: The OR of the input values.
    """
    if dim is None:
        return _call_cpp_func(_cpp.any, x, out=out)
    else:
        return _call_cpp_func(_cpp.any, x, dim=dim, out=out)
