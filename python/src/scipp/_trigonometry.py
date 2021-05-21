# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from ._scipp import core as _cpp
from ._cpp_wrapper_util import call_func as _call_cpp_func


def sin(x, out=None):
    """Element-wise sin.

    :param x: Input data.
    :param out: Optional output buffer.
    :raises: If the unit is not a plane-angle unit, or if the sin cannot be
             computed on the dtype, e.g., if it is an integer.
    :return: The sin values of the input.
    """
    return _call_cpp_func(_cpp.sin, x, out=out)


def cos(x, out=None):
    """Element-wise cos.

    :param x: Input data.
    :param out: Optional output buffer.
    :raises: If the unit is not a plane-angle unit, or if the cos cannot be
             computed on the dtype, e.g., if it is an integer.
    :return: The cos values of the input.
    """
    return _call_cpp_func(_cpp.cos, x, out=out)


def tan(x, out=None):
    """Element-wise tan.

    :param x: Input data.
    :param out: Optional output buffer.
    :raises: If the unit is not a plane-angle unit, or if the tan cannot be
             computed on the dtype, e.g., if it is an integer.
    :return: The tan values of the input.
    """
    return _call_cpp_func(_cpp.tan, x, out=out)


def asin(x, out=None):
    """Element-wise inverse sin.

    :param x: Input data.
    :param out: Optional output buffer.
    :raises: If the unit is not dimensionless.
    :return: The inverse sin values of the input.
    """
    return _call_cpp_func(_cpp.asin, x, out=out)


def acos(x, out=None):
    """Element-wise inverse cos.

    :param x: Input data.
    :param out: Optional output buffer.
    :raises: If the unit is not dimensionless.
    :return: The inverse cos values of the input.
    """
    return _call_cpp_func(_cpp.acos, x, out=out)


def atan(x, out=None):
    """Element-wise inverse tan.

    :param x: Input data.
    :param out: Optional output buffer.
    :raises: If the unit is not dimensionless.
    :return: The inverse tan values of the input.
    """
    return _call_cpp_func(_cpp.atan, x, out=out)


def atan2(y, x, out=None):
    """Element-wise inverse tan providing signed angle.

    :param y: Input y values.
    :param x: Input x values.
    :param out: Optional output buffer.
    :return: The signed inverse tan values of the inputs.
    """
    return _call_cpp_func(_cpp.atan2, y, x, out=out)
