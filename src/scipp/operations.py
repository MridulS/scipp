# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Callable, Optional
from inspect import signature
from .core import Variable
from ._scipp.core import transform as cpp_transform


def _as_numba_cfunc(function, unit_func=None):
    import numba
    dtype = 'double'
    narg = len(signature(function).parameters)
    cfunc = numba.cfunc(dtype + '(' + ','.join([dtype] * narg) + ')')(function)
    cfunc.unit_func = function if unit_func is None else unit_func
    cfunc.name = function.__name__
    return cfunc


def elemwise_func(func: Callable,
                  unit_func: Optional[Callable] = None,
                  dtype: str = 'float64',
                  auto_convert_dtypes: bool = False) -> Callable:
    """
    Create a function for transforming input variables based on element-wise operation.

    This uses ``numba.cfunc`` to compile a kernel that Scipp can use for transforming
    the variable contents. Only variables with dtype=float64 are supported. Variances
    are not supported.

    Custom kernels can reduce intermediate memory consumption and improve performance
    in multi-step operations with large input variables.

    Parameters
    ----------
    func:
        Function to compute an output element from input element values.
    unit_func:
        Function to compute the output unit. If ``None``, ``func`` wil be used.
    auto_convert_dtypes:
        Set to ``True`` to automatically convert all inputs to float64.

    Returns
    -------
    :
        A callable that applies ``func`` to the elements of the variables passed to it.

    Examples
    --------

    We can define a fused multiply-add operation as follows:

      >>> def fmadd(a, b, c):
      ...     return a * b + c

      >>> func = sc.elemwise_func(fmadd)

      >>> x = sc.linspace('x', 0.0, 1.0, num=4, unit='m')
      >>> y = x - 0.2 * x
      >>> z = sc.scalar(1.2, unit='m**2')

      >>> func(x, y, z)
      <scipp.Variable> (x: 4)    float64            [m^2]  [1.2, 1.28889, 1.55556, 2]

    Note that ``fmadd(x, y, z)`` would have the same effect in this case, but requires
    a potentially large intermediate allocation for the result of "a * b".
    """
    func = _as_numba_cfunc(func, unit_func=unit_func)

    def transform_custom(*args: Variable) -> Variable:
        if auto_convert_dtypes:
            args = [arg.to(dtype='float64', copy=False) for arg in args]
        return cpp_transform(func, *args)

    return transform_custom
