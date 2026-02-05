# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

# ruff: noqa: F401

from .._scipp.core import (
    BinEdgeError,
    BinnedDataError,
    CoordError,
    Coords,
    DataArray,
    DataArrayError,
    Dataset,
    DatasetError,
    DefaultUnit,
    DimensionError,
    DType,
    DTypeError,
    GroupByDataArray,
    GroupByDataset,
    Masks,
    Unit,
    UnitError,
    Variable,
    VariableError,
    VariancesError,
)

# Save references to the original C++ read-only variances properties
_Variable_variances_getter = Variable.variances.fget
_DataArray_variances_getter = DataArray.variances.fget


def _variable_variances_getter(self):
    return _Variable_variances_getter(self)


def _variable_variances_setter(self, value):
    self._set_variances(value)


def _dataarray_variances_getter(self):
    return _DataArray_variances_getter(self)


def _dataarray_variances_setter(self, value):
    self._set_variances(value)


# Replace the read-only variances property with one that has a setter
Variable.variances = property(
    _variable_variances_getter,
    _variable_variances_setter,
    doc=Variable.variances.__doc__,
)

DataArray.variances = property(
    _dataarray_variances_getter,
    _dataarray_variances_setter,
    doc=DataArray.variances.__doc__,
)
