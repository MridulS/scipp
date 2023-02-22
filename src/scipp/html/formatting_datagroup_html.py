# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import uuid
from string import Template
from typing import Union

import numpy as np

from ..core.cpp_classes import DataArray, Dataset, Variable
from ..core.data_group import DataGroup
from ..units import dimensionless
from .formatting_html import escape, inline_variable_repr
from .resources import load_atomic_row_tpl, load_collapsible_row_tpl, \
    load_dg_detail_list_tpl, load_dg_repr_tpl, load_dg_style


def _format_shape(var: Union[Variable, DataArray, Dataset, DataGroup], br_at=30) -> str:
    """Return HTML Component that represents the shape of ``var``"""
    shape_list = [f"{escape(str(dim))}: {size}" for dim, size in var.sizes.items()]
    if sum([len(line) - line.count('\\') for line in shape_list]) < br_at:
        return f"({', '.join(shape_list)})"
    else:
        return f"({', <br>&nbsp'.join(shape_list)})"


def _format_atomic_value(value, maxidx: int = 5) -> str:
    """Inline preview of single value"""
    value_repr = str(value)[:maxidx]
    if len(value_repr) < len(str(value)):
        value_repr += "..."
    return value_repr


def _format_dictionary_item(name_item: tuple, maxidx: int = 10) -> str:
    """Inline preview of a dictionary"""
    name, item = name_item
    name = _format_atomic_value(name, maxidx=maxidx)
    type_repr = _format_atomic_value(type(item).__name__, maxidx=maxidx)
    return "(" + ": ".join((name, type_repr)) + ")"


def _format_multi_dim_data(var: Union[Variable, DataArray, Dataset, np.ndarray]) -> str:
    """Inline preview of single or multi-dimensional data"""
    if isinstance(var, (Variable, DataArray)):
        return inline_variable_repr(var)[5:-6]
    elif isinstance(var, Dataset):
        view_iterable = list(var.items())
        var_len = len(var)
        first_idx, last_idx = 0, -1
        format_item = _format_dictionary_item
    elif isinstance(var, np.ndarray):
        view_iterable = var
        var_len = var.size
        first_idx = tuple(np.zeros(var.ndim, dtype=int))
        last_idx = tuple(np.array(var.shape, dtype=int) - np.ones(var.ndim, dtype=int))
        format_item = _format_atomic_value

    view_items = []
    if var_len > 0:
        view_items.append(format_item(view_iterable[first_idx]))
    if var_len > 2:
        view_items.append('... ')
    if var_len > 1:
        view_items.append(format_item(view_iterable[last_idx]))

    return ', '.join(view_items)


def _summarize_atomic_variable(var, name: str, depth: int = 0) -> str:
    """Return HTML Component that contains summary of ``var``"""
    shape_repr = escape("()")
    unit = ''
    dtype_str = ''
    preview = ''
    parent_obj_str = ''
    objtype_str = type(var).__name__
    if isinstance(var, (Dataset, DataArray, Variable)):
        parent_obj_str = "scipp"
        shape_repr = _format_shape(var)
        preview = _format_multi_dim_data(var)
        if not isinstance(var, Dataset):
            dtype_str = str(var.dtype)
            if var.unit is not None:
                unit = '𝟙' if var.unit == dimensionless else str(var.unit)
    elif isinstance(var, np.ndarray):
        parent_obj_str = "numpy"
        preview = f"shape={var.shape}, dtype={var.dtype}, values="
        preview += _format_multi_dim_data(var)

    elif preview == '' and hasattr(var, "__str__"):
        preview = _format_atomic_value(var, maxidx=30)

    html_tpl = load_atomic_row_tpl()
    return Template(html_tpl).substitute(depth=depth,
                                         name=escape(name),
                                         parent=escape(parent_obj_str),
                                         objtype=escape(objtype_str),
                                         shape_repr=shape_repr,
                                         dtype=escape(dtype_str),
                                         unit=escape(unit),
                                         preview=escape(preview))


def _collapsible_summary(var: DataGroup, name: str, name_spaces: list) -> str:
    parent_type = "scipp"
    objtype = type(var).__name__
    shape_repr = _format_shape(var)
    checkbox_id = escape("summary-" + str(uuid.uuid4()))
    depth = len(name_spaces)
    subsection = _datagroup_detail(var, name_spaces + [name])
    html_tpl = load_collapsible_row_tpl()

    return Template(html_tpl).substitute(name=escape(str(name)),
                                         parent=escape(parent_type),
                                         objtype=escape(objtype),
                                         shape_repr=shape_repr,
                                         summary_section_id=checkbox_id,
                                         depth=depth,
                                         checkbox_status='',
                                         subsection=subsection)


def _datagroup_detail(dg: DataGroup, name_spaces: list = None) -> str:
    if name_spaces is None:
        name_spaces = []
    summary_rows = []
    for name, item in dg.items():
        if isinstance(item, DataGroup):
            collapsible_row = _collapsible_summary(item, name, name_spaces)
            summary_rows.append(collapsible_row)
        else:
            summary_rows.append(
                _summarize_atomic_variable(item, name, depth=len(name_spaces)))

    dg_detail_tpl = Template(load_dg_detail_list_tpl())
    return dg_detail_tpl.substitute(summary_rows=''.join(summary_rows))


def datagroup_repr(dg: DataGroup) -> str:
    """Return HTML Component containing details of ``dg``"""
    obj_type = "scipp.{} ".format(type(dg).__name__)
    checkbox_status = "checked" if len(dg) < 15 else ''
    header_id = "datagroup-view-" + str(uuid.uuid4())
    details = _datagroup_detail(dg)
    html = Template(load_dg_repr_tpl())
    return html.substitute(style_sheet=load_dg_style(),
                           header_id=header_id,
                           checkbox_status=checkbox_status,
                           obj_type=obj_type,
                           shape_repr=_format_shape(dg, br_at=200),
                           details=details)
