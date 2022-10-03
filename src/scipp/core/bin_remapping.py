# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Dict
from .._scipp import core as _cpp
from .cpp_classes import DataArray, Variable, Dataset
from .like import empty_like
from .cumulative import cumsum
from .dataset import irreducible_mask
from ..typing import VariableLikeType
from .variable import arange, full
from .operations import sort


def _reduced(obj: Dict[str, Variable], dim: str) -> Dict[str, Variable]:
    return {name: var for name, var in obj.items() if dim not in var.dims}


def reduced_coords(da: DataArray, dim: str) -> Dict[str, Variable]:
    return _reduced(da.coords, dim)


def reduced_attrs(da: DataArray, dim: str) -> Dict[str, Variable]:
    return _reduced(da.attrs, dim)


def reduced_masks(da: DataArray, dim: str) -> Dict[str, Variable]:
    return {name: mask.copy() for name, mask in _reduced(da.masks, dim).items()}


def _copy_dict_for_overwrite(mapping: Dict[str, Variable]) -> Dict[str, Variable]:
    return {name: copy_for_overwrite(var) for name, var in mapping.items()}


def copy_for_overwrite(obj: VariableLikeType) -> VariableLikeType:
    """
    Copy a Scipp object for overwriting.

    Unlike :py:func:`scipp.empty_like` this does not preserve (and share) coord,
    mask, and attr values. Instead, those values are not initialized, just like the
    data values.
    """
    if isinstance(obj, Variable):
        return empty_like(obj)
    if isinstance(obj, DataArray):
        return DataArray(copy_for_overwrite(obj.data),
                         coords=_copy_dict_for_overwrite(obj.coords),
                         masks=_copy_dict_for_overwrite(obj.masks),
                         attrs=_copy_dict_for_overwrite(obj.attrs))
    ds = Dataset(coords=_copy_dict_for_overwrite(obj.coords))
    for name, da in obj.items():
        ds[name] = DataArray(copy_for_overwrite(da.data),
                             masks=_copy_dict_for_overwrite(da.masks),
                             attrs=_copy_dict_for_overwrite(da.attrs))
    return ds


def hide_masked_and_reduce_meta(da: DataArray, dim: str) -> DataArray:
    if (mask := irreducible_mask(da.masks, dim)) is not None:
        # Avoid using boolean indexing since it would result in (partial) content
        # buffer copy. Instead index just begin/end and reuse content buffer.
        comps = da.bins.constituents
        select = ~mask
        data = _cpp._bins_no_validate(
            data=comps['data'],
            dim=comps['dim'],
            begin=comps['begin'][select],
            end=comps['end'][select],
        )
    else:
        data = da.data
    return DataArray(data,
                     coords=reduced_coords(da, dim),
                     masks=reduced_masks(da, dim),
                     attrs=reduced_attrs(da, dim))


def _replace_bin_sizes(var: Variable, sizes: Variable) -> Variable:
    out_end = cumsum(sizes)
    out_begin = out_end - sizes
    return _cpp._bins_no_validate(
        data=var.bins.constituents['data'],
        dim=var.bins.constituents['dim'],
        begin=out_begin,
        end=out_end,
    )


def _remap_bins(var: Variable, begin, end, sizes) -> Variable:
    out = _cpp._bins_no_validate(
        data=copy_for_overwrite(var.bins.constituents['data']),
        dim=var.bins.constituents['dim'],
        begin=begin,
        end=end,
    )

    # Copy all bin contents, performing the actual reordering with the content buffer.
    out[...] = var

    # Setup output indices. This will have the "merged" bins, referencing the new
    # contiguous layout in the content buffer.
    return _replace_bin_sizes(out, sizes)


def _concat_bins_variable(var: Variable, dim: str) -> Variable:
    # We want to write data from all bins along dim to a contiguous chunk in the
    # content buffer. This will then allow us to create new, larger bins covering the
    # respective input bins. We use `cumsum` after moving `dim` to the innermost dim.
    # This will allow us to setup offsets for the new contiguous layout.
    sizes = var.bins.size()
    out_dims = [d for d in var.dims if d != dim]
    out_end = cumsum(sizes.transpose(out_dims + [dim])).transpose(var.dims)
    out_begin = out_end - sizes
    out_sizes = sizes.sum(dim)
    return _remap_bins(var, out_begin, out_end, out_sizes)


def _combine_bins_by_binning_variable(var: Variable, param: Variable,
                                      edges: Variable) -> Variable:
    sizes = DataArray(var.bins.size())
    index_range = arange(param.dim, len(param), unit=None)
    input_bin = DataArray(index_range, coords={edges.dim: param})
    sizes.coords['input_bin'] = index_range
    sizes.coords[edges.dim] = param

    dim = param.dim
    unchanged_dims = [d for d in var.dims if d != dim]
    sizes2 = sizes.transpose(unchanged_dims + [dim])
    meta_sizes = full(dims=unchanged_dims,
                      shape=[var.sizes[dim] for dim in unchanged_dims],
                      value=var.sizes[dim])
    print(f'{meta_sizes=}')
    meta_end = cumsum(meta_sizes)
    meta_begin = meta_end - meta_sizes
    tmp = _cpp._bins_no_validate(data=sizes2.flatten(to='dummy'),
                                 dim='dummy',
                                 begin=meta_begin,
                                 end=meta_end)
    print(tmp)
    out_dims = [d if d != dim else edges.dim for d in var.dims]
    # TODO This is some duplicate work, as all target indices are the same
    tmp = DataArray(tmp).bin({edges.dim: edges}).transpose(out_dims)
    print(tmp.bins.sum())

    # Setup shuffle indices for reordering input sizes such that we can use cumsum for
    # computing output bin sizes and offsets.
    # We bin only the indices and not the sizes since the latter might not be 1-D
    grouped_input_bin = input_bin.bin({edges.dim: edges})
    shuffle = grouped_input_bin.bins.concat(edges.dim).value.values
    print(f'{shuffle=}')
    sizes_sort_out = sizes2[param.dim, shuffle]

    # Indices for reverting shuffle
    reverse_shuffle = DataArray(index_range,
                                coords={'order': sizes_sort_out.coords['input_bin']})
    reverse_shuffle = sort(reverse_shuffle, 'order').values

    out_end = cumsum(sizes_sort_out.data)[param.dim, reverse_shuffle]
    out_begin = out_end - sizes2.data
    out_sizes = tmp.data.bins.sum()
    print(f'{param=}')
    print(f'{var.bins.size().values=}')
    print(f'{out_begin.values=}')
    print(f'{out_end.values=}')
    print(f'{(out_end-out_begin).values=}')
    print(f'{out_sizes=}')
    return _remap_bins(var, out_begin, out_end, out_sizes)


def concat_bins(obj: VariableLikeType, dim: str) -> VariableLikeType:
    if isinstance(obj, Variable):
        return _concat_bins_variable(obj, dim)
    else:
        da = hide_masked_and_reduce_meta(obj, dim)
        data = _concat_bins_variable(da.data, dim)
        return DataArray(data, coords=da.coords, masks=da.masks, attrs=da.attrs)
