# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
import scipp as sc
import numpy as np


def make_var() -> sc.Variable:
    return sc.arange('dummy', 12, dtype='int64').fold(dim='dummy',
                                                      sizes={
                                                          'xx': 4,
                                                          'yy': 3
                                                      })


def make_array() -> sc.DataArray:
    da = sc.DataArray(make_var())
    da.coords['xx'] = sc.arange('xx', 4, dtype='int64')
    da.coords['yy'] = sc.arange('yy', 3, dtype='int64')
    return da


def make_dataset() -> sc.Dataset:
    ds = sc.Dataset()
    ds['xy'] = make_array()
    ds['x'] = ds.coords['xx']
    return ds


@pytest.mark.parametrize("obj", [make_var(), make_array(), make_dataset()])
@pytest.mark.parametrize("pos", [0, 1, 2, 3, -2, -3, -4])
def test_length_1_list_gives_corresponding_length_1_slice(obj, pos):
    assert sc.identical(obj['xx', [pos]], obj['xx', pos:pos + 1])


def test_slicing_with_numpy_array_works_and_gives_equivalent_result():
    var = make_var()
    assert sc.identical(var['xx', np.array([2, 3, 0])], var['xx', [2, 3, 0]])


@pytest.mark.parametrize("obj", [make_var(), make_array(), make_dataset()])
def test_omitting_dim_when_slicing_2d_object_raises_DimensionError(obj):
    with pytest.raises(sc.DimensionError):
        obj[[3, 1]]


@pytest.mark.parametrize("obj", [make_var(), make_array(), make_dataset()])
def test_omitted_dim_is_equivalent_to_unique_dim(obj):
    obj = obj['yy', 0]
    assert sc.identical(obj[[3, 1]], obj[obj.dim, [3, 1]])


@pytest.mark.parametrize("obj", [make_var(), make_array(), make_dataset()])
def test_every_other_index_gives_stride_2_slice(obj):
    assert sc.identical(obj['xx', [0, 2]], obj['xx', 0::2])
    assert sc.identical(obj['xx', [1, 3]], obj['xx', 1::2])


@pytest.mark.parametrize("obj", [make_var(), make_array(), make_dataset()])
def test_unordered_outer_indices_yields_result_with_reordered_slices(obj):
    assert sc.identical(obj['xx', [2, 3, 0]],
                        sc.concat([obj['xx', 2:4], obj['xx', 0:1]], 'xx'))


@pytest.mark.parametrize("obj", [make_var(), make_array(), make_dataset()])
def test_unordered_inner_indices_yields_result_with_reordered_slices(obj):
    s0 = obj['yy', 0:1]
    s1 = obj['yy', 1:2]
    assert sc.identical(obj['yy', [1, 1, 0, 1]], sc.concat([s1, s1, s0, s1], 'yy'))


@pytest.mark.parametrize("obj", [make_var(), make_array(), make_dataset()])
def test_duplicate_indices_duplicate_slices_in_output(obj):
    s1 = obj['xx', 1:2]
    s2 = obj['xx', 2:3]
    assert sc.identical(obj['xx', [2, 1, 1, 2]], sc.concat([s2, s1, s1, s2], 'xx'))


@pytest.mark.parametrize("obj", [make_var(), make_array(), make_dataset()])
def test_reversing_twice_gives_original(obj):
    assert sc.identical(obj['xx', [3, 2, 1, 0]]['xx', [3, 2, 1, 0]], obj)


@pytest.mark.parametrize("obj", [make_array(), make_dataset()])
@pytest.mark.parametrize("what", ["coords", "masks", "attrs"])
def test_bin_edges_are_dropped(obj, what):
    obj = obj.copy()
    base = obj.copy()
    edges = sc.concat([obj.coords['xx'], obj.coords['xx'][-1] + 1], 'xx')
    da = obj if isinstance(obj, sc.DataArray) or what == 'coords' else obj['xy']
    getattr(da, what)['edges'] = edges
    assert sc.identical(obj['xx', [0, 2, 3]],
                        sc.concat([base['xx', 0], base['xx', 2:]], 'xx'))


def test_dataset_item_independent_of_slice_dim_preserved_unchanged():
    ds = make_dataset()
    assert sc.identical(ds['yy', [0, 2]]['x'], ds['x'])


def test_2d_list_raises_TypeError():
    var = make_var()
    with pytest.raises(TypeError):
        var['xx', [[0], [2]]]


@pytest.mark.parametrize("pos", [-6, -5, 4, 5])
def test_out_of_range_index_raises_IndexError(pos):
    var = sc.arange('xx', 4)
    with pytest.raises(IndexError):
        var['xx', pos]
