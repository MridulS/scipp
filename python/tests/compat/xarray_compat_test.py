import numpy
import xarray
import scipp as sc
from scipp.compat.xarray_compat import from_xarray


def test_empty_attrs_dataarray():
    xr_da = xarray.DataArray(data=numpy.zeros((1, )), dims={"x"}, attrs={})

    sc_da = from_xarray(xr_da)

    assert len(sc_da.attrs) == 0

    assert len(sc_da.dims) == 1
    assert "x" in sc_da.dims

    assert len(sc_da.masks) == 0


def test_attrs_dataarray():
    xr_da = xarray.DataArray(data=numpy.zeros((1, )),
                             dims={"x"},
                             attrs={
                                 "attrib_int": 5,
                                 "attrib_float": 6.54321,
                                 "attrib_str": "test-string",
                             })

    sc_da = from_xarray(xr_da)

    assert sc_da.attrs["attrib_int"].values == 5
    assert sc_da.attrs["attrib_float"].values == 6.54321
    assert sc_da.attrs["attrib_str"].values == "test-string"


def test_named_dataarray():
    xr_da = xarray.DataArray(data=numpy.zeros((1, )),
                             dims={"x"},
                             name="my-test-dataarray")

    sc_da = from_xarray(xr_da)

    assert sc_da.name == "my-test-dataarray"


def test_1d_1_element_dataarray():
    xr_da = xarray.DataArray(data=numpy.zeros((1, )), dims={"x"}, attrs={})

    sc_da = from_xarray(xr_da)

    assert sc.identical(sc_da,
                        sc.DataArray(data=sc.zeros(dims=["x"], shape=(1, ))))


def test_1d_100_element_dataarray():
    xr_da = xarray.DataArray(data=numpy.zeros((100, )), dims={"x"}, attrs={})

    sc_da = from_xarray(xr_da)

    assert sc.identical(sc_da,
                        sc.DataArray(data=sc.zeros(dims=["x"], shape=(100, ))))


def test_2d_100x100_element_dataarray():
    xr_da = xarray.DataArray(data=numpy.zeros((100, 100)),
                             dims={"x", "y"},
                             attrs={})

    sc_da = from_xarray(xr_da)

    assert sc.identical(
        sc_da, sc.DataArray(data=sc.zeros(dims=["x", "y"], shape=(100, 100))))


def test_empty_dataset():
    xr_ds = xarray.Dataset(data_vars={})

    sc_ds = from_xarray(xr_ds)

    assert sc.identical(sc_ds, sc.Dataset(data={}))


def test_dataset_with_data():
    xr_ds = xarray.Dataset(
        data_vars={
            "array1":
            xarray.DataArray(data=numpy.zeros((100, )), dims={"x"}, attrs={}),
            "array2":
            xarray.DataArray(data=numpy.zeros((50, )), dims={"y"}, attrs={}),
        })

    sc_ds = from_xarray(xr_ds)

    reference_ds = sc.Dataset(
        data={
            "array1": sc.DataArray(data=sc.zeros(dims=["x"], shape=(100, ))),
            "array2": sc.DataArray(data=sc.zeros(dims=["y"], shape=(50, ))),
        })

    assert sc.identical(sc_ds, reference_ds)
