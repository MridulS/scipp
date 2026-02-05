// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
/// NumPy C API array creation utilities.

#include "numpy_array.h"

// Must define PY_ARRAY_UNIQUE_SYMBOL before including numpy headers
// to avoid multiple definition errors when used across compilation units
#define PY_ARRAY_UNIQUE_SYMBOL scipp_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace scipp::python {

namespace {
// Initialize numpy on first use
bool ensure_numpy_initialized() {
  static bool initialized = false;
  if (!initialized) {
    if (_import_array() < 0) {
      PyErr_Print();
      return false;
    }
    initialized = true;
  }
  return true;
}
} // namespace

_object *create_numpy_array_view(void *data, int ndim, npy_intp *shape,
                                 npy_intp *strides, int typenum, bool readonly,
                                 _object *base) {
  if (!ensure_numpy_initialized()) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to initialize NumPy");
    return nullptr;
  }

  int flags = readonly ? 0 : NPY_ARRAY_WRITEABLE;

  // Cast _object* to PyObject* for Python C API
  auto *py_base = reinterpret_cast<PyObject *>(base);

  PyObject *arr =
      PyArray_New(&PyArray_Type, ndim, shape, typenum, strides, data,
                  0, // itemsize (0 = use default)
                  flags, nullptr);

  if (!arr) {
    return nullptr;
  }

  // Set base object to keep data alive
  if (py_base) {
    Py_INCREF(py_base);
    if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(arr), py_base) <
        0) {
      Py_DECREF(arr);
      return nullptr;
    }
  }

  return reinterpret_cast<_object *>(arr);
}

int get_numpy_typenum_float64() { return NPY_FLOAT64; }
int get_numpy_typenum_float32() { return NPY_FLOAT32; }
int get_numpy_typenum_int64() { return NPY_INT64; }
int get_numpy_typenum_int32() { return NPY_INT32; }
int get_numpy_typenum_bool() { return NPY_BOOL; }

} // namespace scipp::python
