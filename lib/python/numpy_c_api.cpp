// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
/// @file

#include "numpy_c_api.h"

namespace nb = nanobind;

namespace scipp::python {
namespace {

/// Pointers into numpy's C API jump table. The slot indices are part of
/// numpy's stable, append-only C ABI (verified against numpy/__multiarray_api.h
/// for numpy >= 2) and apply across the entire numpy >= 2 range that scipp
/// supports.
struct NumpyCApi {
  // PyArray_Type, slot 2 (used as the `subtype` argument).
  PyObject *array_type;
  // PyArray_DescrFromType, slot 45.
  PyObject *(*descr_from_type)(int typenum);
  // PyArray_NewFromDescr, slot 94. `descr` and the returned base reference are
  // stolen by numpy. Strides are in bytes; npy_intp is a pointer-sized signed
  // integer.
  PyObject *(*new_from_descr)(PyObject *subtype, PyObject *descr, int nd,
                              const std::intptr_t *dims,
                              const std::intptr_t *strides, void *data,
                              int flags, PyObject *obj);
  // PyArray_SetBaseObject, slot 282. Steals a reference to `base`.
  int (*set_base_object)(PyObject *arr, PyObject *base);
};

/// Resolve the numpy C API once. The table is intentionally leaked so that no
/// destructor runs after interpreter finalization. Callers hold the GIL.
const NumpyCApi &numpy_c_api() {
  static_assert(sizeof(std::int64_t) == sizeof(std::intptr_t),
                "numpy npy_intp is expected to be 64-bit");
  static const NumpyCApi &api = *[] {
    const nb::object capsule =
        nb::module_::import_("numpy._core._multiarray_umath").attr("_ARRAY_API");
    auto **table =
        static_cast<void **>(PyCapsule_GetPointer(capsule.ptr(), nullptr));
    if (table == nullptr)
      throw nb::python_error();
    auto *resolved = new NumpyCApi{};
    resolved->array_type = static_cast<PyObject *>(table[2]);
    resolved->descr_from_type =
        reinterpret_cast<decltype(resolved->descr_from_type)>(table[45]);
    resolved->new_from_descr =
        reinterpret_cast<decltype(resolved->new_from_descr)>(table[94]);
    resolved->set_base_object =
        reinterpret_cast<decltype(resolved->set_base_object)>(table[282]);
    return resolved;
  }();
  return api;
}

} // namespace

nanobind::object make_numpy_array(int typenum, void *data, std::size_t ndim,
                                  const std::int64_t *shape,
                                  const std::int64_t *byte_strides,
                                  bool writeable, nanobind::object owner) {
  const NumpyCApi &api = numpy_c_api();
  // PyArray_NewFromDescr steals the descriptor reference.
  PyObject *descr = api.descr_from_type(typenum);
  if (descr == nullptr)
    throw nb::python_error();
  constexpr int npy_array_writeable = 0x0400; // NPY_ARRAY_WRITEABLE
  PyObject *array = api.new_from_descr(
      api.array_type, descr, static_cast<int>(ndim),
      reinterpret_cast<const std::intptr_t *>(shape),
      reinterpret_cast<const std::intptr_t *>(byte_strides), data,
      writeable ? npy_array_writeable : 0, nullptr);
  if (array == nullptr)
    throw nb::python_error();
  // PyArray_SetBaseObject steals a reference to the owner.
  if (api.set_base_object(array, owner.release().ptr()) < 0) {
    Py_DECREF(array);
    throw nb::python_error();
  }
  return nb::steal(array);
}

} // namespace scipp::python
