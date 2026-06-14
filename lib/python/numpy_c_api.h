// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
/// @file
/// Direct construction of numpy arrays via the numpy C API.
///
/// nanobind's nb::ndarray numpy export allocates a wrapper object and calls
/// numpy.array(wrapper, copy=False) at the Python level, which is ~1.5x slower
/// than the array construction pybind11 performed. This helper instead calls
/// the numpy C API (PyArray_NewFromDescr + PyArray_SetBaseObject) directly, as
/// pybind11 did, resolving the API at runtime from numpy's `_ARRAY_API` capsule
/// so that no numpy headers are needed at build time.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <nanobind/nanobind.h>

#include "scipp/core/time_point.h"

namespace scipp::python {

/// numpy type numbers (a subset of numpy's NPY_TYPES enum), defined here to
/// avoid a build-time dependency on numpy's C headers.
template <class T> constexpr int numpy_typenum() {
  if constexpr (std::is_same_v<T, double>)
    return 12; // NPY_DOUBLE
  else if constexpr (std::is_same_v<T, float>)
    return 11; // NPY_FLOAT
  else if constexpr (std::is_same_v<T, std::int64_t> ||
                     std::is_same_v<T, scipp::core::time_point>)
    // numpy's canonical int64 is NPY_LONG on LP64 (Linux, macOS) and
    // NPY_LONGLONG on LLP64 (Windows).
    return sizeof(long) == 8 ? 7 : 9;
  else if constexpr (std::is_same_v<T, std::int32_t>)
    return 5; // NPY_INT
  else if constexpr (std::is_same_v<T, bool>)
    return 0; // NPY_BOOL
  else {
    static_assert(sizeof(T) == 0, "type has no numpy type number");
    return -1;
  }
}

/// Build a numpy array that views existing memory without copying.
///
/// @param typenum  numpy type number of the elements (see numpy_typenum).
/// @param data     pointer to the first element.
/// @param ndim     number of dimensions.
/// @param shape    extent of each dimension.
/// @param byte_strides  stride of each dimension, in bytes.
/// @param writeable  whether the resulting array may be modified.
/// @param owner    object set as the array's base to keep `data` alive; its
///                 reference is consumed.
nanobind::object make_numpy_array(int typenum, void *data, std::size_t ndim,
                                  const std::int64_t *shape,
                                  const std::int64_t *byte_strides,
                                  bool writeable, nanobind::object owner);

} // namespace scipp::python
