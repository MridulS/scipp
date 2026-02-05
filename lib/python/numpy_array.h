// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
/// NumPy C API array creation utilities.
#pragma once

#include <cstdint>
#include <type_traits>

#include "dtype_util.h"

// Forward declare types to avoid including Python/numpy headers here
// (which would cause issues with PY_ARRAY_UNIQUE_SYMBOL)
struct _object;
using npy_intp = std::intptr_t;

namespace scipp::python {

/// Create a numpy array view of existing data.
/// This uses the numpy C API directly for maximum efficiency.
/// @param data Pointer to the data buffer
/// @param ndim Number of dimensions
/// @param shape Array of dimension sizes
/// @param strides Array of strides in bytes
/// @param typenum NumPy type number (use get_numpy_typenum_* functions)
/// @param readonly Whether the array should be read-only
/// @param base Base object to keep alive (will be INCREF'd)
/// @return New reference to numpy array, or nullptr on error
_object *create_numpy_array_view(void *data, int ndim, npy_intp *shape,
                                 npy_intp *strides, int typenum, bool readonly,
                                 _object *base);

// Type number getters (to avoid exposing numpy headers)
int get_numpy_typenum_float64();
int get_numpy_typenum_float32();
int get_numpy_typenum_int64();
int get_numpy_typenum_int32();
int get_numpy_typenum_bool();

/// Get numpy type number for a C++ type
template <class T> inline int get_numpy_typenum() {
  if constexpr (std::is_same_v<T, double>)
    return get_numpy_typenum_float64();
  else if constexpr (std::is_same_v<T, float>)
    return get_numpy_typenum_float32();
  else if constexpr (std::is_same_v<T, int64_t>)
    return get_numpy_typenum_int64();
  else if constexpr (std::is_same_v<T, int32_t>)
    return get_numpy_typenum_int32();
  else if constexpr (std::is_same_v<T, bool>)
    return get_numpy_typenum_bool();
  else
    return get_numpy_typenum_float64(); // fallback
}

} // namespace scipp::python
