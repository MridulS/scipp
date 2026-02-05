// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
/// Dtype string utilities for numpy interop.
#pragma once

#include <cstdint>

namespace scipp::python {

/// Get numpy dtype string for a C++ type.
/// Returns nullptr for unsupported types.
template <class T> constexpr const char *numpy_dtype_str() {
  if constexpr (std::is_same_v<T, double>)
    return "float64";
  else if constexpr (std::is_same_v<T, float>)
    return "float32";
  else if constexpr (std::is_same_v<T, int64_t>)
    return "int64";
  else if constexpr (std::is_same_v<T, int32_t>)
    return "int32";
  else if constexpr (std::is_same_v<T, bool>)
    return "bool";
  else
    return nullptr;
}

} // namespace scipp::python
