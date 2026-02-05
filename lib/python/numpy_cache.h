// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
/// Provides cached access to numpy module and common functions.
#pragma once

#include "nanobind.h"

namespace nb = nanobind;

namespace scipp::python {

/// Cached numpy module access.
/// Uses heap allocation to avoid GIL issues during shutdown.
/// The allocated objects are intentionally leaked.
inline nb::module_ &numpy_module() {
  static nb::module_ *numpy = new nb::module_(nb::module_::import_("numpy"));
  return *numpy;
}

/// Cached numpy.asarray function.
inline nb::object &numpy_asarray() {
  static nb::object *func = new nb::object(numpy_module().attr("asarray"));
  return *func;
}

/// Cached numpy.ascontiguousarray function.
inline nb::object &numpy_ascontiguousarray() {
  static nb::object *func =
      new nb::object(numpy_module().attr("ascontiguousarray"));
  return *func;
}

/// Cached numpy.dtype function.
inline nb::object &numpy_dtype() {
  static nb::object *func = new nb::object(numpy_module().attr("dtype"));
  return *func;
}

// Backward compatibility alias (deprecated, use numpy_dtype instead)
inline nb::object &numpy_dtype_func() { return numpy_dtype(); }

/// Cached numpy.array function.
inline nb::object &numpy_array() {
  static nb::object *func = new nb::object(numpy_module().attr("array"));
  return *func;
}

/// Cached numpy.frombuffer function.
inline nb::object &numpy_frombuffer() {
  static nb::object *func = new nb::object(numpy_module().attr("frombuffer"));
  return *func;
}

/// Cached numpy.lib.stride_tricks.as_strided function.
inline nb::object &numpy_as_strided() {
  static nb::object *func = new nb::object(
      numpy_module().attr("lib").attr("stride_tricks").attr("as_strided"));
  return *func;
}

} // namespace scipp::python
