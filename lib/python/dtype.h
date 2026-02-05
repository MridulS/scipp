// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
/// @file
#pragma once

#include <string_view>

#include <scipp/core/dtype.h>
#include <scipp/units/unit.h>

#include "nanobind.h"
#include "unit.h"

namespace nanobind {
class dtype;
}

scipp::core::DType dtype_of(const nanobind::object &x);

scipp::core::DType scipp_dtype(const nanobind::object &type);

std::tuple<scipp::core::DType, std::optional<scipp::sc_units::Unit>>
cast_dtype_and_unit(const nanobind::object &dtype, const ProtoUnit &unit);

void ensure_conversion_possible(scipp::core::DType from, scipp::core::DType to,
                                const std::string &data_name);

template <class T, class = void> struct converting_cast {
  static decltype(auto) cast(const nanobind::object &obj) {
    return nanobind::cast<T>(obj);
  }
};

template <class T>
struct converting_cast<T, std::enable_if_t<std::is_integral_v<T>>> {
  static decltype(auto) cast(const nanobind::object &obj) {
    if (dtype_of(obj) == scipp::dtype<double>) {
      // This conversion is not implemented in nanobind directly
      return nanobind::cast<T>(nanobind::int_(obj));
    } else {
      // All other conversions are either supported by nanobind or not
      // desired anyway.
      return nanobind::cast<T>(obj);
    }
  }
};

scipp::core::DType
common_dtype(const nanobind::object &values, const nanobind::object &variances,
             scipp::core::DType dtype,
             scipp::core::DType default_dtype = scipp::core::dtype<double>);

bool has_datetime_dtype(const nanobind::object &obj);

[[nodiscard]] scipp::sc_units::Unit
parse_datetime_dtype(const std::string &dtype_name);
[[nodiscard]] scipp::sc_units::Unit
parse_datetime_dtype(const nanobind::object &dtype);
