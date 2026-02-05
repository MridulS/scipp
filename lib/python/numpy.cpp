// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Jan-Lukas Wynen

#include "numpy.h"

#include "dtype.h"

void ElementTypeMap<scipp::core::time_point>::check_assignable(
    const nb::object &obj, const sc_units::Unit unit) {
  nb::module_ numpy = nb::module_::import_("numpy");
  const auto arr = numpy.attr("asarray")(obj);
  const auto dtype = arr.attr("dtype");
  if (nb::cast<char>(dtype.attr("kind")) == 'i') {
    return; // just assume we can assign from int
  }
  const auto np_unit =
      parse_datetime_dtype(std::string(nb::str(dtype.attr("name")).c_str()));
  if (np_unit != unit) {
    std::ostringstream oss;
    oss << "Unable to assign datetime with unit " << to_string(np_unit)
        << " to " << to_string(unit);
    throw std::invalid_argument(oss.str());
  }
}

scipp::core::time_point make_time_point(const nb::object &buffer,
                                        const int64_t scale) {
  // buffer.cast does not always work because numpy.datetime64.__int__
  // delegates to datetime.datetime if the unit is larger than ns and
  // that cannot be converted to long.
  using PyType = typename ElementTypeMap<core::time_point>::PyType;
  nb::module_ numpy = nb::module_::import_("numpy");
  nb::object np_dtype = numpy.attr("dtype")(numpy.attr("int64"));
  return core::time_point{nb::cast<PyType>(buffer.attr("astype")(np_dtype)) *
                          scale};
}
