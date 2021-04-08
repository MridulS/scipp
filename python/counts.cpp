// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
#include "scipp/dataset/counts.h"
#include "scipp/dataset/dataset.h"

#include "pybind11.h"

using namespace scipp;
using namespace scipp::dataset;

namespace py = pybind11;

void init_counts(py::module &m) {
  m.def(
      "counts_to_density",
      [](const Dataset &d, const Dim dim) { return counts::toDensity(d, dim); },
      py::arg("x"), py::arg("dim"));

  m.def(
      "counts_to_density",
      [](const DataArray &d, const Dim dim) {
        return counts::toDensity(d, dim);
      },
      py::arg("x"), py::arg("dim"));

  m.def(
      "density_to_counts",
      [](const Dataset &d, const Dim dim) {
        return counts::fromDensity(d, dim);
      },
      py::arg("x"), py::arg("dim"));

  m.def(
      "density_to_counts",
      [](const DataArray &d, const Dim dim) {
        return counts::fromDensity(d, dim);
      },
      py::arg("x"), py::arg("dim"));
}
