// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/dataset/except.h"
#include "scipp/dataset/dataset.h"

namespace scipp::dataset::expect {

void coordsAreSuperset(const DataArrayConstView &a,
                       const DataArrayConstView &b) {
  const auto &a_coords = a.coords();
  for (const auto b_coord : b.coords())
    if (a_coords[b_coord.first] != b_coord.second)
      throw except::CoordMismatchError(*a_coords.find(b_coord.first), b_coord);
}

void isKey(const VariableConstView &key) {
  if (key.dims().ndim() != 1)
    throw except::DimensionError(
        "Coord for binning or grouping must be 1-dimensional");
  if (key.hasVariances())
    throw except::VariancesError(
        "Coord for binning or grouping cannot have variances");
}

} // namespace scipp::dataset::expect
