// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Owen Arnold
#pragma once

#include "scipp/dataset/dataset.h"

namespace scipp::dataset {

[[nodiscard]] SCIPP_DATASET_EXPORT DataArrayConstView
slice(const DataArrayConstView &data, const Dim dim,
      const VariableConstView value);

[[nodiscard]] SCIPP_DATASET_EXPORT DataArrayConstView
slice(const DataArrayConstView &data, const Dim dim,
      const VariableConstView begin, const VariableConstView end);

} // namespace scipp::dataset
