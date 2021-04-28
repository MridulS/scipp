// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp/core/except.h"
#include "scipp/variable/variable.h"
#include "scipp/variable/visit.h"

namespace scipp::variable {

/// Apply functor to variables of given arguments.
template <class... Ts, class Op, class Var, class... Vars>
void apply_in_place(Op op, Var &&var, const Vars &... vars) {
  try {
    scipp::variable::visit<Ts...>::apply(op, var.data(), vars.data()...);
  } catch (const std::bad_variant_access &) {
    throw except::TypeError("Cannot apply operation to item dtypes: ", var,
                            vars...);
  }
}

} // namespace scipp::variable
