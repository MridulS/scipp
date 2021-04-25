// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <gtest/gtest.h>

#include "scipp/variable/variable.h"

class DenseVariablesTest : public ::testing::TestWithParam<scipp::Variable> {};
