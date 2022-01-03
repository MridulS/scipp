// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

#include "scipp/core/element/bin_detail.h"

#include <gtest/gtest.h>
#include <vector>

using namespace scipp;
using namespace scipp::core::element;

TEST(ElementBinUtilTest, begin_edge) {
  constexpr auto check = [](const double coord, const scipp::index expected) {
    const std::vector<double> edges{0, 1, 2, 3};
    scipp::index bin{0};
    scipp::index index;
    begin_edge(bin, index, coord, edges);
    EXPECT_EQ(index, expected);
  };
  check(-0.1, 0);
  check(0.0, 0);
  check(1.0, 1);
  check(1.5, 1);
  check(2.0, 2);
  check(2.5, 2);
  check(3.0, 2);
}

TEST(ElementBinUtilTest, end_edge) {
  constexpr auto check = [](const double coord, const scipp::index expected) {
    const std::vector<double> edges{0, 1, 2, 3};
    scipp::index bin{0};
    scipp::index index;
    end_edge(bin, index, coord, edges);
    EXPECT_EQ(index, expected);
  };
  check(-0.1, 2);
  check(0.0, 2);
  check(1.0, 2);
  check(1.5, 3);
  check(2.0, 3);
  check(2.5, 4);
  check(3.0, 4);
}
