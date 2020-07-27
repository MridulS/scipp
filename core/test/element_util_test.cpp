// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)

#include "scipp/core/element/util.h"
#include "scipp/units/except.h"
#include "scipp/units/unit.h"
#include "scipp/core/string.h"

#include "fix_typed_test_suite_warnings.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace scipp::core::element;

TEST(ElementUtilTest, convertMaskedToZero_masks_special_vals) {
  EXPECT_EQ(convertMaskedToZero(1.0, true), 0.0);
  EXPECT_EQ(convertMaskedToZero(std::numeric_limits<double>::quiet_NaN(), true),
            0.0);
  EXPECT_EQ(convertMaskedToZero(std::numeric_limits<double>::infinity(), true),
            0.0);
  EXPECT_EQ(convertMaskedToZero(1.0, false), 1.0);
}

TEST(ElementUtilTest, convertMaskedToZero_ignores_unmasked) {
  EXPECT_TRUE(std::isnan(
      convertMaskedToZero(std::numeric_limits<double>::quiet_NaN(), false)));

  EXPECT_TRUE(std::isinf(
      convertMaskedToZero(std::numeric_limits<double>::infinity(), false)));
}

TEST(ElementUtilTest, convertMaskedToZero_handles_units) {
  const auto dimensionless = scipp::units::dimensionless;

  for (const auto &unit :
       {scipp::units::m, scipp::units::dimensionless, scipp::units::s}) {
    // Unit 'a' should always be preserved
    EXPECT_EQ(convertMaskedToZero(unit, dimensionless), unit);
  }
}

TEST(ElementUtilTest, convertMaskedToZero_rejects_units_with_dim) {
  const auto seconds = scipp::units::s;

  for (const auto &unit :
       {scipp::units::m, scipp::units::kg, scipp::units::s}) {
    // Unit 'b' should always be dimensionless as its a mask
    EXPECT_THROW(convertMaskedToZero(seconds, unit), scipp::except::UnitError);
  }
}

TEST(ElementUtilTest, convertMaskedToZero_accepts_all_types) {
  static_assert(
      std::is_same_v<decltype(convertMaskedToZero(bool{}, true)), bool>);
  static_assert(
      std::is_same_v<decltype(convertMaskedToZero(double{}, true)), double>);
  static_assert(
      std::is_same_v<decltype(convertMaskedToZero(float{}, true)), float>);
  static_assert(
      std::is_same_v<decltype(convertMaskedToZero(int32_t{}, true)), int32_t>);
  static_assert(
      std::is_same_v<decltype(convertMaskedToZero(int64_t{}, true)), int64_t>);
}

TEST(ElementUtilTest, convertToIsoDate_test) {
  int64_t ts(1595846471200000011);
  scipp::core::time_point date(ts);
  std::string unit("ns");
  EXPECT_EQ(to_iso_date(date, unit), "2020-07-27T10:41:11.200000011\n");

  int64_t ts2(1595846471);
  scipp::core::time_point date2(ts2);
  std::string unit2("s");
  EXPECT_EQ(to_iso_date(date2, unit2), "2020-07-27T10:41:11\n");
}
