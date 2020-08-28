// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/variable/arithmetic.h"
#include "scipp/variable/bucket_model.h"

using namespace scipp;
using namespace scipp::variable;

TEST(BucketTest, member_types) {
  static_assert(std::is_same_v<bucket<Variable>::element_type, VariableView>);
  static_assert(
      std::is_same_v<bucket<Variable>::const_element_type, VariableConstView>);
}

using Model = DataModel<bucket<Variable>>;

class BucketModelTest : public ::testing::Test {
protected:
  Dimensions dims{Dim::Y, 2};
  element_array<std::pair<scipp::index, scipp::index>> indices{{0, 2}, {2, 4}};
  Variable buffer =
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
};

TEST_F(BucketModelTest, construct) {
  Model model(dims, indices, Dim::X, buffer);
  EXPECT_THROW(Model(dims, indices, Dim::Y, buffer), except::DimensionError);
}

TEST_F(BucketModelTest, construct_negative_range_fail) {
  element_array<std::pair<scipp::index, scipp::index>> overlapping{{0, 2},
                                                                   {2, 1}};
  EXPECT_THROW(Model(dims, overlapping, Dim::X, buffer), except::SliceError);
}

TEST_F(BucketModelTest, construct_overlapping_fail) {
  element_array<std::pair<scipp::index, scipp::index>> overlapping{{0, 3},
                                                                   {2, 4}};
  EXPECT_THROW(Model(dims, overlapping, Dim::X, buffer), except::SliceError);
}

TEST_F(BucketModelTest, construct_before_begin_fail) {
  element_array<std::pair<scipp::index, scipp::index>> before_begin{{-1, 2},
                                                                    {2, 4}};
  EXPECT_THROW(Model(dims, before_begin, Dim::X, buffer), except::SliceError);
}

TEST_F(BucketModelTest, construct_beyond_end_fail) {
  element_array<std::pair<scipp::index, scipp::index>> beyond_end{{0, 2},
                                                                  {2, 5}};
  EXPECT_THROW(Model(dims, beyond_end, Dim::X, buffer), except::SliceError);
}

TEST_F(BucketModelTest, dtype) {
  Model model(dims, indices, Dim::X, buffer);
  EXPECT_NE(model.dtype(), buffer.dtype());
  EXPECT_EQ(model.dtype(), dtype<bucket<Variable>>);
}

TEST_F(BucketModelTest, variances) {
  Model model(dims, indices, Dim::X, buffer);
  EXPECT_FALSE(model.hasVariances());
  EXPECT_THROW(model.setVariances(Variable(buffer)), except::VariancesError);
  EXPECT_FALSE(model.hasVariances());
}

TEST_F(BucketModelTest, comparison) {
  EXPECT_EQ(Model(dims, indices, Dim::X, buffer),
            Model(dims, indices, Dim::X, buffer));
  EXPECT_NE(Model(dims, indices, Dim::X, buffer),
            Model({{Dim::X, Dim::Y}, {2, 1}}, indices, Dim::X, buffer));
  auto indices2 = indices;
  indices2.data()[0] = {0, 1};
  EXPECT_NE(Model(dims, indices, Dim::X, buffer),
            Model(dims, indices2, Dim::X, buffer));
  auto buffer2 = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                      Values{1, 2, 3, 4});
  element_array<std::pair<scipp::index, scipp::index>> indices3{{0, 1}, {1, 2}};
  EXPECT_NE(Model(dims, indices3, Dim::X, buffer2),
            Model(dims, indices3, Dim::Y, buffer2));
  EXPECT_NE(Model(dims, indices3, Dim::X, buffer),
            Model(dims, indices3, Dim::X, buffer2));
}

TEST_F(BucketModelTest, clone) {
  Model model(dims, indices, Dim::X, buffer);
  const auto copy = model.clone();
  EXPECT_EQ(dynamic_cast<const Model &>(*copy), model);
}

TEST_F(BucketModelTest, values) {
  Model model(dims, indices, Dim::X, buffer);
  EXPECT_EQ(*(model.values().begin() + 0), buffer.slice({Dim::X, 0, 2}));
  EXPECT_EQ(*(model.values().begin() + 1), buffer.slice({Dim::X, 2, 4}));
  (*model.values().begin()) += 2.0 * units::one;
  EXPECT_EQ(*(model.values().begin() + 0), buffer.slice({Dim::X, 2, 4}));
}

TEST_F(BucketModelTest, values_const) {
  const Model model(dims, indices, Dim::X, buffer);
  EXPECT_EQ(*(model.values().begin() + 0), buffer.slice({Dim::X, 0, 2}));
  EXPECT_EQ(*(model.values().begin() + 1), buffer.slice({Dim::X, 2, 4}));
}

TEST_F(BucketModelTest, values_non_range) {
  element_array<std::pair<scipp::index, scipp::index>> i{{2, 4}, {0, -1}};
  Model model(dims, i, Dim::X, buffer);
  EXPECT_EQ(*(model.values().begin() + 0), buffer.slice({Dim::X, 2, 4}));
  EXPECT_EQ(*(model.values().begin() + 1), buffer.slice({Dim::X, 0}));
}

TEST_F(BucketModelTest, out_of_order_indices) {
  element_array<std::pair<scipp::index, scipp::index>> reverse{{2, 4}, {0, 2}};
  Model model(dims, reverse, Dim::X, buffer);
  EXPECT_EQ(*(model.values().begin() + 0), buffer.slice({Dim::X, 2, 4}));
  EXPECT_EQ(*(model.values().begin() + 1), buffer.slice({Dim::X, 0, 2}));
}
