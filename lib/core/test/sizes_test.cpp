// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "test_macros.h"

#include "scipp/core/sizes.h"

using namespace scipp;

using SmallMap =
    scipp::core::small_stable_map<Dim, scipp::index, core::NDIM_OP_MAX>;

TEST(SmallMapTest, empty_size) {
  SmallMap map;
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(map.size(), 0);
}

class SizesTest : public ::testing::Test {
protected:
};

TEST_F(SizesTest, 0d) {
  Sizes sizes;
  EXPECT_TRUE(sizes.empty());
  EXPECT_EQ(sizes.size(), 0);
  EXPECT_EQ(std::distance(sizes.begin(), sizes.end()), 0);
  EXPECT_EQ(std::distance(sizes.rbegin(), sizes.rend()), 0);
  EXPECT_FALSE(sizes.contains(Dim::X));
}

TEST_F(SizesTest, 1d) {
  Sizes sizes;
  sizes.set(Dim::X, 2);
  EXPECT_FALSE(sizes.empty());
  EXPECT_EQ(sizes.size(), 1);
  EXPECT_EQ(std::distance(sizes.begin(), sizes.end()), 1);
  EXPECT_EQ(std::distance(sizes.rbegin(), sizes.rend()), 1);
  EXPECT_EQ(*sizes.begin(), Dim::X);
  EXPECT_EQ(*sizes.rbegin(), Dim::X);
  EXPECT_TRUE(sizes.contains(Dim::X));
  EXPECT_EQ(sizes[Dim::X], 2);
}

TEST_F(SizesTest, 2d) {
  Sizes sizes;
  sizes.set(Dim::X, 2);
  sizes.set(Dim::Y, 3);
  EXPECT_FALSE(sizes.empty());
  EXPECT_EQ(sizes.size(), 2);
  EXPECT_EQ(std::distance(sizes.begin(), sizes.end()), 2);
  EXPECT_EQ(std::distance(sizes.rbegin(), sizes.rend()), 2);
  EXPECT_EQ(*sizes.begin(), Dim::X);
  EXPECT_EQ(*sizes.rbegin(), Dim::Y);
  auto it = sizes.begin();
  EXPECT_TRUE(*it == Dim::X || *it == Dim::Y);
  ++it;
  EXPECT_TRUE(*it == Dim::X || *it == Dim::Y);
  EXPECT_TRUE(sizes.contains(Dim::X));
  EXPECT_TRUE(sizes.contains(Dim::Y));
  EXPECT_EQ(sizes[Dim::X], 2);
  EXPECT_EQ(sizes[Dim::Y], 3);
}

TEST_F(SizesTest, many_dims) {
  Sizes sizes;
  sizes.set(Dim("axis-0"), 2);
  sizes.set(Dim("axis-1"), 2);
  sizes.set(Dim("axis-2"), 2);
  sizes.set(Dim("axis-3"), 2);
  sizes.set(Dim("axis-4"), 2);
  sizes.set(Dim("axis-5"), 2);
  sizes.set(Dim("axis-6"), 2);
  sizes.set(Dim("axis-7"), 2);
  sizes.set(Dim("axis-8"), 2);
  sizes.set(Dim("axis-9"), 2);
}

TEST_F(SizesTest, comparison) {
  Sizes a;
  a.set(Dim::X, 2);
  a.set(Dim::Y, 3);
  Sizes b;
  b.set(Dim::Y, 3);
  b.set(Dim::X, 2);
  EXPECT_EQ(a, b);
}

TEST_F(SizesTest, erase) {
  Sizes sizes;
  sizes.set(Dim::X, 2);
  sizes.set(Dim::Y, 3);
  sizes.set(Dim::Z, 4);
  auto original(sizes);
  EXPECT_THROW(sizes.erase(Dim::Time), std::runtime_error);
  EXPECT_EQ(sizes, original);
  sizes.erase(Dim::X);
  Sizes yz;
  yz.set(Dim::Y, 3);
  yz.set(Dim::Z, 4);
  EXPECT_EQ(sizes, yz);
}

TEST_F(SizesTest, clear) {
  Sizes sizes;
  sizes.set(Dim::X, 2);
  sizes.set(Dim::Y, 3);
  sizes.set(Dim::Z, 4);
  sizes.clear();
  EXPECT_TRUE(sizes.empty());
  EXPECT_EQ(sizes.begin(), sizes.end());
}

TEST_F(SizesTest, slice_none) {
  Sizes sizes;
  sizes.set(Dim::X, 2);
  sizes.set(Dim::Y, 3);
  sizes.set(Dim::Z, 4);
  EXPECT_EQ(sizes.slice({}), sizes);
}
