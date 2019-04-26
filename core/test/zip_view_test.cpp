// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "range/v3/algorithm.hpp"

#include "test_macros.h"

#include "dataset.h"
#include "zip_view.h"

using namespace scipp::core;

TEST(ZipView, construct_fail) {
  Dataset d;

  d.insert(Coord::X, {Dim::X, 3});
  d.insert(Data::Value, "", {Dim::X, 3});
  EXPECT_THROW_MSG(
      ZipView<std::remove_cv_t<decltype(Coord::X)>> view(d), std::runtime_error,
      "ZipView must be constructed based on *all* variables in a dataset.");
  d.erase(Data::Value);

  d.insert(Data::Value, "", {});
  EXPECT_THROW_MSG((ZipView<std::remove_cv_t<decltype(Coord::X)>,
                            std::remove_cv_t<decltype(Data::Value)>>(d)),
                   std::runtime_error,
                   "ZipView supports only datasets where all variables are "
                   "1-dimensional.");
  d.erase(Data::Value);

  d.insert(Coord::Y, {Dim::Y, 3});
  EXPECT_THROW_MSG((ZipView<std::remove_cv_t<decltype(Coord::X)>,
                            std::remove_cv_t<decltype(Coord::Y)>>(d)),
                   std::runtime_error,
                   "ZipView supports only 1-dimensional datasets.");
}

TEST(ZipView, construct) {
  Dataset d;
  d.insert(Coord::X, {Dim::X, 3});
  EXPECT_NO_THROW(ZipView<std::remove_cv_t<decltype(Coord::X)>> view(d));
}

TEST(ZipView, push_back_1_variable) {
  Dataset d;
  d.insert(Coord::X, {Dim::X, 3});
  ZipView<std::remove_cv_t<decltype(Coord::X)>> view(d);
  view.push_back({1.1});
  ASSERT_EQ(d.get(Coord::X).size(), 4);
  ASSERT_EQ(d(Coord::X).dimensions().size(0), 4);
  view.push_back(2.2);
  ASSERT_EQ(d.get(Coord::X).size(), 5);
  ASSERT_EQ(d(Coord::X).dimensions().size(0), 5);
  const auto data = d.get(Coord::X);
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 0.0);
  EXPECT_EQ(data[2], 0.0);
  EXPECT_EQ(data[3], 1.1);
  EXPECT_EQ(data[4], 2.2);
}

TEST(ZipView, push_back_2_variables) {
  Dataset d;
  d.insert(Coord::X, {Dim::X, 2});
  d.insert(Data::Value, "", {Dim::X, 2});
  ZipView<std::remove_cv_t<decltype(Coord::X)>,
          std::remove_cv_t<decltype(Data::Value)>>
      view(d);
  view.push_back({1.1, 1.2});
  ASSERT_EQ(d.get(Coord::X).size(), 3);
  ASSERT_EQ(d(Coord::X).dimensions().size(0), 3);
  view.push_back({2.2, 2.3});
  ASSERT_EQ(d.get(Coord::X).size(), 4);
  ASSERT_EQ(d(Coord::X).dimensions().size(0), 4);

  const auto coord = d.get(Coord::X);
  EXPECT_EQ(coord[0], 0.0);
  EXPECT_EQ(coord[1], 0.0);
  EXPECT_EQ(coord[2], 1.1);
  EXPECT_EQ(coord[3], 2.2);
  const auto data = d.get(Data::Value);
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 0.0);
  EXPECT_EQ(data[2], 1.2);
  EXPECT_EQ(data[3], 2.3);
}

TEST(ZipView, std_algorithm_generate_n_with_back_inserter) {
  Dataset d;
  d.insert(Coord::X, {Dim::X, 0});
  d.insert(Data::Value, "", {Dim::X, 0});

  ZipView<std::remove_cv_t<decltype(Coord::X)>,
          std::remove_cv_t<decltype(Data::Value)>>
      view(d);

  std::mt19937 rng;
  std::generate_n(std::back_inserter(view), 5, [&] {
    const double v = rng();
    const double x = rng();
    return std::make_tuple(x, v);
  });

  ASSERT_EQ(d.get(Coord::X).size(), 5);
  ASSERT_EQ(d(Coord::X).dimensions().size(0), 5);
  ASSERT_EQ(d.get(Data::Value).size(), 5);
  ASSERT_EQ(d(Data::Value).dimensions().size(0), 5);

  rng = std::mt19937{};
  for (const auto x : d.get(Coord::X)) {
    rng();
    EXPECT_EQ(x, rng());
  }
  rng = std::mt19937{};
  for (const auto v : d.get(Data::Value)) {
    EXPECT_EQ(v, rng());
    rng();
  }
}

TEST(ZipView, iterator_1_variable) {
  Dataset d;
  d.insert(Coord::X, {Dim::X, 3}, {1.0, 2.0, 3.0});
  ZipView<std::remove_cv_t<decltype(Coord::X)>> view(d);
  EXPECT_EQ(std::distance(view.begin(), view.end()), 3);
  auto it = view.begin();
  EXPECT_EQ(std::get<0>(*it++), 1.0);
  EXPECT_EQ(std::get<0>(*it++), 2.0);
  EXPECT_EQ(std::get<0>(*it++), 3.0);
  EXPECT_EQ(it, view.end());
}

TEST(ZipView, iterator_modify) {
  Dataset d;
  d.insert(Coord::X, {Dim::X, 3}, {1.0, 2.0, 3.0});
  d.insert(Data::Value, "", {Dim::X, 3}, {1.1, 2.1, 3.1});
  ZipView<std::remove_cv_t<decltype(Coord::X)>,
          std::remove_cv_t<decltype(Data::Value)>>
      view(d);

  // Note this peculiarity: `item` is returned by value but it is a proxy
  // object, i.e., it containts references that can be used to modify the
  // dataset.
  for (auto item : view)
    std::get<1>(item) *= 2.0;

  EXPECT_TRUE(equals(d.get(Coord::X), {1.0, 2.0, 3.0}));
  EXPECT_TRUE(equals(d.get(Data::Value), {2.2, 4.2, 6.2}));
}
// Two following tests disabled because of unresolved issues with
// conversion from pair to tuple as part of copy and copy_if
#ifndef _APPLE_CLANG_

TEST(ZipView, iterator_copy) {
  Dataset source;
  source.insert(Coord::X, {Dim::X, 3}, {1.0, 2.0, 3.0});
  source.insert(Data::Value, "", {Dim::X, 3}, {1.1, 2.1, 3.1});
  ZipView<std::remove_cv_t<decltype(Coord::X)>,
          std::remove_cv_t<decltype(Data::Value)>>
      source_view(source);

  Dataset d;
  d.insert(Coord::X, {Dim::X, 0});
  d.insert(Data::Value, "", {Dim::X, 0});
  ZipView<std::remove_cv_t<decltype(Coord::X)>,
          std::remove_cv_t<decltype(Data::Value)>>
      view(d);

  std::copy(source_view.begin(), source_view.end(), std::back_inserter(view));
  std::copy(source_view.begin(), source_view.end(), std::back_inserter(view));

  EXPECT_TRUE(equals(d.get(Coord::X), {1.0, 2.0, 3.0, 1.0, 2.0, 3.0}));
  EXPECT_TRUE(equals(d.get(Data::Value), {1.1, 2.1, 3.1, 1.1, 2.1, 3.1}));

  std::copy(source_view.begin(), source_view.end(), view.begin() + 1);

  EXPECT_TRUE(equals(d.get(Coord::X), {1.0, 1.0, 2.0, 3.0, 2.0, 3.0}));
  EXPECT_TRUE(equals(d.get(Data::Value), {1.1, 1.1, 2.1, 3.1, 2.1, 3.1}));
}

TEST(ZipView, iterator_copy_if) {
  Dataset source;
  source.insert(Coord::X, {Dim::X, 3}, {1.0, 2.0, 3.0});
  source.insert(Data::Value, "", {Dim::X, 3}, {1.1, 2.1, 3.1});
  ZipView<std::remove_cv_t<decltype(Coord::X)>,
          std::remove_cv_t<decltype(Data::Value)>>
      source_view(source);

  Dataset d;
  d.insert(Coord::X, {Dim::X, 0});
  d.insert(Data::Value, "", {Dim::X, 0});
  ZipView<std::remove_cv_t<decltype(Coord::X)>,
          std::remove_cv_t<decltype(Data::Value)>>
      view(d);

  std::copy_if(source_view.begin(), source_view.end(), std::back_inserter(view),
               [](const auto &item) { return std::get<1>(item) > 2.0; });

  EXPECT_TRUE(equals(d.get(Coord::X), {2.0, 3.0}));
  EXPECT_TRUE(equals(d.get(Data::Value), {2.1, 3.1}));

  std::copy_if(source_view.begin(), source_view.end(), std::back_inserter(view),
               [](const auto &item) { return std::get<1>(item) > 2.0; });

  EXPECT_TRUE(equals(d.get(Coord::X), {2.0, 3.0, 2.0, 3.0}));
  EXPECT_TRUE(equals(d.get(Data::Value), {2.1, 3.1, 2.1, 3.1}));
}
#endif // _CLANG_

TEST(ZipView, iterator_sort) {
  Dataset d;
  d.insert(Coord::X, {Dim::X, 4}, {3.0, 2.0, 1.0, 0.0});
  ZipView<std::remove_cv_t<decltype(Coord::X)>> view(d);

  // Note: Unlike other std algorithms, std::sort does not work with these
  // iterators, must use ranges::sort.
  using ranges::sort;
  sort(view.begin(), view.end(), [](const auto &a, const auto &b) {
    return std::get<0>(a) < std::get<0>(b);
  });

  EXPECT_TRUE(equals(d.get(Coord::X), {0.0, 1.0, 2.0, 3.0}));
}

TEST(Zip, single_scalar_item) {
  Dataset d;
  d.insert(Coord::X, {Dim::X, 4}, {1, 2, 3, 4});

  auto zipped = zip(d, Access::Key(Coord::X));

  EXPECT_EQ(zipped.size(), 4);
  auto it = zipped.begin();
  // Could consider returning the single item by reference, instead of having a
  // size-1 proxy. In practice this is probably not used a lot, so we keep
  // things simple for now.
  EXPECT_EQ(std::get<0>(*it), 1.0);
  std::get<0>(*it) += 1.0;
  EXPECT_EQ(std::get<0>(*it++), 2.0);
  EXPECT_EQ(std::get<0>(*it++), 2.0);
  EXPECT_EQ(std::get<0>(*it++), 3.0);
  EXPECT_EQ(std::get<0>(*it++), 4.0);

  EXPECT_EQ(std::get<0>(zipped[0]), 2.0);
  EXPECT_EQ(std::get<0>(zipped[1]), 2.0);
  EXPECT_EQ(std::get<0>(zipped[2]), 3.0);
  EXPECT_EQ(std::get<0>(zipped[3]), 4.0);
}

TEST(Zip, multiple_scalar_items) {
  Dataset d;
  d.insert(Data::Value, "a", {Dim::X, 2}, {1, 2});
  d.insert(Data::Value, "b", {Dim::X, 2}, {3, 4});

  auto zipped =
      zip(d, Access::Key(Data::Value, "a"), Access::Key(Data::Value, "b"));

  EXPECT_EQ(zipped.size(), 2);
  auto it = zipped.begin();
  EXPECT_EQ(std::get<0>(*it), 1.0);
  EXPECT_EQ(std::get<1>(*it), 3.0);
  std::get<0>(*it) += 1.0;
  EXPECT_EQ(std::get<0>(*it), 2.0);
  EXPECT_EQ(std::get<1>(*it), 3.0);
  ++it;
  EXPECT_EQ(std::get<0>(*it), 2.0);
  EXPECT_EQ(std::get<1>(*it), 4.0);

  EXPECT_EQ(std::get<0>(zipped[0]), 2.0);
  EXPECT_EQ(std::get<1>(zipped[0]), 3.0);
  EXPECT_EQ(std::get<0>(zipped[1]), 2.0);
  EXPECT_EQ(std::get<1>(zipped[1]), 4.0);

  std::get<0>(zipped[0]) = 0.0;
  EXPECT_EQ(std::get<0>(zipped[0]), 0.0);
}

TEST(Zip, const_multiple_scalar_items) {
  Dataset d;
  d.insert(Data::Value, "a", {Dim::X, 2}, {1, 2});
  d.insert(Data::Value, "b", {Dim::X, 2}, {3, 4});
  const auto const_d(d);

  auto zipped = zip(const_d, Access::Key(Data::Value, "a"),
                    Access::Key(Data::Value, "b"));

  EXPECT_EQ(zipped.size(), 2);
  auto it = zipped.begin();
  EXPECT_EQ(std::get<0>(*it), 1.0);
  EXPECT_EQ(std::get<1>(*it), 3.0);
  // Modification not possible in this case.
  // std::get<0>(*it) += 1.0;
  ++it;
  EXPECT_EQ(std::get<0>(*it), 2.0);
  EXPECT_EQ(std::get<1>(*it), 4.0);
}

TEST(Zip, duplicate_key_fail) {
  Dataset d;
  d.insert(Data::Value, "a", {Dim::X, 2}, {1, 2});
  d.insert(Data::Value, "b", {Dim::X, 2}, {3, 4});

  EXPECT_THROW_MSG(
      zip(d, Access::Key(Data::Value, "a"), Access::Key(Data::Value, "a")),
      std::runtime_error, "Duplicate key.");
  EXPECT_NO_THROW(
      zip(d, Access::Key(Data::Value, "a"), Access::Key(Data::Value, "b")));
}

TEST(ConstItemZipProxy, length_mismatch_fail) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2};
  EXPECT_THROW_MSG(ConstItemZipProxy x(a, b), std::runtime_error,
                   "Cannot zip data with mismatching length.");
}

TEST(ItemZipProxy, length_mismatch_fail) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2};
  EXPECT_THROW_MSG(ItemZipProxy x(true, a, b), std::runtime_error,
                   "Cannot zip data with mismatching length.");
}

TEST(ConstItemZipProxy, from_vectors) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2, 3};
  ConstItemZipProxy proxy(a, b);
  EXPECT_EQ(std::get<0>(*proxy.begin()), 1.1);
  EXPECT_EQ(std::get<1>(*proxy.begin()), 1);
}

TEST(ItemZipProxy, from_vectors) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2, 3};
  ItemZipProxy proxy(true, a, b);
  EXPECT_EQ(std::get<0>(*proxy.begin()), 1.1);
  EXPECT_EQ(std::get<1>(*proxy.begin()), 1);
  std::get<0>(*proxy.begin()) = 0.0;
  EXPECT_EQ(std::get<0>(*proxy.begin()), 0.0);
}

TEST(ItemZipProxy, push_back) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2, 3};
  ItemZipProxy proxy(true, a, b);
  proxy.push_back(4.4, 4);
  EXPECT_EQ(std::get<0>(*(proxy.begin() + 3)), 4.4);
  EXPECT_EQ(std::get<1>(*(proxy.begin() + 3)), 4);
  proxy.push_back(*proxy.begin());
  EXPECT_EQ(std::get<0>(*(proxy.begin() + 4)), 1.1);
  EXPECT_EQ(std::get<1>(*(proxy.begin() + 4)), 1);
}

TEST(ItemZipProxy, push_back_3) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2, 3};
  std::vector<int32_t> c{3, 2, 1};
  ItemZipProxy proxy(true, a, b, c);
  proxy.push_back(4.4, 4, 1);
  EXPECT_EQ(std::get<0>(*(proxy.begin() + 3)), 4.4);
  EXPECT_EQ(std::get<1>(*(proxy.begin() + 3)), 4);
  EXPECT_EQ(std::get<2>(*(proxy.begin() + 3)), 1);
  proxy.push_back(*proxy.begin());
  EXPECT_EQ(std::get<0>(*(proxy.begin() + 4)), 1.1);
  EXPECT_EQ(std::get<1>(*(proxy.begin() + 4)), 1);
  EXPECT_EQ(std::get<2>(*(proxy.begin() + 4)), 3);
}

TEST(ItemZipProxy, push_back_duplicate_broken) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2, 3};

  // This is not allowed. We could add a check, but at this point it is not
  // clear if that is required, since creation should typically be under our
  // control, and we may want to avoid performance penalties.
  ItemZipProxy proxy(true, a, b, b);

  proxy.push_back(4.4, 4, 5);
  EXPECT_EQ(std::get<0>(*(proxy.begin() + 3)), 4.4);
  EXPECT_EQ(std::get<1>(*(proxy.begin() + 3)), 4);
  // b is now longer than a, we view the wrong element.
  EXPECT_EQ(std::get<2>(*(proxy.begin() + 3)), 4);
}

TEST(EventListsProxy, missing_field) {
  Dataset d;
  d.insert<std::vector<double>>(Data::Value, "a", {Dim::X, 4});
  d.insert<std::vector<double>>(Data::Variance, "a", {Dim::X, 4});

  EXPECT_THROW_MSG_SUBSTR(
      zip(d, Access::Key<std::vector<double>>{Data::Value, "a"},
          Access::Key<std::vector<double>>{Data::Variance, "b"}),
      std::runtime_error,
      "could not find variable with tag Data::Variance and name `b`.");
}

TEST(EventListsProxy, dimension_mismatch) {
  Dataset d;
  d.insert<std::vector<double>>(Data::Value, "a", {Dim::X, 4});
  d.insert<std::vector<double>>(Data::Variance, "a", {}, {5});

  EXPECT_THROW_MSG(zip(d, Access::Key<std::vector<double>>{Data::Value, "a"},
                       Access::Key<std::vector<double>>{Data::Variance, "a"}),
                   std::runtime_error,
                   "Variables to be zipped have mismatching dimensions, use "
                   "`zipMD()` instead.");
}

TEST(EventListsProxy, create) {
  Dataset d;
  d.insert<std::vector<double>>(Data::Value, "a", {Dim::X, 4});
  d.insert<std::vector<double>>(Data::Variance, "a", {Dim::X, 4});

  EXPECT_NO_THROW(zip(d, Access::Key<std::vector<double>>{Data::Value, "a"},
                      Access::Key<std::vector<double>>{Data::Variance, "a"}));
}

TEST(EventListsProxy, const_create) {
  Dataset d;
  d.insert<std::vector<double>>(Data::Value, "a", {Dim::X, 4});
  d.insert<std::vector<double>>(Data::Variance, "a", {Dim::X, 4});
  const auto const_d(d);

  EXPECT_NO_THROW(zip(const_d,
                      Access::Key<std::vector<double>>{Data::Value, "a"},
                      Access::Key<std::vector<double>>{Data::Variance, "a"}));
}

TEST(EventListsProxy, item_access) {
  Dataset d;
  d.insert<std::vector<double>>(Data::Value, "a", {Dim::X, 4});
  d.insert<std::vector<double>>(Data::Variance, "a", {Dim::X, 4});

  auto eventLists = zip(d, Access::Key<std::vector<double>>{Data::Value, "a"},
                        Access::Key<std::vector<double>>{Data::Variance, "a"});

  ASSERT_EQ(eventLists.size(), 4);
  auto events = *eventLists.begin();
  events.push_back(1.0, 2.0);
  EXPECT_EQ(d.span<std::vector<double>>(Data::Value, "a")[0].size(), 1ul);
  EXPECT_EQ(d.span<std::vector<double>>(Data::Value, "a")[1].size(), 0ul);
}

TEST(EventListsProxy, range_based_for) {
  // rvalue overloads of begin and end are deleted, just checking if we can
  // nevertheless use a range-based for loop.
  Dataset d;
  d.insert<std::vector<double>>(Data::Value, "a", {Dim::X, 4});
  d.insert<std::vector<double>>(Data::Variance, "a", {Dim::X, 4});
  for (auto el : zip(d, Access::Key<std::vector<double>>{Data::Value, "a"},
                     Access::Key<std::vector<double>>{Data::Variance, "a"})) {
    EXPECT_EQ(el.size(), 0);
  }
}

TEST(EventListsProxy, item_access_prevented_if_partial_proxy) {
  Dataset d;
  d.insert<std::vector<double>>(Data::Value, "a", {Dim::X, 4});
  d.insert<std::vector<double>>(Data::Variance, "a", {Dim::X, 4});

  auto eventLists = zip(d, Access::Key<std::vector<double>>{Data::Value, "a"});

  ASSERT_EQ(eventLists.size(), 4);
  auto events = *eventLists.begin();
  EXPECT_THROW_MSG(events.push_back(1.0), std::runtime_error,
                   "Event list cannot be resized via an incomplete proxy.");
}

TEST(EventListsProxy, const_item_access) {
  Dataset d;
  d.insert<std::vector<double>>(Data::Value, "a", {Dim::X, 4});
  d.insert<std::vector<double>>(Data::Variance, "a", {Dim::X, 4});
  {
    auto eventLists =
        zip(d, Access::Key<std::vector<double>>{Data::Value, "a"},
            Access::Key<std::vector<double>>{Data::Variance, "a"});

    ASSERT_EQ(eventLists.size(), 4);
    auto events = *eventLists.begin();
    events.push_back(1.0, 2.0);
  }

  const auto const_d(d);

  auto eventLists =
      zip(const_d, Access::Key<std::vector<double>>{Data::Value, "a"},
          Access::Key<std::vector<double>>{Data::Variance, "a"});
  auto events = *eventLists.begin();
  EXPECT_EQ(std::get<0>(*events.begin()), 1.0);
}