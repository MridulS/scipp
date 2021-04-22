// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/dataset/dataset.h"
#include "scipp/dataset/math.h"
#include "scipp/variable/math.h"

#include "dataset_test_common.h"

using namespace scipp;

TEST(DataArrayTest, reciprocal) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();
  DataArray array(dataset["data_zyx"]);
  EXPECT_EQ(reciprocal(array).data(), reciprocal(array.data()));
}
