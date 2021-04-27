// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/core/element_array_view.h"
#include "scipp/core/except.h"

namespace scipp::core {

namespace {
void expectCanBroadcastFromTo(const Dimensions &source,
                              const Dimensions &target) {
  if (source == target)
    return;
  for (const auto &dim : target.labels())
    if (source.contains(dim) && (source[dim] < target[dim]))
      throw except::DimensionError("Cannot broadcast/slice dimension since "
                                   "data has mismatching but smaller "
                                   "dimension extent.");
}
} // namespace

/// Construct ElementArrayViewParams.
///
/// @param offset Start offset from beginning of array.
/// @param iter_dims Dimensions to use for iteration.
/// @param strides Strides in memory, order matches that of iterDims.
/// @param bucket_params Optional, in case of view onto bucket-variable this
/// holds parameters for accessing individual buckets.
ElementArrayViewParams::ElementArrayViewParams(
    const scipp::index offset, const Dimensions &iter_dims,
    const Strides &strides, const BucketParams &bucket_params)
    : m_offset(offset), m_iterDims(iter_dims), m_strides(strides),
      m_bucketParams(bucket_params) {}

/// Construct ElementArrayViewParams from another ElementArrayViewParams, with
/// different iteration dimensions.
///
/// A good way to think of this is of a non-contiguous underlying data array,
/// e.g., since the other view may represent a slice. This also supports
/// broadcasting the slice.
ElementArrayViewParams::ElementArrayViewParams(
    const ElementArrayViewParams &other, const Dimensions &iterDims)
    : m_offset(other.m_offset), m_iterDims(iterDims),
      m_dataDims(other.m_dataDims), m_bucketParams(other.m_bucketParams) {
  expectCanBroadcastFromTo(other.m_iterDims, m_iterDims);
  // See implementation of ViewIndex regarding this relabeling.
  for (const auto &label : m_dataDims.labels())
    if (label != Dim::Invalid && !other.m_iterDims.contains(label))
      m_dataDims.relabel(m_dataDims.index(label), Dim::Invalid);
}

void ElementArrayViewParams::requireContiguous() const {
  bool is_contiguous = true;
  if (m_bucketParams) {
    is_contiguous = false;
  } else {
    scipp::index expected_stride = 1;
    for (scipp::index dim = m_iterDims.ndim() - 1; dim <= 0; --dim) {
      if (m_strides[dim] != expected_stride) {
        is_contiguous = false;
        break;
      }
      expected_stride *= m_iterDims[m_iterDims.label(dim)];
    }
  }
  if (!is_contiguous)
    throw std::runtime_error("Data is not contiguous");
}

} // namespace scipp::core
