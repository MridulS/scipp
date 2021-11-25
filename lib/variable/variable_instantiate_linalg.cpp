// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/core/eigen.h"
#include "scipp/core/spatial_transforms.h"
#include "scipp/variable/structure_array_variable.tcc"
#include "scipp/variable/structures.h"
#include "scipp/variable/variable.h"

namespace scipp::variable {

// TODO For now we use hard-coded field names and offsets. The intention is to
// generalize StructureArrayModel to support more general structures. Field
// names and sizes/offsets would then be stored as part of the model, and would
// be initialized dynamically at runtime.
template <>
constexpr auto structure_element_offset<Eigen::Vector3d> =
    [](const std::string &key) {
      static std::map<std::string, scipp::index> offsets{
          {"x", 0}, {"y", 1}, {"z", 2}};
      return offsets.at(key);
    };

template <>
constexpr auto structure_element_offset<Eigen::Matrix3d> =
    [](const std::string &key) {
      static std::map<std::string, scipp::index> offsets{
          {"xx", 0}, {"xy", 3}, {"xz", 6}, {"yx", 1}, {"yy", 4},
          {"yz", 7}, {"zx", 2}, {"zy", 5}, {"zz", 8}};
      return offsets.at(key);
    };

template <>
constexpr auto structure_element_offset<Eigen::Affine3d> =
    [](const std::string &key) -> scipp::index {
  throw except::TypeError("Not supported for Affine3d types");
};

template <>
constexpr auto structure_element_offset<scipp::core::eigen_translation_type> =
    [](const std::string &key) -> scipp::index {
  throw except::TypeError("Not supported for transform types");
};

template <>
constexpr auto structure_element_offset<scipp::core::eigen_rotation_type> =
    [](const std::string &key) -> scipp::index {
  throw except::TypeError("Not supported for transform types");
};

template <>
constexpr auto structure_element_offset<scipp::core::eigen_scaling_type> =
    [](const std::string &key) -> scipp::index {
  throw except::TypeError("Not supported for transform types");
};

template <>
constexpr auto structure_element_offset<
    scipp::index_pair> = [](const std::string &key) {
  static std::map<std::string, scipp::index> offsets{{"begin", 0}, {"end", 1}};
  return offsets.at(key);
};

std::vector<std::string> element_keys(const Variable &var) {
  if (variableFactory().elem_dtype(var) == dtype<Eigen::Vector3d>)
    return {"x", "y", "z"};
  if (variableFactory().elem_dtype(var) == dtype<Eigen::Matrix3d>)
    return {"xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"};
  if (variableFactory().elem_dtype(var) == dtype<Eigen::Affine3d>)
    throw except::TypeError("Not supported for Affine3d types");
  if (variableFactory().elem_dtype(var) ==
          dtype<scipp::core::eigen_rotation_type> ||
      variableFactory().elem_dtype(var) ==
          dtype<scipp::core::eigen_translation_type> ||
      variableFactory().elem_dtype(var) ==
          dtype<scipp::core::eigen_scaling_type>)
    throw except::TypeError("Not supported for transform types");
  if (variableFactory().elem_dtype(var) == dtype<scipp::index_pair>)
    return {"begin", "end"};
  throw except::TypeError("dtype is not structured");
}

INSTANTIATE_STRUCTURE_ARRAY_VARIABLE(vector_3_float64, Eigen::Vector3d, double)
INSTANTIATE_STRUCTURE_ARRAY_VARIABLE(matrix_3_float64, Eigen::Matrix3d, double)
INSTANTIATE_STRUCTURE_ARRAY_VARIABLE(affine_transform, Eigen::Affine3d, double)
INSTANTIATE_ELEMENT_ARRAY_VARIABLE(rotation, scipp::core::eigen_rotation_type)
INSTANTIATE_ELEMENT_ARRAY_VARIABLE(scaling, scipp::core::eigen_scaling_type)
INSTANTIATE_ELEMENT_ARRAY_VARIABLE(translation, scipp::core::eigen_translation_type)
INSTANTIATE_STRUCTURE_ARRAY_VARIABLE(index_pair, scipp::index_pair, scipp::index)

} // namespace scipp::variable
