// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/core/dimensions.h"
#include "scipp/core/element_array_view.h"
#include "scipp/core/except.h"
#include "scipp/units/unit.h"
#include "scipp/variable/data_model.h"
#include "scipp/variable/except.h"
#include "scipp/variable/variable.h"

namespace scipp::variable {

template <class T>
Variable::Variable(const units::Unit unit, const Dimensions &dimensions,
                   T values_, std::optional<T> variances_)
    : m_unit{unit},
      m_object(std::make_unique<DataModel<typename T::value_type>>(
          std::move(dimensions), std::move(values_), std::move(variances_))) {}

template <class T> const DataModel<T> &cast(const Variable &var) {
  return requireT<const DataModel<T>>(var.data());
}

template <class T> DataModel<T> &cast(Variable &var) {
  return requireT<DataModel<T>>(var.data());
}

template <class T> ElementArrayView<const T> Variable::values() const {
  return cast<T>(*this).values();
}
template <class T> ElementArrayView<T> Variable::values() {
  return cast<T>(*this).values();
}
template <class T> ElementArrayView<const T> Variable::variances() const {
  return cast<T>(*this).variances();
}
template <class T> ElementArrayView<T> Variable::variances() {
  return cast<T>(*this).variances();
}

template <class T> ElementArrayView<const T> VariableConstView::values() const {
  return cast<T>(*m_variable).values(m_offset, m_dims, m_dataDims);
}
template <class T>
ElementArrayView<const T> VariableConstView::variances() const {
  return cast<T>(*m_variable).variances(m_offset, m_dims, m_dataDims);
}

template <class T> ElementArrayView<T> VariableView::values() const {
  return cast<T>(*m_mutableVariable).values(m_offset, m_dims, m_dataDims);
}
template <class T> ElementArrayView<T> VariableView::variances() const {
  return cast<T>(*m_mutableVariable).variances(m_offset, m_dims, m_dataDims);
}

#define INSTANTIATE_VARIABLE_BASE(name, ...)                                   \
  namespace {                                                                  \
  auto register_dtype_name_##name(                                             \
      (core::dtypeNameRegistry().emplace(dtype<__VA_ARGS__>, #name), 0));      \
  }                                                                            \
  template ElementArrayView<const __VA_ARGS__> Variable::values() const;       \
  template ElementArrayView<__VA_ARGS__> Variable::values();                   \
  template ElementArrayView<const __VA_ARGS__> VariableConstView::values()     \
      const;                                                                   \
  template ElementArrayView<__VA_ARGS__> VariableView::values() const;

/// Macro for instantiating classes and functions required for support a new
/// dtype in Variable.
#define INSTANTIATE_VARIABLE(name, ...)                                        \
  INSTANTIATE_VARIABLE_BASE(name, __VA_ARGS__)                                 \
  template Variable::Variable(const units::Unit, const Dimensions &,           \
                              element_array<__VA_ARGS__>,                      \
                              std::optional<element_array<__VA_ARGS__>>);      \
  template ElementArrayView<const __VA_ARGS__> Variable::variances() const;    \
  template ElementArrayView<__VA_ARGS__> Variable::variances();                \
  template ElementArrayView<const __VA_ARGS__> VariableConstView::variances()  \
      const;                                                                   \
  template ElementArrayView<__VA_ARGS__> VariableView::variances() const;

} // namespace scipp::variable
