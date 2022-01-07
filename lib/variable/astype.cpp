// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Jan-Lukas Wynen
#include <cmath>

#include "scipp/core/tag_util.h"
#include "scipp/core/time_point.h"
#include "scipp/core/transform_common.h"
#include "scipp/variable/astype.h"
#include "scipp/variable/transform.h"
#include "scipp/variable/variable.h"
#include "scipp/variable/variable_factory.h"

namespace scipp::variable {

struct MakeVariableWithType {
  using AllSourceTypes =
      std::tuple<double, float, int64_t, int32_t, bool, core::time_point>;

  template <class T> struct Maker {
    template <size_t I, class... Types> constexpr static auto source_types() {
      if constexpr (I == std::tuple_size_v<AllSourceTypes>) {
        return std::tuple<Types...>{};
      } else {
        using Next = typename std::tuple_element<I, AllSourceTypes>::type;
        if constexpr (std::is_same_v<Next, T>) {
          return source_types<I + 1, Types...>();
        } else {
          return source_types<I + 1, Types..., Next>();
        }
      }
    }

    template <class... SourceTypes>
    static Variable apply_impl(const Variable &parent,
                               std::tuple<SourceTypes...>) {
      using namespace core::transform_flags;
      constexpr auto expect_input_variances =
          conditional_flag<!core::canHaveVariances<T>()>(
              expect_no_variance_arg<0>);
      return transform<SourceTypes...>(
          parent,
          overloaded{
              expect_input_variances, [](const units::Unit &x) { return x; },
              [](const auto &x) {
                if constexpr (std::is_same_v<T, core::time_point>)
                  return T{static_cast<int64_t>(x)};
                if constexpr (is_ValueAndVariance_v<std::decay_t<decltype(x)>>)
                  return ValueAndVariance<T>{static_cast<T>(x.value),
                                             static_cast<T>(x.variance)};
                else
                  return static_cast<T>(x);
              },
              [](const core::time_point &x) {
                return static_cast<T>(x.time_since_epoch());
              }},
          "astype");
    }

    static Variable apply(const Variable &parent) {
      return apply_impl(parent, source_types<0>());
    }
  };

  static Variable make(const Variable &var, DType type) {
    return core::CallDType<double, float, int64_t, int32_t, bool,
                           core::time_point>::apply<Maker>(type, var);
  }
};

Variable astype(const Variable &var, DType type, const CopyPolicy copy) {
  return type == variableFactory().elem_dtype(var)
             ? (copy == CopyPolicy::TryAvoid ? var : variable::copy(var))
             : MakeVariableWithType::make(var, type);
}
} // namespace scipp::variable
