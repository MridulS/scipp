// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <optional>

#include "scipp/dataset/arithmetic.h"
#include "scipp/dataset/astype.h"
#include "scipp/dataset/generated_comparison.h"
#include "scipp/dataset/generated_logical.h"
#include "scipp/dataset/generated_math.h"
#include "scipp/dataset/to_unit.h"
#include "scipp/dataset/util.h"
#include "scipp/units/except.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/astype.h"
#include "scipp/variable/comparison.h"
#include "scipp/variable/logical.h"
#include "scipp/variable/pow.h"
#include "scipp/variable/to_unit.h"

#include "dtype.h"
#include "format.h"
#include "nanobind.h"

namespace nb = nanobind;

template <class T, class... Ignored>
void bind_common_operators(nanobind::class_<T, Ignored...> &c) {
  c.def("__abs__", [](const T &self) { return abs(self); });
  c.def("__repr__", [](const T &self) { return to_string(self); });
  c.def("__bool__", [](const T &self) {
    if constexpr (std::is_same_v<T, scipp::Variable>) {
      if (self.unit() != scipp::sc_units::none)
        throw scipp::except::UnitError(
            "The truth value of a variable with unit is undefined.");
      return self.template value<bool>() == true;
    }
    throw std::runtime_error("The truth value of a variable, data array, or "
                             "dataset is ambiguous. Use any() or all().");
  });
  c.def(
      "copy",
      [](const T &self, const bool deep) { return deep ? copy(self) : self; },
      nb::arg("deep") = true, nb::call_guard<nb::gil_scoped_release>(),
      R"(
      Return a (by default deep) copy.

      If `deep=True` (the default), a deep copy is made. Otherwise, a shallow
      copy is made, and the returned data (and meta data) values are new views
      of the data and meta data values of this object.

      Examples
      --------

        >>> import scipp as sc
        >>> var = sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='m')
        >>> var_copy = var.copy()
        >>> var_copy.values[0] = 999.0
        >>> var  # Original unchanged
        <scipp.Variable> (x: 3)    float64              [m]  [1, 2, 3]
)");
  c.def(
      "__copy__", [](const T &self) { return self; },
      nb::call_guard<nb::gil_scoped_release>(), "Return a (shallow) copy.");
  c.def(
       "__deepcopy__",
       [](const T &self, const nb::dict &) { return copy(self); },
       nb::call_guard<nb::gil_scoped_release>(), "Return a (deep) copy.")
      .def(
          "__sizeof__",
          [](const T &self) {
            return size_of(self, scipp::SizeofTag::ViewOnly);
          },
          R"doc(Return the size of the object in bytes.

The size includes the object itself and all arrays contained in it.
But arrays may be counted multiple times if components share buffers,
e.g. multiple coordinates referencing the same memory.
Conversely, the size may be underestimated. Especially, but not only,
with dtype=PyObject.

This function only includes memory of the current slice. Use
``underlying_size`` to get the full memory size of the underlying structure.)doc")
      .def(
          "underlying_size",
          [](const T &self) {
            return size_of(self, scipp::SizeofTag::Underlying);
          },
          R"doc(Return the size of the object in bytes.

The size includes the object itself and all arrays contained in it.
But arrays may be counted multiple times if components share buffers,
e.g. multiple coordinates referencing the same memory.
Conversely, the size may be underestimated. Especially, but not only,
with dtype=PyObject.

This function includes all memory of the underlying buffers. Use
``__sizeof__`` to get the size of the current slice only.)doc");
}

template <class T, class... Ignored>
void bind_astype(nb::class_<T, Ignored...> &c) {
  c.def(
      "astype",
      [](const T &self, const nb::object &type, const bool copy) {
        const auto [scipp_dtype, dtype_unit] =
            cast_dtype_and_unit(type, DefaultUnit{});
        if (dtype_unit.has_value() &&
            (dtype_unit != scipp::sc_units::one && dtype_unit != self.unit())) {
          throw scipp::except::UnitError(scipp::python::format(
              "Conversion of units via the dtype is not allowed. Occurred when "
              "trying to change dtype from ",
              self.dtype(), " to ", type,
              ". Use to_unit in combination with astype."));
        }
        [[maybe_unused]] nb::gil_scoped_release release;
        return astype(self, scipp_dtype,
                      copy ? scipp::CopyPolicy::Always
                           : scipp::CopyPolicy::TryAvoid);
      },
      nb::arg("type"), nb::kw_only(), nb::arg("copy") = true,
      R"(
        Converts a Variable or DataArray to a different dtype.

        If the dtype is unchanged and ``copy`` is `False`, the object
        is returned without making a deep copy.

        :param type: Target dtype.
        :param copy: If `False`, return the input object if possible.
                     If `True`, the function always returns a new object.
        :raises: If the data cannot be converted to the requested dtype.
        :return: New variable or data array with specified dtype.
        :rtype: Union[scipp.Variable, scipp.DataArray]

        Examples
        --------

        Convert integers to floating-point:

          >>> import scipp as sc
          >>> var = sc.array(dims=['x'], values=[1, 2, 3])
          >>> var.dtype
          DType('int64')
          >>> var.astype('float64')
          <scipp.Variable> (x: 3)    float64  [dimensionless]  [1, 2, 3]

        Convert floats to integers (truncates toward zero):

          >>> sc.array(dims=['x'], values=[1.7, 2.3, -0.8]).astype('int64')
          <scipp.Variable> (x: 3)      int64  [dimensionless]  [1, 2, 0]

        With ``copy=False``, avoid unnecessary copies when dtype is unchanged:

          >>> var_f64 = sc.array(dims=['x'], values=[1.0, 2.0], dtype='float64')
          >>> var_f64.astype('float64', copy=False).dtype
          DType('float64')
)");
}

/// pybind11 set __hash__ = None whenever __eq__ was bound to a class that
/// does not define __hash__ itself (mirroring Python class semantics);
/// nanobind does not, so restore unhashability explicitly.
template <class T, class... Ignored>
void unset_default_hash(nanobind::class_<T, Ignored...> &c) {
  if (!nb::cast<bool>(c.attr("__dict__").attr("__contains__")("__hash__")))
    c.attr("__hash__") = nb::none();
}

template <class Other, class T, class... Ignored>
void bind_inequality_to_operator(nanobind::class_<T, Ignored...> &c) {
  c.def(
      "__eq__", [](const T &a, const Other &b) { return a == b; },
      nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
  c.def(
      "__ne__", [](const T &a, const Other &b) { return a != b; },
      nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
  unset_default_hash(c);
}

struct Identity {
  template <class T> const T &operator()(const T &x) const noexcept {
    return x;
  }
};

template <class RHSSetup> struct OpBinder {
  // Returns a callable for use with nanobind's c.def() that implements an
  // in-place operator. The returned lambda:
  // 1. Extracts the C++ reference from the nb::object while the Python thread
  //    state is still attached (the cast calls into the Python C API; on
  //    free-threaded Python it may also need to lock the object).
  // 2. Releases the GIL around the pure C++ operation.
  // 3. Returns the self object by value, preserving Python-level identity.
  //    The reference count is only touched after the scoped release has
  //    reacquired the GIL, so this is safe.
  template <class T, class Other, class Op> static auto inplace_op(Op op) {
    return [op](nb::object &a, Other &b) -> nb::object {
      auto &self = nb::cast<T &>(a);
      {
        nb::gil_scoped_release release;
        op(self, RHSSetup{}(b));
      }
      return a;
    };
  }

  template <class Other, class T, class... Ignored>
  static void in_place_binary(nanobind::class_<T, Ignored...> &c) {
    using namespace scipp;
    c.def("__iadd__", inplace_op<T, Other>([](auto &a, auto &&b) { a += b; }),
          nb::is_operator());
    c.def("__isub__", inplace_op<T, Other>([](auto &a, auto &&b) { a -= b; }),
          nb::is_operator());
    c.def("__imul__", inplace_op<T, Other>([](auto &a, auto &&b) { a *= b; }),
          nb::is_operator());
    c.def("__itruediv__",
          inplace_op<T, Other>([](auto &a, auto &&b) { a /= b; }),
          nb::is_operator());
    if constexpr (!(std::is_same_v<T, Dataset> ||
                    std::is_same_v<Other, Dataset>)) {
      c.def("__imod__", inplace_op<T, Other>([](auto &a, auto &&b) { a %= b; }),
            nb::is_operator());
      c.def("__ifloordiv__", inplace_op<T, Other>([](auto &a, auto &&b) {
              floor_divide_equals(a, b);
            }),
            nb::is_operator());
      if constexpr (!(std::is_same_v<T, DataArray> ||
                      std::is_same_v<Other, DataArray>)) {
        c.def(
            "__ipow__",
            [](T &base, Other &exponent) -> T & {
              return pow(base, RHSSetup{}(exponent), base);
            },
            nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
      }
    }
  }

  template <class Other, class T, class... Ignored>
  static void binary(nanobind::class_<T, Ignored...> &c) {
    using namespace scipp;
    c.def(
        "__add__", [](const T &a, const Other &b) { return a + RHSSetup{}(b); },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    c.def(
        "__sub__", [](const T &a, const Other &b) { return a - RHSSetup{}(b); },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    c.def(
        "__mul__", [](const T &a, const Other &b) { return a * RHSSetup{}(b); },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    c.def(
        "__truediv__",
        [](const T &a, const Other &b) { return a / RHSSetup{}(b); },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    if constexpr (!(std::is_same_v<T, Dataset> ||
                    std::is_same_v<Other, Dataset>)) {
      c.def(
          "__floordiv__",
          [](const T &a, const Other &b) {
            return floor_divide(a, RHSSetup{}(b));
          },
          nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
      c.def(
          "__mod__",
          [](const T &a, const Other &b) { return a % RHSSetup{}(b); },
          nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
      c.def(
          "__pow__",
          [](const T &base, const Other &exponent) {
            return pow(base, RHSSetup{}(exponent));
          },
          nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    }
  }

  template <class Other, class T, class... Ignored>
  static void reverse_binary(nanobind::class_<T, Ignored...> &c) {
    using namespace scipp;
    c.def(
        "__radd__", [](const T &a, const Other b) { return RHSSetup{}(b) + a; },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    c.def(
        "__rsub__", [](const T &a, const Other b) { return RHSSetup{}(b)-a; },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    c.def(
        "__rmul__", [](const T &a, const Other b) { return RHSSetup{}(b)*a; },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    c.def(
        "__rtruediv__",
        [](const T &a, const Other b) { return RHSSetup{}(b) / a; },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    if constexpr (!(std::is_same_v<T, Dataset> ||
                    std::is_same_v<Other, Dataset>)) {
      c.def(
          "__rfloordiv__",
          [](const T &a, const Other &b) {
            return floor_divide(RHSSetup{}(b), a);
          },
          nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
      c.def(
          "__rmod__",
          [](const T &a, const Other &b) { return RHSSetup{}(b) % a; },
          nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
      c.def(
          "__rpow__",
          [](const T &exponent, const Other &base) {
            return pow(RHSSetup{}(base), exponent);
          },
          nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    }
  }

  template <class Other, class T, class... Ignored>
  static void comparison(nanobind::class_<T, Ignored...> &c) {
    c.def(
        "__eq__", [](const T &a, Other &b) { return equal(a, RHSSetup{}(b)); },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    unset_default_hash(c);
    c.def(
        "__ne__",
        [](const T &a, Other &b) { return not_equal(a, RHSSetup{}(b)); },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    c.def(
        "__lt__", [](const T &a, Other &b) { return less(a, RHSSetup{}(b)); },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    c.def(
        "__gt__",
        [](const T &a, Other &b) { return greater(a, RHSSetup{}(b)); },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    c.def(
        "__le__",
        [](const T &a, Other &b) { return less_equal(a, RHSSetup{}(b)); },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
    c.def(
        "__ge__",
        [](const T &a, Other &b) { return greater_equal(a, RHSSetup{}(b)); },
        nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
  }
};

template <class Other, class T, class... Ignored>
static void bind_in_place_binary(nanobind::class_<T, Ignored...> &c) {
  OpBinder<Identity>::in_place_binary<Other>(c);
}

template <class Other, class T, class... Ignored>
static void bind_binary(nanobind::class_<T, Ignored...> &c) {
  OpBinder<Identity>::binary<Other>(c);
}

template <class Other, class T, class... Ignored>
static void bind_comparison(nanobind::class_<T, Ignored...> &c) {
  OpBinder<Identity>::comparison<Other>(c);
}

/// Convert a Python number to a 0-d dimensionless Variable, mirroring
/// pybind11's overload resolution over the previous [double, int64_t]
/// scalar operator overloads: floats (including np.float64, a subclass of
/// float) become float64; objects providing __index__ (int, bool, numpy
/// integers and bools) become int64; any other number (np.float32, ints
/// too large for int64) becomes float64. Returns nullopt for non-numbers
/// so that operators can return NotImplemented, which enables reflected
/// operations on the other operand. Requires the GIL.
inline std::optional<scipp::Variable> to_scalar_operand(const nb::object &obj) {
  using namespace scipp;
  if (PyFloat_Check(obj.ptr()))
    return nb::cast<double>(obj) * sc_units::one;
  if (PyIndex_Check(obj.ptr())) {
    // PyNumber_Index implements true __index__ semantics; in contrast,
    // __int__ would silently truncate, e.g., size-1 float arrays.
    if (nb::object index = nb::steal(PyNumber_Index(obj.ptr()));
        index.is_valid()) {
      if (int64_t value = 0; nb::try_cast<int64_t>(index, value))
        return value * sc_units::one;
      // Too large for int64, fall through to double like pybind11's
      // converting double overload did.
    } else {
      PyErr_Clear();
    }
  }
  if (double value = 0.0; nb::try_cast<double>(obj, value))
    return value * sc_units::one;
  return std::nullopt;
}

/// Binds operators accepting Python number scalars via a single nb::object
/// overload, registered after the overloads for bound types (which win the
/// dispatch for Variable etc. operands). A plain [double, int64_t] overload
/// pair, as used with pybind11, does not work with nanobind: its exact-match
/// first dispatch pass rejects numpy scalars and bools, and the convert pass
/// would then turn np.int64 into double, silently changing the result dtype.
struct ScalarOpBinder {
  template <class T, class Op> static auto scalar_op(Op op) {
    // No call_guard: converting the operand requires the GIL; only the C++
    // operation releases it.
    return [op](const T &a, const nb::object &b) -> nb::object {
      const auto operand = to_scalar_operand(b);
      if (!operand)
        return nb::borrow(nb::handle(Py_NotImplemented));
      auto result = [&]() {
        nb::gil_scoped_release release;
        return op(a, *operand);
      }();
      return nb::cast(std::move(result));
    };
  }

  template <class T, class Op> static auto scalar_inplace_op(Op op) {
    return [op](nb::object &obj, const nb::object &b) -> nb::object {
      const auto operand = to_scalar_operand(b);
      if (!operand)
        return nb::borrow(nb::handle(Py_NotImplemented));
      auto &self = nb::cast<T &>(obj);
      {
        nb::gil_scoped_release release;
        op(self, *operand);
      }
      return obj;
    };
  }

  template <class T, class... Ignored>
  static void binary(nanobind::class_<T, Ignored...> &c) {
    using namespace scipp;
    c.def("__add__",
          scalar_op<T>([](const T &a, const Variable &b) { return a + b; }),
          nb::is_operator());
    c.def("__sub__",
          scalar_op<T>([](const T &a, const Variable &b) { return a - b; }),
          nb::is_operator());
    c.def("__mul__",
          scalar_op<T>([](const T &a, const Variable &b) { return a * b; }),
          nb::is_operator());
    c.def("__truediv__",
          scalar_op<T>([](const T &a, const Variable &b) { return a / b; }),
          nb::is_operator());
    if constexpr (!std::is_same_v<T, Dataset>) {
      c.def("__floordiv__", scalar_op<T>([](const T &a, const Variable &b) {
              return floor_divide(a, b);
            }),
            nb::is_operator());
      c.def("__mod__",
            scalar_op<T>([](const T &a, const Variable &b) { return a % b; }),
            nb::is_operator());
      c.def("__pow__", scalar_op<T>([](const T &base, const Variable &exp) {
              return pow(base, exp);
            }),
            nb::is_operator());
    }
  }

  template <class T, class... Ignored>
  static void reverse_binary(nanobind::class_<T, Ignored...> &c) {
    using namespace scipp;
    c.def("__radd__",
          scalar_op<T>([](const T &a, const Variable &b) { return b + a; }),
          nb::is_operator());
    c.def("__rsub__",
          scalar_op<T>([](const T &a, const Variable &b) { return b - a; }),
          nb::is_operator());
    c.def("__rmul__",
          scalar_op<T>([](const T &a, const Variable &b) { return b * a; }),
          nb::is_operator());
    c.def("__rtruediv__",
          scalar_op<T>([](const T &a, const Variable &b) { return b / a; }),
          nb::is_operator());
    if constexpr (!std::is_same_v<T, Dataset>) {
      c.def("__rfloordiv__", scalar_op<T>([](const T &a, const Variable &b) {
              return floor_divide(b, a);
            }),
            nb::is_operator());
      c.def("__rmod__",
            scalar_op<T>([](const T &a, const Variable &b) { return b % a; }),
            nb::is_operator());
      c.def("__rpow__", scalar_op<T>([](const T &exp, const Variable &base) {
              return pow(base, exp);
            }),
            nb::is_operator());
    }
  }

  template <class T, class... Ignored>
  static void in_place_binary(nanobind::class_<T, Ignored...> &c) {
    using namespace scipp;
    c.def("__iadd__",
          scalar_inplace_op<T>([](T &a, const Variable &b) { a += b; }),
          nb::is_operator());
    c.def("__isub__",
          scalar_inplace_op<T>([](T &a, const Variable &b) { a -= b; }),
          nb::is_operator());
    c.def("__imul__",
          scalar_inplace_op<T>([](T &a, const Variable &b) { a *= b; }),
          nb::is_operator());
    c.def("__itruediv__",
          scalar_inplace_op<T>([](T &a, const Variable &b) { a /= b; }),
          nb::is_operator());
    if constexpr (!std::is_same_v<T, Dataset>) {
      c.def("__imod__",
            scalar_inplace_op<T>([](T &a, const Variable &b) { a %= b; }),
            nb::is_operator());
      c.def("__ifloordiv__", scalar_inplace_op<T>([](T &a, const Variable &b) {
              floor_divide_equals(a, b);
            }),
            nb::is_operator());
      if constexpr (!std::is_same_v<T, DataArray>) {
        c.def("__ipow__",
              scalar_inplace_op<T>(
                  [](T &base, const Variable &exp) { pow(base, exp, base); }),
              nb::is_operator());
      }
    }
  }

  template <class T, class... Ignored>
  static void comparison(nanobind::class_<T, Ignored...> &c) {
    using namespace scipp;
    c.def("__eq__", scalar_op<T>([](const T &a, const Variable &b) {
            return equal(a, b);
          }),
          nb::is_operator());
    unset_default_hash(c);
    c.def("__ne__", scalar_op<T>([](const T &a, const Variable &b) {
            return not_equal(a, b);
          }),
          nb::is_operator());
    c.def("__lt__", scalar_op<T>([](const T &a, const Variable &b) {
            return less(a, b);
          }),
          nb::is_operator());
    c.def("__gt__", scalar_op<T>([](const T &a, const Variable &b) {
            return greater(a, b);
          }),
          nb::is_operator());
    c.def("__le__", scalar_op<T>([](const T &a, const Variable &b) {
            return less_equal(a, b);
          }),
          nb::is_operator());
    c.def("__ge__", scalar_op<T>([](const T &a, const Variable &b) {
            return greater_equal(a, b);
          }),
          nb::is_operator());
  }
};

template <class T, class... Ignored>
void bind_in_place_binary_scalars(nanobind::class_<T, Ignored...> &c) {
  ScalarOpBinder::in_place_binary(c);
}

template <class T, class... Ignored>
void bind_binary_scalars(nanobind::class_<T, Ignored...> &c) {
  ScalarOpBinder::binary(c);
}

template <class T, class... Ignored>
static void bind_reverse_binary_scalars(nanobind::class_<T, Ignored...> &c) {
  ScalarOpBinder::reverse_binary(c);
}

template <class T, class... Ignored>
void bind_comparison_scalars(nanobind::class_<T, Ignored...> &c) {
  ScalarOpBinder::comparison(c);
}

template <class T, class... Ignored>
void bind_unary(nanobind::class_<T, Ignored...> &c) {
  c.def(
      "__neg__", [](const T &a) { return -a; }, nb::is_operator(),
      nb::call_guard<nb::gil_scoped_release>());
}

template <class T, class... Ignored>
void bind_boolean_unary(nanobind::class_<T, Ignored...> &c) {
  c.def(
      "__invert__", [](const T &a) { return ~a; }, nb::is_operator(),
      nb::call_guard<nb::gil_scoped_release>());
}

template <class Other, class T, class... Ignored>
void bind_logical(nanobind::class_<T, Ignored...> &c) {
  using T1 = const T;
  using T2 = const Other;
  c.def(
      "__or__", [](const T1 &a, const T2 &b) { return a | b; },
      nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
  c.def(
      "__xor__", [](const T1 &a, const T2 &b) { return a ^ b; },
      nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
  c.def(
      "__and__", [](const T1 &a, const T2 &b) { return a & b; },
      nb::is_operator(), nb::call_guard<nb::gil_scoped_release>());
  using InplaceOp = OpBinder<Identity>;
  c.def("__ior__",
        InplaceOp::inplace_op<T, T2>([](auto &a, auto &&b) { a |= b; }),
        nb::is_operator());
  c.def("__ixor__",
        InplaceOp::inplace_op<T, T2>([](auto &a, auto &&b) { a ^= b; }),
        nb::is_operator());
  c.def("__iand__",
        InplaceOp::inplace_op<T, T2>([](auto &a, auto &&b) { a &= b; }),
        nb::is_operator());
}
