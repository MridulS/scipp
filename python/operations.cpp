// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "docstring.h"
#include "pybind11.h"

#include "scipp/dataset/dataset.h"
#include "scipp/dataset/sort.h"
#include "scipp/variable/operations.h"
#include "scipp/variable/sort.h"
#include "scipp/variable/util.h"

using namespace scipp;
using namespace scipp::variable;
using namespace scipp::dataset;

namespace py = pybind11;

template <typename T> void bind_dot(py::module &m) {
  auto doc = Docstring()
                 .description("Element-wise dot product.")
                 .raises("If the dtype of the input is not vector_3_float64.")
                 .returns("The dot product of the input vectors.")
                 .rtype<T>()
                 .template param<T>("x", "Input left hand side operand.")
                 .template param<T>("y", "Input right hand side operand.");
  m.def(
      "dot",
      [](const typename T::const_view_type &x,
         const typename T::const_view_type &y) { return dot(x, y); },
      py::arg("x"), py::arg("y"), py::call_guard<py::gil_scoped_release>(),
      doc.c_str());
}

template <typename T> void bind_sort(py::module &m) {
  auto doc =
      Docstring()
          .description("Sort variable along a dimension by a sort key.")
          .raises("If the key is invalid, e.g., if it has not exactly one "
                  "dimension, or if its dtype is not sortable.")
          .returns("The sorted equivalent of the input.")
          .rtype<T>()
          .template param<T>("x", "Data to be sorted")
          .template param<T>("key", "Sort key.");
  m.def(
      "sort",
      [](const typename T::const_view_type &x,
         const typename Variable::const_view_type &key) {
        return sort(x, key);
      },
      py::arg("x"), py::arg("key"), py::call_guard<py::gil_scoped_release>(),
      doc.c_str());
}

template <typename T> void bind_sort_dim(py::module &m) {
  auto doc =
      Docstring()
          .description("Sort data array along a dimension by the coordinate "
                       "values for that dimension.")
          .raises(
              "If the key is invalid, e.g., if the dimension does not exist.")
          .returns("The sorted equivalent of the input.")
          .rtype<T>()
          .template param<T>("x", "Data to be sorted")
          .param("dim", "Dimension to sort along.", "Dim");
  m.def(
      "sort",
      [](const typename T::const_view_type &x, const Dim &dim) {
        return sort(x, dim);
      },
      py::arg("x"), py::arg("dim"), py::call_guard<py::gil_scoped_release>(),
      doc.c_str());
}

auto get_sort_order(const std::string &order) {
  if (order == "ascending")
    return variable::SortOrder::Ascending;
  else if (order == "descending")
    return variable::SortOrder::Descending;
  else
    throw std::runtime_error("Sort order must be 'ascending' or 'descending'");
}

void bind_is_sorted(py::module &m) {
  m.def(
      "is_sorted",
      [](const VariableConstView &x, const Dim dim, const std::string &order) {
        return is_sorted(x, dim, get_sort_order(order));
      },
      py::arg("x"), py::arg("dim"), py::arg("order") = "ascending",
      py::call_guard<py::gil_scoped_release>(),
      Docstring()
          .description("Check if the values of a variable are sorted in.\n\nIf "
                       "'order' is 'ascending' checks if values are "
                       "non-decreasing along 'dim'. If 'order' is 'descending' "
                       "checks if values are non-increasing along 'dim'.")
          .param("x", "Variable to check.", "Variable")
          .param("dim", "Dimension along which order is checked.", "Dim")
          .param("order",
                 "Sorted order. Valid options are 'ascending' and "
                 "'descending'. Default is 'ascending'.",
                 "str")
          .returns(
              "True if the variable values are monotonously ascending or "
              "descending (depending on the requested order), False otherwise.")
          .rtype("bool")
          .c_str());
}

template <typename T> void bind_sort_variable(py::module &m) {
  auto doc = Docstring()
                 .description("Sort a Variable according to its values along "
                              "the inner dimension.")
                 .raises("If the key is not the inner dimension.")
                 .returns("The sorted equivalent of the input.")
                 .rtype<T>()
                 .template param<T>("x", "Data to be sorted")
                 .param("dim", "Dimension to sort along.", "Dim")
                 .param("order",
                        "Sorted order. Valid options are 'ascending' and "
                        "'descending'. Default is 'ascending'.",
                        "str");
  m.def(
      "sort",
      [](const typename T::const_view_type &x, const Dim dim,
         const std::string &order) {
        return sort(x, dim, get_sort_order(order));
      },
      py::arg("x"), py::arg("dim"), py::arg("order") = "ascending",
      py::call_guard<py::gil_scoped_release>(), doc.c_str());
}

template <typename T> void bind_contains_events(py::module &m) {
  m.def(
      "contains_events",
      [](const typename T::const_view_type &x) { return contains_events(x); },
      Docstring()
          .description("Return `True` if the input contains event data. "
                       "Note that data may be stored as a scalar.\n"
                       "In the case of a DataArray, this "
                       "returns `True` if any coord contains events.")
          .returns("`True` or `False`.")
          .rtype("bool")
          .param<T>("x", "Input data.")
          .c_str());
}

void init_operations(py::module &m) {
  bind_dot<Variable>(m);

  bind_sort<Variable>(m);
  bind_sort<DataArray>(m);
  bind_sort<Dataset>(m);
  bind_sort_dim<DataArray>(m);
  bind_sort_dim<Dataset>(m);

  bind_sort_variable<Variable>(m);
  bind_is_sorted(m);

  bind_contains_events<Variable>(m);
  bind_contains_events<DataArray>(m);

  m.def("values", variable::values,
        Docstring()
            .description("Return the variable without variances.")
            .seealso(":py:func:`scipp.variances`")
            .c_str(),
        py::call_guard<py::gil_scoped_release>());
  m.def("variances", variable::variances,
        Docstring()
            .description("Return variable containing the variances of the "
                         "input as values.")
            .seealso(":py:func:`scipp.values`")
            .c_str(),
        py::call_guard<py::gil_scoped_release>());
}
