// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <algorithm>
#include <variant>

#include "scipp/core/dtype.h"
#include "scipp/core/eigen.h"
#include "scipp/core/spatial_transforms.h"
#include "scipp/core/tag_util.h"
#include "scipp/dataset/dataset.h"
#include "scipp/dataset/except.h"
#include "scipp/variable/shape.h"
#include "scipp/variable/variable.h"
#include "scipp/variable/variable_concept.h"

#include "dtype.h"
#include "nanobind.h"
#include "numpy.h"
#include "py_object.h"
#include "unit.h"

namespace nb = nanobind;
using namespace scipp;

template <class T> void remove_variances(T &obj) {
  if constexpr (std::is_same_v<T, DataArray>)
    obj.data().setVariances(Variable());
  else
    obj.setVariances(Variable());
}

template <class T> void init_variances(T &obj) {
  if constexpr (std::is_same_v<T, DataArray>)
    obj.data().setVariances(Variable(obj.data()));
  else
    obj.setVariances(Variable(obj));
}

/// Add element size as factor to strides.
template <class T>
std::vector<scipp::index>
numpy_strides(const std::span<const scipp::index> &s) {
  std::vector<scipp::index> strides(s.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    strides[i] = sizeof(T) * s[i];
  }
  return strides;
}

template <typename T> decltype(auto) get_data_variable(T &&x) {
  if constexpr (std::is_same_v<std::decay_t<T>, scipp::Variable>) {
    return std::forward<T>(x);
  } else {
    return std::forward<T>(x).data();
  }
}

/// Return a nanobind handle to the VariableConcept of x.
/// Refers to the data variable if T is a DataArray.
template <typename T> auto get_data_variable_concept_handle(T &&x) {
  return nb::cast(get_data_variable(std::forward<T>(x)).data_handle());
}

template <class... Ts> class as_ElementArrayViewImpl;

class DataAccessHelper {
  template <class... Ts> friend class as_ElementArrayViewImpl;

  template <class Getter, class T, class View>
  static nb::object as_py_array_t_impl(View &&view) {
    auto &&var = get_data_variable(view);
    const auto &dims = view.dims();
    const auto &shape = dims.shape();
    const auto strides = numpy_strides<T>(var.strides());

    nb::module_ numpy = nb::module_::import_("numpy");

    // Get the numpy dtype string
    std::string dtype_str;
    if constexpr (std::is_same_v<T, scipp::core::time_point>) {
      dtype_str = "datetime64[" + to_numpy_time_string(view.unit()) + ']';
    } else if constexpr (std::is_same_v<T, double>) {
      dtype_str = "float64";
    } else if constexpr (std::is_same_v<T, float>) {
      dtype_str = "float32";
    } else if constexpr (std::is_same_v<T, int64_t>) {
      dtype_str = "int64";
    } else if constexpr (std::is_same_v<T, int32_t>) {
      dtype_str = "int32";
    } else if constexpr (std::is_same_v<T, bool>) {
      dtype_str = "bool";
    } else {
      dtype_str = "float64"; // fallback
    }
    nb::object np_dtype = numpy.attr("dtype")(dtype_str);

    // Get data pointer
    const T *data_ptr = Getter::template get<T>(view).data();

    // Create array shape and strides as Python lists
    nb::list py_shape;
    nb::list py_strides;
    for (size_t i = 0; i < shape.size(); ++i) {
      py_shape.append(shape[i]);
      py_strides.append(strides[i]);
    }

    // Build __array_interface__ dict
    nb::dict array_interface;
    array_interface["version"] = 3;
    array_interface["shape"] = nb::tuple(py_shape);
    array_interface["strides"] = nb::tuple(py_strides);
    array_interface["typestr"] = np_dtype.attr("str");
    array_interface["data"] = nb::make_tuple(
        reinterpret_cast<uintptr_t>(data_ptr), var.is_readonly());

    // Create a holder class instance with __array_interface__
    // We use a simple approach: create a memoryview-like object
    nb::object owner = get_data_variable_concept_handle(view);

    // Use ctypes to create a pointer that numpy can use
    nb::module_ ctypes = nb::module_::import_("ctypes");

    // Calculate total size in bytes
    scipp::index total_elements = 1;
    for (const auto s : shape) {
      total_elements *= s;
    }

    // Create ctypes array type: (c_char * total_bytes)
    nb::object c_char = ctypes.attr("c_char");
    // In Python: array_type = c_char * total_bytes
    // We need to call c_char.__mul__(total_bytes)
    nb::object array_type = c_char.attr("__mul__")(total_elements * sizeof(T));
    nb::object c_array =
        array_type.attr("from_address")(reinterpret_cast<uintptr_t>(data_ptr));

    // Create numpy array from ctypes array
    nb::object result = numpy.attr("ctypeslib").attr("as_array")(c_array);

    // Reshape and reinterpret dtype
    result = result.attr("view")(np_dtype);
    // Reshape to target shape (including empty tuple for 0-D scalars)
    result = result.attr("reshape")(nb::tuple(py_shape));

    // Apply strides if non-contiguous (use as_strided)
    bool is_contiguous = true;
    scipp::index expected_stride = sizeof(T);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      if (strides[i] != expected_stride) {
        is_contiguous = false;
        break;
      }
      expected_stride *= shape[i];
    }

    if (!is_contiguous) {
      nb::object lib = numpy.attr("lib");
      nb::object stride_tricks = lib.attr("stride_tricks");
      result = stride_tricks.attr("as_strided")(
          result, nb::arg("shape") = nb::tuple(py_shape),
          nb::arg("strides") = nb::tuple(py_strides));
    }

    if (var.is_readonly()) {
      result.attr("flags").attr("writeable") = false;
    }

    // Set the base of the numpy array to keep the underlying Variable alive.
    // We use a helper function from scipp._array_util that properly sets the
    // numpy array's base using numpy's C API.
    nb::module_ array_util = nb::module_::import_("scipp._array_util");
    result = array_util.attr("set_array_base")(result, owner);

    return result;
  }

  struct get_values {
    template <class T, class View> static constexpr auto get(View &&view) {
      return view.template values<T>();
    }
  };

  struct get_variances {
    template <class T, class View> static constexpr auto get(View &&view) {
      return view.template variances<T>();
    }
  };
};

inline void expect_scalar(const Dimensions &dims, const std::string_view name) {
  if (dims != Dimensions{}) {
    std::ostringstream oss;
    oss << "The '" << name << "' property cannot be used with non-scalar "
        << "Variables. Got dimensions " << to_string(dims) << ". Did you mean '"
        << name << "s'?";
    throw except::DimensionError(oss.str());
  }
}

template <class... Ts> class as_ElementArrayViewImpl {
  using get_values = DataAccessHelper::get_values;
  using get_variances = DataAccessHelper::get_variances;

  template <class View>
  using outVariant_t = std::variant<ElementArrayView<Ts>...>;

  template <class Getter, class View>
  static outVariant_t<View> get(View &view) {
    const DType type = view.dtype();
    if (type == dtype<double>)
      return {Getter::template get<double>(view)};
    if (type == dtype<float>)
      return {Getter::template get<float>(view)};
    if constexpr (std::is_same_v<Getter, get_values>) {
      if (type == dtype<int64_t>)
        return {Getter::template get<int64_t>(view)};
      if (type == dtype<int32_t>)
        return {Getter::template get<int32_t>(view)};
      if (type == dtype<bool>)
        return {Getter::template get<bool>(view)};
      if (type == dtype<std::string>)
        return {Getter::template get<std::string>(view)};
      if (type == dtype<scipp::core::time_point>)
        return {Getter::template get<scipp::core::time_point>(view)};
      if (type == dtype<Variable>)
        return {Getter::template get<Variable>(view)};
      if (type == dtype<DataArray>)
        return {Getter::template get<DataArray>(view)};
      if (type == dtype<Dataset>)
        return {Getter::template get<Dataset>(view)};
      if (type == dtype<Eigen::Vector3d>)
        return {Getter::template get<Eigen::Vector3d>(view)};
      if (type == dtype<Eigen::Matrix3d>)
        return {Getter::template get<Eigen::Matrix3d>(view)};
      if (type == dtype<Eigen::Affine3d>)
        return {Getter::template get<Eigen::Affine3d>(view)};
      if (type == dtype<scipp::core::Quaternion>)
        return {Getter::template get<scipp::core::Quaternion>(view)};
      if (type == dtype<scipp::core::Translation>)
        return {Getter::template get<scipp::core::Translation>(view)};
      if (type == dtype<scipp::python::PyObject>)
        return {Getter::template get<scipp::python::PyObject>(view)};
      if (type == dtype<bucket<Variable>>)
        return {Getter::template get<bucket<Variable>>(view)};
      if (type == dtype<bucket<DataArray>>)
        return {Getter::template get<bucket<DataArray>>(view)};
      if (type == dtype<bucket<Dataset>>)
        return {Getter::template get<bucket<Dataset>>(view)};
    }
    throw std::runtime_error("Value-access not implemented for this type.");
  }

  template <class View>
  static void set(const Dimensions &dims, const sc_units::Unit unit,
                  const View &view, const nb::object &obj) {
    std::visit(
        [&dims, &unit, &obj](const auto &view_) {
          using T =
              typename std::remove_reference_t<decltype(view_)>::value_type;
          copy_array_into_view(cast_to_array_like<T>(obj, unit), view_, dims);
        },
        view);
  }

  template <typename View, typename T>
  static auto
  get_matrix_elements(const View &view,
                      const std::initializer_list<scipp::index> shape) {
    auto elems = get_data_variable(view).template elements<T>();
    elems = fold(
        elems, Dim::InternalStructureComponent,
        Dimensions({Dim::InternalStructureRow, Dim::InternalStructureColumn},
                   shape));
    std::vector labels(elems.dims().labels().begin(),
                       elems.dims().labels().end());
    std::iter_swap(labels.end() - 2, labels.end() - 1);
    return transpose(elems, labels);
  }

  template <class View> static auto structure_elements(View &&view) {
    if (view.dtype() == dtype<Eigen::Vector3d>) {
      return get_data_variable(view).template elements<Eigen::Vector3d>();
    } else if (view.dtype() == dtype<Eigen::Matrix3d>) {
      return get_matrix_elements<View, Eigen::Matrix3d>(view, {3, 3});
    } else if (view.dtype() == dtype<scipp::core::Quaternion>) {
      return get_data_variable(view)
          .template elements<scipp::core::Quaternion>();
    } else if (view.dtype() == dtype<scipp::core::Translation>) {
      return get_data_variable(view)
          .template elements<scipp::core::Translation>();
    } else if (view.dtype() == dtype<Eigen::Affine3d>) {
      return get_matrix_elements<View, Eigen::Affine3d>(view, {4, 4});
    } else {
      throw std::runtime_error("Unsupported structured dtype");
    }
  }

public:
  template <class Getter, class View>
  static nb::object get_py_array_t(nb::object &obj) {
    auto &view = nb::cast<View &>(obj);
    if (!std::is_const_v<View> && get_data_variable(view).is_readonly())
      return as_ElementArrayViewImpl<const Ts...>::template get_py_array_t<
          Getter, const View>(obj);
    const DType type = view.dtype();
    if (type == dtype<double>)
      return DataAccessHelper::as_py_array_t_impl<Getter, double>(view);
    if (type == dtype<float>)
      return DataAccessHelper::as_py_array_t_impl<Getter, float>(view);
    if (type == dtype<int64_t>)
      return DataAccessHelper::as_py_array_t_impl<Getter, int64_t>(view);
    if (type == dtype<int32_t>)
      return DataAccessHelper::as_py_array_t_impl<Getter, int32_t>(view);
    if (type == dtype<bool>)
      return DataAccessHelper::as_py_array_t_impl<Getter, bool>(view);
    if (type == dtype<scipp::core::time_point>)
      return DataAccessHelper::as_py_array_t_impl<Getter,
                                                  scipp::core::time_point>(
          view);
    if (is_structured(type)) {
      // For readonly structured types, we need to handle specially to avoid
      // the readonly check in elements(). Make a copy of the view which
      // won't be readonly, then get elements from that, and set the result
      // to readonly.
      if (get_data_variable(view).is_readonly()) {
        auto view_copy = variable::copy(get_data_variable(view));
        auto result = DataAccessHelper::as_py_array_t_impl<Getter, double>(
            structure_elements(view_copy));
        // Set the numpy array to readonly since the original was readonly
        result.attr("flags").attr("writeable") = false;
        return result;
      }
      return DataAccessHelper::as_py_array_t_impl<Getter, double>(
          structure_elements(view));
    }
    return std::visit(
        [&obj, &view](const auto &data) {
          const auto &dims = view.dims();
          // We return an individual item in two cases:
          // 1. For 0-D data (consistent with numpy behavior, e.g., when slicing
          // a 1-D array).
          // 2. For 1-D event data, where the individual item is then a
          // vector-like object.
          if (dims.ndim() == 0) {
            return make_scalar(data[0], get_data_variable_concept_handle(view),
                               view);
          } else {
            // Returning view (span or ElementArrayView) by value. This
            // references data in variable, so it must be kept alive.
            // Use rv_policy::reference and keep the parent alive via capsule
            auto ret = nb::cast(data, nb::rv_policy::move);
            // Note: nanobind handles lifetime differently - the owner object
            // should be passed as part of the binding or via nb::keep_alive
            return ret;
          }
        },
        get<Getter>(view));
  }

  template <class Var> static nb::object values(nb::object &object) {
    return get_py_array_t<get_values, Var>(object);
  }

  template <class Var> static nb::object variances(nb::object &object) {
    if (!nb::cast<Var &>(object).has_variances())
      return nb::none();
    return get_py_array_t<get_variances, Var>(object);
  }

  template <class Var>
  static void set_values(Var &view, const nb::object &obj) {
    if (is_structured(view.dtype())) {
      auto elems = structure_elements(view);
      set_values(elems, obj);
    } else {
      set(view.dims(), view.unit(), get<get_values>(view), obj);
    }
  }

  template <class Var>
  static void set_variances(Var &view, const nb::object &obj) {
    if (obj.is_none())
      return remove_variances(view);
    if (!view.has_variances())
      init_variances(view);
    set(view.dims(), view.unit(), get<get_variances>(view), obj);
  }

private:
  static auto numpy_attr(const char *const name) {
    return nb::module_::import_("numpy").attr(name);
  }

  template <class Scalar, class View>
  static nb::object make_scalar(Scalar &&scalar, nb::object parent,
                                const View &view) {
    if constexpr (std::is_same_v<std::decay_t<Scalar>,
                                 scipp::python::PyObject>) {
      // Returning PyObject. This increments the reference counter of
      // the element, so it is ok if the parent `parent` (the variable)
      // goes out of scope.
      return scalar.to_pybind();
    } else if constexpr (std::is_same_v<std::decay_t<Scalar>,
                                        core::time_point>) {
      const auto np_datetime64 = numpy_attr("datetime64");
      return np_datetime64(scalar.time_since_epoch(),
                           to_numpy_time_string(view.unit()));
    } else if constexpr (std::is_arithmetic_v<std::decay_t<Scalar>>) {
      // Create a numpy scalar with the correct dtype
      // We need to preserve the exact dtype (e.g., int32 vs int64)
      nb::module_ numpy = nb::module_::import_("numpy");
      // Get numpy dtype string for this C++ type
      constexpr const char *dtype_str = []() {
        if constexpr (std::is_same_v<std::decay_t<Scalar>, float>)
          return "float32";
        else if constexpr (std::is_same_v<std::decay_t<Scalar>, double>)
          return "float64";
        else if constexpr (std::is_same_v<std::decay_t<Scalar>, int32_t>)
          return "int32";
        else if constexpr (std::is_same_v<std::decay_t<Scalar>, int64_t>)
          return "int64";
        else if constexpr (std::is_same_v<std::decay_t<Scalar>, bool>)
          return "bool";
        else
          return "float64"; // fallback
      }();
      nb::object arr =
          numpy.attr("asarray")(scalar, nb::arg("dtype") = dtype_str);
      return arr.attr("flat").attr("__getitem__")(0);
    } else if constexpr (!std::is_reference_v<Scalar>) {
      // Views such as slices of data arrays for binned data are
      // returned by value and require separate handling to avoid the
      // nb::rv_policy::reference_internal in the default case
      // below.
      return nb::cast(scalar, nb::rv_policy::move);
    } else if constexpr (std::is_same_v<std::decay_t<Scalar>, Variable> ||
                         std::is_same_v<std::decay_t<Scalar>, DataArray> ||
                         std::is_same_v<std::decay_t<Scalar>, Dataset>) {
      // For scipp types, return a reference to preserve ownership semantics.
      // When the .value property is accessed, modifications to the returned
      // object or reassignment of .value should be visible from both sides.
      // The parent object must be kept alive to avoid dangling references.
      auto result = nb::cast(scalar, nb::rv_policy::reference);
      // Keep the parent (self) alive as long as the reference exists
      nb::detail::keep_alive(result.ptr(), parent.ptr());
      return result;
    } else {
      // Returning reference to element in variable.
      // Note: nanobind handles parent lifetime differently
      return nb::cast(scalar, nb::rv_policy::reference);
    }
  }

  // Helper function object to get a scalar value or variance.
  template <class View> struct GetScalarVisitor {
    nb::object &self; // The object we're getting the value / variance from.
    std::remove_reference_t<View> &view; // self as a view.

    template <class Data> auto operator()(const Data &&data) const {
      return make_scalar(data[0], self, view);
    }
  };

  // Helper function object to set a scalar value or variance.
  template <class View> struct SetScalarVisitor {
    const nb::object &rhs;               // The object we are assigning.
    std::remove_reference_t<View> &view; // View of self.

    template <class Data> auto operator()(Data &&data) const {
      using T = typename std::decay_t<decltype(data)>::value_type;
      if constexpr (std::is_same_v<T, scipp::python::PyObject>)
        data[0] = rhs;
      else if constexpr (std::is_same_v<T, scipp::core::time_point>) {
        // TODO support int
        if (view.unit() != parse_datetime_dtype(rhs)) {
          // TODO implement
          throw std::invalid_argument(
              "Conversion of time units is not implemented.");
        }
        data[0] = make_time_point(rhs);
      } else
        data[0] = nb::cast<T>(rhs);
    }
  };

public:
  // Return a scalar value from a variable, implicitly requiring that the
  // variable is 0-dimensional and thus has only a single item.
  template <class Var> static nb::object value(nb::object &obj) {
    auto &view = nb::cast<Var &>(obj);
    if (!std::is_const_v<Var> && get_data_variable(view).is_readonly())
      return as_ElementArrayViewImpl<const Ts...>::template value<const Var>(
          obj);
    expect_scalar(view.dims(), "value");
    if (view.dtype() == dtype<scipp::core::Quaternion> ||
        view.dtype() == dtype<scipp::core::Translation> ||
        view.dtype() == dtype<Eigen::Affine3d> ||
        view.dtype() == dtype<Eigen::Vector3d> ||
        view.dtype() == dtype<Eigen::Matrix3d>)
      return get_py_array_t<get_values, Var>(obj);
    return std::visit(GetScalarVisitor<decltype(view)>{obj, view},
                      get<get_values>(view));
  }
  // Return a scalar variance from a variable, implicitly requiring that the
  // variable is 0-dimensional and thus has only a single item.
  template <class Var> static nb::object variance(nb::object &obj) {
    auto &view = nb::cast<Var &>(obj);
    if (!std::is_const_v<Var> && get_data_variable(view).is_readonly())
      return as_ElementArrayViewImpl<const Ts...>::template variance<const Var>(
          obj);
    expect_scalar(view.dims(), "variance");
    if (!view.has_variances())
      return nb::none();
    return std::visit(GetScalarVisitor<decltype(view)>{obj, view},
                      get<get_variances>(view));
  }
  // Set a scalar value in a variable, implicitly requiring that the
  // variable is 0-dimensional and thus has only a single item.
  template <class Var> static void set_value(Var &view, const nb::object &obj) {
    expect_scalar(view.dims(), "value");
    if (is_structured(view.dtype())) {
      auto elems = structure_elements(view);
      set_values(elems, obj);
    } else {
      std::visit(SetScalarVisitor<decltype(view)>{obj, view},
                 get<get_values>(view));
    }
  }
  // Set a scalar variance in a variable, implicitly requiring that the
  // variable is 0-dimensional and thus has only a single item.
  template <class Var>
  static void set_variance(Var &view, const nb::object &obj) {
    expect_scalar(view.dims(), "variance");
    if (obj.is_none())
      return remove_variances(view);
    if (!view.has_variances())
      init_variances(view);

    std::visit(SetScalarVisitor<decltype(view)>{obj, view},
               get<get_variances>(view));
  }
};

using as_ElementArrayView = as_ElementArrayViewImpl<
    double, float, int64_t, int32_t, bool, std::string, scipp::core::time_point,
    Variable, DataArray, Dataset, bucket<Variable>, bucket<DataArray>,
    bucket<Dataset>, Eigen::Vector3d, Eigen::Matrix3d, scipp::python::PyObject,
    Eigen::Affine3d, scipp::core::Quaternion, scipp::core::Translation>;

template <class T, class... Ignored>
void bind_common_data_properties(nb::class_<T, Ignored...> &c) {
  c.def_prop_ro(
      "dims",
      [](const T &self) {
        const auto &labels = self.dims().labels();
        const auto ndim = static_cast<size_t>(self.ndim());
        nb::tuple dims = nb::steal<nb::tuple>(PyTuple_New(ndim));
        for (size_t i = 0; i < ndim; ++i) {
          PyTuple_SET_ITEM(dims.ptr(), i,
                           PyUnicode_FromString(labels[i].name().c_str()));
        }
        return dims;
      },
      R"(Dimension labels of the data (read-only).

Examples
--------

  >>> import scipp as sc
  >>> var = sc.array(dims=['x', 'y'], values=[[1, 2], [3, 4]])
  >>> var.dims
  ('x', 'y')

  >>> da = sc.DataArray(
  ...     sc.array(dims=['x', 'y'], values=[[1.0, 2.0], [3.0, 4.0]]),
  ...     coords={'x': sc.array(dims=['x'], values=[0.0, 1.0], unit='m')}
  ... )
  >>> da.dims
  ('x', 'y')
)");
  c.def_prop_ro(
      "dim", [](const T &self) { return self.dim().name(); },
      R"(The only dimension label for 1-dimensional data, raising an exception
if the data is not 1-dimensional.

Examples
--------

  >>> import scipp as sc
  >>> var = sc.array(dims=['x'], values=[1, 2, 3], unit='m')
  >>> var.dim
  'x'

  >>> da = sc.DataArray(sc.array(dims=['time'], values=[1.0, 2.0, 3.0], unit='K'))
  >>> da.dim
  'time'
)");
  c.def_prop_ro(
      "ndim", [](const T &self) { return self.ndim(); },
      R"(Number of dimensions of the data (read-only).

Examples
--------

  >>> import scipp as sc
  >>> sc.scalar(1.0).ndim
  0

  >>> sc.array(dims=['x'], values=[1, 2, 3]).ndim
  1

  >>> sc.array(dims=['x', 'y'], values=[[1, 2], [3, 4]]).ndim
  2
)");
  c.def_prop_ro(
      "shape",
      [](const T &self) {
        const auto &sizes = self.dims().sizes();
        const auto ndim = static_cast<size_t>(self.ndim());
        nb::tuple shape = nb::steal<nb::tuple>(PyTuple_New(ndim));
        for (size_t i = 0; i < ndim; ++i) {
          PyTuple_SET_ITEM(shape.ptr(), i, PyLong_FromSsize_t(sizes[i]));
        }
        return shape;
      },
      R"(Shape of the data (read-only).

Examples
--------

  >>> import scipp as sc
  >>> var = sc.array(dims=['x', 'y'], values=[[1, 2, 3], [4, 5, 6]])
  >>> var.shape
  (2, 3)

  >>> sc.scalar(1.0).shape
  ()
)");
  c.def_prop_ro(
      "sizes",
      [](const T &self) {
        const auto &dims = self.dims();
        // Use nb::dict directly instead of std::map in order to guarantee
        // that items are stored in the order of insertion.
        nb::dict sizes;
        for (const auto label : dims.labels()) {
          sizes[label.name().c_str()] = dims[label];
        }
        return sizes;
      },
      R"(dict mapping dimension labels to dimension sizes (read-only).

Examples
--------

  >>> import scipp as sc
  >>> var = sc.array(dims=['x', 'y'], values=[[1, 2, 3], [4, 5, 6]])
  >>> var.sizes
  {'x': 2, 'y': 3}

  >>> da = sc.DataArray(
  ...     sc.array(dims=['time', 'channel'], values=[[1, 2], [3, 4], [5, 6]])
  ... )
  >>> da.sizes
  {'time': 3, 'channel': 2}
)");
}

namespace {
template <class T>
Variable get_data_variable(const T &self, const std::string &property_name) {
  Variable var;
  if constexpr (std::is_same_v<T, DataArray>)
    var = self.data();
  else
    var = self;
  const auto dt = var.dtype();
  if (dt == dtype<bucket<Variable>>) {
    var = var.template bin_buffer<Variable>();
  } else if (dt == dtype<bucket<DataArray>>) {
    var = var.template bin_buffer<DataArray>().data();
  } else if (dt == dtype<bucket<Dataset>>) {
    throw std::runtime_error("Binned data with content of type Dataset "
                             "does not have a well-defined " +
                             property_name + ".");
  }
  return var;
}
} // namespace

template <class T, class... Ignored>
void bind_data_properties(nb::class_<T, Ignored...> &c) {
  bind_common_data_properties(c);
  c.def_prop_ro(
      "dtype",
      [](const T &self) { return get_data_variable(self, "dtype").dtype(); },
      R"(Data type contained in the variable.

Examples
--------

  >>> import scipp as sc
  >>> sc.array(dims=['x'], values=[1, 2, 3]).dtype
  DType('int64')
  >>> sc.array(dims=['x'], values=[1.0, 2.0, 3.0]).dtype
  DType('float64')
  >>> sc.array(dims=['x'], values=['a', 'b', 'c']).dtype
  DType('string')
)");
  c.def_prop_rw(
      "unit",
      [](const T &self) {
        const auto &var = get_data_variable(self, "unit");
        return var.unit() == sc_units::none ? std::optional<sc_units::Unit>()
                                            : var.unit();
      },
      [](const T &self, const ProtoUnit &unit) {
        auto var = get_data_variable(self, "unit");
        var.setUnit(unit_or_default(unit, var.dtype()));
      },
      R"(Physical unit of the data.

Examples
--------

  >>> import scipp as sc
  >>> var = sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='m')
  >>> var.unit
  Unit(m)
  >>> var.unit = 'cm'
  >>> var
  <scipp.Variable> (x: 3)    float64             [cm]  [1, 2, 3]

Note: Changing the unit does not convert the values.
)");
  c.def_prop_rw("values", &as_ElementArrayView::values<T>,
                &as_ElementArrayView::set_values<T>,
                R"(Array of values of the data.

Returns a NumPy array that shares memory with the variable's data buffer.
Modifications to the array will affect the variable and vice versa.

Examples
--------

  >>> import scipp as sc
  >>> var = sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='m')
  >>> var.values
  array([1., 2., 3.])
  >>> type(var.values)
  <class 'numpy.ndarray'>

Values can be modified in place:

  >>> var.values[0] = 10.0
  >>> var
  <scipp.Variable> (x: 3)    float64              [m]  [10, 2, 3]

Or replaced entirely:

  >>> var.values = [4.0, 5.0, 6.0]
  >>> var
  <scipp.Variable> (x: 3)    float64              [m]  [4, 5, 6]
)");
  c.def(
      "_set_variances",
      [](T &view, nb::object obj) {
        as_ElementArrayView::set_variances<T>(view, obj);
      },
      nb::arg("obj").none(),
      "Internal setter for variances that accepts None.");
  c.def_prop_ro("variances", &as_ElementArrayView::variances<T>,
                R"(Array of variances of the data.

Returns a NumPy array that shares memory with the variable's variance buffer,
or None if the variable has no variances.

Examples
--------

  >>> import scipp as sc
  >>> var = sc.array(dims=['x'], values=[1.0, 2.0, 3.0], variances=[0.1, 0.2, 0.3])
  >>> var.variances
  array([0.1, 0.2, 0.3])

Variables without variances return None:

  >>> var_no_var = sc.array(dims=['x'], values=[1.0, 2.0, 3.0])
  >>> var_no_var.variances is None
  True

Variances can be set or removed:

  >>> var_no_var.variances = [0.01, 0.02, 0.03]
  >>> var_no_var.variances
  array([0.01, 0.02, 0.03])
  >>> var_no_var.variances = None
  >>> var_no_var.variances is None
  True
)");
  c.def_prop_rw(
      "value", &as_ElementArrayView::value<T>,
      &as_ElementArrayView::set_value<T>,
      R"(The only value for 0-dimensional data, raising an exception if the data
is not 0-dimensional.

Use this property to access or modify the single value of a scalar (0-D) variable.
For multi-dimensional data, use :py:attr:`values` instead.

Examples
--------

  >>> import scipp as sc
  >>> import numpy as np
  >>> scalar = sc.scalar(3.14, unit='rad')
  >>> scalar.value
  np.float64(3.14)
  >>> scalar.value = 2.0
  >>> scalar
  <scipp.Variable> ()    float64            [rad]  2

Integer scalars return numpy scalar types:

  >>> int_scalar = sc.scalar(42)
  >>> int_scalar.value
  np.int64(42)
)");
  c.def_prop_rw(
      "variance", &as_ElementArrayView::variance<T>,
      &as_ElementArrayView::set_variance<T>,
      R"(The only variance for 0-dimensional data, raising an exception if the
data is not 0-dimensional.

Use this property to access or modify the single variance of a scalar (0-D) variable.
Returns None if the variable has no variances.
For multi-dimensional data, use :py:attr:`variances` instead.

Examples
--------

  >>> import scipp as sc
  >>> import numpy as np
  >>> scalar = sc.scalar(5.0, variance=0.5)
  >>> scalar.variance
  np.float64(0.5)
  >>> scalar.variance = 0.1
  >>> scalar
  <scipp.Variable> ()    float64  [dimensionless]  5  0.1

Scalars without variance return None:

  >>> sc.scalar(5.0).variance is None
  True
)");
  if constexpr (std::is_same_v<T, DataArray> || std::is_same_v<T, Variable>) {
    c.def_prop_ro(
        "size", [](const T &self) { return self.dims().volume(); },
        R"(Number of elements in the data (read-only).

This is the product of all dimension sizes.

Examples
--------

  >>> import scipp as sc
  >>> sc.array(dims=['x'], values=[1, 2, 3]).size
  3
  >>> sc.array(dims=['x', 'y'], values=[[1, 2, 3], [4, 5, 6]]).size
  6
  >>> sc.scalar(1.0).size
  1
)");
  }
}
