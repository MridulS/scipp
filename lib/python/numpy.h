// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <cstddef>
#include <functional>

#include "scipp/common/index_composition.h"
#include "scipp/core/parallel.h"
#include "scipp/variable/variable.h"

#include "dtype_util.h"
#include "nanobind.h"
#include "numpy_cache.h"
#include "py_object.h"

namespace nb = nanobind;

using namespace scipp;

/// Map C++ types to Python types to perform conversion between scipp containers
/// and numpy arrays.
template <class T> struct ElementTypeMap {
  using PyType = T;
  constexpr static bool convert = false;

  static void check_assignable(const nb::object &, const sc_units::Unit &) {}
};

template <> struct ElementTypeMap<scipp::core::time_point> {
  using PyType = int64_t;
  constexpr static bool convert = true;

  static void check_assignable(const nb::object &obj, sc_units::Unit unit);
};

template <> struct ElementTypeMap<scipp::python::PyObject> {
  using PyType = nb::object;
  constexpr static bool convert = true;

  static void check_assignable(const nb::object &, const sc_units::Unit &) {}
};

/// Cast a nb::object referring to an array to nb::ndarray<auto> if supported.
/// Otherwise, copies the contents into a std::vector<auto>.
template <class T>
auto cast_to_array_like(const nb::object &obj, const sc_units::Unit unit) {
  using TM = ElementTypeMap<T>;
  using PyType = typename TM::PyType;
  TM::check_assignable(obj, unit);
  if constexpr (std::is_same_v<T, core::time_point>) {
    // obj.cast<nb::ndarray<PyType>> does not always work because
    // numpy.datetime64.__int__ delegates to datetime.datetime if the unit is
    // larger than ns and that cannot be converted to long.
    nb::object np_dtype =
        python::numpy_dtype_func()(python::numpy_module().attr("int64"));
    nb::object arr = python::numpy_asarray()(obj).attr("astype")(np_dtype);
    return nb::cast<nb::ndarray<PyType, nb::numpy>>(arr);
  } else if constexpr (std::is_standard_layout_v<T> && std::is_trivial_v<T>) {
    // nanobind's nb::ndarray doesn't auto-convert lists like pybind11's
    // py::array_t, so we need to explicitly convert to numpy array first.
    // We also need to convert to the correct dtype (e.g., int to bool).
    constexpr const char *dtype_str = python::numpy_dtype_str<PyType>();
    // First convert to numpy array, preserving shape (important for scalars)
    nb::object arr = python::numpy_asarray()(obj);
    // If it's a multi-dimensional non-contiguous array, make it contiguous.
    // We check flags['C_CONTIGUOUS'] because nb::cast requires contiguous data.
    // Scalars (ndim=0) are always contiguous.
    if (nb::cast<int>(arr.attr("ndim")) > 0 &&
        !nb::cast<bool>(arr.attr("flags")["C_CONTIGUOUS"])) {
      arr = python::numpy_ascontiguousarray()(arr);
    }
    if constexpr (dtype_str != nullptr) {
      arr = arr.attr("astype")(dtype_str, nb::arg("copy") = false);
    }
    return nb::cast<nb::ndarray<PyType, nb::numpy>>(arr);
  } else {
    // nb::ndarray only supports POD types. Use a simple but expensive
    // solution for other types.
    // TODO Related to #290, we should properly support
    //  multi-dimensional input, and ignore bad shapes.
    try {
      return nb::cast<const std::vector<PyType>>(obj);
    } catch (std::runtime_error &) {
      nb::object array = python::numpy_asarray()(obj);
      std::ostringstream oss;
      oss << "Unable to assign object of dtype "
          << std::string(nb::str(array.attr("dtype")).c_str()) << " to "
          << scipp::core::dtype<T>;
      throw std::invalid_argument(oss.str());
    }
  }
}

namespace scipp::detail {
namespace {
constexpr static size_t grainsize_1d = 10000;

template <class T>
bool is_c_contiguous(const nb::ndarray<T, nb::numpy> &array) {
  // Check if the array is C-contiguous by verifying strides
  if (array.ndim() == 0)
    return true;
  scipp::index expected_stride = 1;
  for (int i = array.ndim() - 1; i >= 0; --i) {
    if (array.stride(i) != expected_stride)
      return false;
    expected_stride *= array.shape(i);
  }
  return true;
}

template <bool convert, class Source, class Destination>
void copy_element(const Source &src, Destination &&dst) {
  if constexpr (convert) {
    dst = std::remove_reference_t<Destination>{src};
  } else {
    std::forward<Destination>(dst) = src;
  }
}

// Helper to access element at given indices using strides
template <class T>
const T &array_at(const nb::ndarray<T, nb::numpy> &arr,
                  const scipp::index *indices) {
  scipp::index offset = 0;
  for (size_t i = 0; i < arr.ndim(); ++i) {
    offset += indices[i] * arr.stride(i);
  }
  return arr.data()[offset];
}

template <bool convert, class T, class Dst>
void copy_array_0d(const nb::ndarray<T, nb::numpy> &src_array, Dst &dst) {
  auto it = dst.begin();
  copy_element<convert>(src_array.data()[0], *it);
}

template <bool convert, class T, class Dst>
void copy_array_1d(const nb::ndarray<T, nb::numpy> &src_array, Dst &dst) {
  const auto *data = src_array.data();
  const auto stride0 = src_array.stride(0);
  const auto shape0 = static_cast<scipp::index>(src_array.shape(0));
  const auto begin = dst.begin();
  core::parallel::parallel_for(
      core::parallel::blocked_range(0, shape0, grainsize_1d),
      [&](const auto &range) {
        auto it = begin + range.begin();
        for (scipp::index i = range.begin(); i < range.end(); ++i, ++it) {
          copy_element<convert>(data[i * stride0], *it);
        }
      });
}

template <bool convert, class T, class Dst>
void copy_array_2d(const nb::ndarray<T, nb::numpy> &src_array, Dst &dst) {
  const auto *data = src_array.data();
  const auto stride0 = src_array.stride(0);
  const auto stride1 = src_array.stride(1);
  const auto shape0 = static_cast<scipp::index>(src_array.shape(0));
  const auto shape1 = static_cast<scipp::index>(src_array.shape(1));
  const auto begin = dst.begin();
  core::parallel::parallel_for(
      core::parallel::blocked_range(0, shape0), [&](const auto &range) {
        auto it = begin + range.begin() * shape1;
        for (scipp::index i = range.begin(); i < range.end(); ++i)
          for (scipp::index j = 0; j < shape1; ++j, ++it)
            copy_element<convert>(data[i * stride0 + j * stride1], *it);
      });
}

template <bool convert, class T, class Dst>
void copy_array_3d(const nb::ndarray<T, nb::numpy> &src_array, Dst &dst) {
  const auto *data = src_array.data();
  const auto stride0 = src_array.stride(0);
  const auto stride1 = src_array.stride(1);
  const auto stride2 = src_array.stride(2);
  const auto shape0 = static_cast<scipp::index>(src_array.shape(0));
  const auto shape1 = static_cast<scipp::index>(src_array.shape(1));
  const auto shape2 = static_cast<scipp::index>(src_array.shape(2));
  const auto begin = dst.begin();
  core::parallel::parallel_for(
      core::parallel::blocked_range(0, shape0), [&](const auto &range) {
        auto it = begin + range.begin() * shape1 * shape2;
        for (scipp::index i = range.begin(); i < range.end(); ++i)
          for (scipp::index j = 0; j < shape1; ++j)
            for (scipp::index k = 0; k < shape2; ++k, ++it)
              copy_element<convert>(
                  data[i * stride0 + j * stride1 + k * stride2], *it);
      });
}

template <bool convert, class T, class Dst>
void copy_array_4d(const nb::ndarray<T, nb::numpy> &src_array, Dst &dst) {
  const auto *data = src_array.data();
  const auto stride0 = src_array.stride(0);
  const auto stride1 = src_array.stride(1);
  const auto stride2 = src_array.stride(2);
  const auto stride3 = src_array.stride(3);
  const auto shape0 = static_cast<scipp::index>(src_array.shape(0));
  const auto shape1 = static_cast<scipp::index>(src_array.shape(1));
  const auto shape2 = static_cast<scipp::index>(src_array.shape(2));
  const auto shape3 = static_cast<scipp::index>(src_array.shape(3));
  const auto begin = dst.begin();
  core::parallel::parallel_for(
      core::parallel::blocked_range(0, shape0), [&](const auto &range) {
        auto it = begin + range.begin() * shape1 * shape2 * shape3;
        for (scipp::index i = range.begin(); i < range.end(); ++i)
          for (scipp::index j = 0; j < shape1; ++j)
            for (scipp::index k = 0; k < shape2; ++k)
              for (scipp::index l = 0; l < shape3; ++l, ++it)
                copy_element<convert>(
                    data[i * stride0 + j * stride1 + k * stride2 + l * stride3],
                    *it);
      });
}

template <bool convert, class T, class Dst>
void copy_array_5d(const nb::ndarray<T, nb::numpy> &src_array, Dst &dst) {
  const auto *data = src_array.data();
  const auto stride0 = src_array.stride(0);
  const auto stride1 = src_array.stride(1);
  const auto stride2 = src_array.stride(2);
  const auto stride3 = src_array.stride(3);
  const auto stride4 = src_array.stride(4);
  const auto shape0 = static_cast<scipp::index>(src_array.shape(0));
  const auto shape1 = static_cast<scipp::index>(src_array.shape(1));
  const auto shape2 = static_cast<scipp::index>(src_array.shape(2));
  const auto shape3 = static_cast<scipp::index>(src_array.shape(3));
  const auto shape4 = static_cast<scipp::index>(src_array.shape(4));
  const auto begin = dst.begin();
  core::parallel::parallel_for(
      core::parallel::blocked_range(0, shape0), [&](const auto &range) {
        auto it = begin + range.begin() * shape1 * shape2 * shape3 * shape4;
        for (scipp::index i = range.begin(); i < range.end(); ++i)
          for (scipp::index j = 0; j < shape1; ++j)
            for (scipp::index k = 0; k < shape2; ++k)
              for (scipp::index l = 0; l < shape3; ++l)
                for (scipp::index m = 0; m < shape4; ++m, ++it)
                  copy_element<convert>(
                      data[i * stride0 + j * stride1 + k * stride2 +
                           l * stride3 + m * stride4],
                      *it);
      });
}

template <bool convert, class T, class Dst>
void copy_array_6d(const nb::ndarray<T, nb::numpy> &src_array, Dst &dst) {
  const auto *data = src_array.data();
  const auto stride0 = src_array.stride(0);
  const auto stride1 = src_array.stride(1);
  const auto stride2 = src_array.stride(2);
  const auto stride3 = src_array.stride(3);
  const auto stride4 = src_array.stride(4);
  const auto stride5 = src_array.stride(5);
  const auto shape0 = static_cast<scipp::index>(src_array.shape(0));
  const auto shape1 = static_cast<scipp::index>(src_array.shape(1));
  const auto shape2 = static_cast<scipp::index>(src_array.shape(2));
  const auto shape3 = static_cast<scipp::index>(src_array.shape(3));
  const auto shape4 = static_cast<scipp::index>(src_array.shape(4));
  const auto shape5 = static_cast<scipp::index>(src_array.shape(5));
  const auto begin = dst.begin();
  core::parallel::parallel_for(
      core::parallel::blocked_range(0, shape0), [&](const auto &range) {
        auto it =
            begin + range.begin() * shape1 * shape2 * shape3 * shape4 * shape5;
        for (scipp::index i = range.begin(); i < range.end(); ++i)
          for (scipp::index j = 0; j < shape1; ++j)
            for (scipp::index k = 0; k < shape2; ++k)
              for (scipp::index l = 0; l < shape3; ++l)
                for (scipp::index m = 0; m < shape4; ++m)
                  for (scipp::index n = 0; n < shape5; ++n, ++it)
                    copy_element<convert>(
                        data[i * stride0 + j * stride1 + k * stride2 +
                             l * stride3 + m * stride4 + n * stride5],
                        *it);
      });
}

template <bool convert, class T, class Dst>
void copy_flattened(const nb::ndarray<T, nb::numpy> &src_array, Dst &dst) {
  const auto *src = src_array.data();
  const auto size = static_cast<scipp::index>(src_array.size());
  const auto begin = dst.begin();
  core::parallel::parallel_for(
      core::parallel::blocked_range(0, size, grainsize_1d),
      [&](const auto &range) {
        auto it = begin + range.begin();
        for (scipp::index i = range.begin(); i < range.end(); ++i, ++it) {
          copy_element<convert>(src[i], *it);
        }
      });
}

template <class T, class View>
bool memory_overlaps(const nb::ndarray<T, nb::numpy> &data, const View &view) {
  // Compute memory bounds of the ndarray
  const auto *data_ptr = reinterpret_cast<const std::byte *>(data.data());
  scipp::index min_offset = 0;
  scipp::index max_offset = 0;
  for (size_t i = 0; i < data.ndim(); ++i) {
    if (data.stride(i) < 0) {
      min_offset += data.stride(i) * (data.shape(i) - 1);
    } else {
      max_offset += data.stride(i) * (data.shape(i) - 1);
    }
  }
  const auto *data_begin = data_ptr + min_offset * sizeof(T);
  const auto *data_end = data_ptr + (max_offset + 1) * sizeof(T);

  const auto begin = view.begin();
  const auto end = view.end();
  const auto view_begin = reinterpret_cast<const std::byte *>(&*begin);
  const auto view_end = reinterpret_cast<const std::byte *>(&*end);
  // Note the use of std::less, pointer comparison with operator< may be
  // undefined behavior with pointers from different arrays.
  return std::less<>()(data_begin, view_end) &&
         std::greater<>()(data_end, view_begin);
}

/*
 * The code here is not pretty.
 * But a generic copy function would be much more complicated than the
 * straightforward nested loops we use here.
 * In practice, there is also little need to support ndim > 6 for non-contiguous
 * data as transform does not support such variables either.
 *
 * For a working, generic implementation, see git ref
 *  bd2e5f0a84d02bd5baf6d0afc32a2ab66dc09e2b
 * and its history, in particular
 *  86761b1e280a63b4f0b723a165188d21dd097972
 *  8721b2d02b98c1acae5c786ffda88055551d832b
 *  4c03a553827f2881672ae1f00f43ae06e879452c
 *  c2a1e3898467083bf7d019a3cb54702c8b50ba86
 *  c2a1e3898467083bf7d019a3cb54702c8b50ba86
 */
/// Copy all elements from src into dst.
/// Performs an explicit conversion of elements in `src` to the element type of
/// `dst` if `convert == true`.
/// Otherwise, elements in src are simply assigned to dst.
template <bool convert, class T, class Dst>
void copy_elements(const nb::ndarray<T, nb::numpy> &src, Dst &dst) {
  if (scipp::size(dst) != static_cast<scipp::index>(src.size()))
    throw std::runtime_error(
        "Numpy data size does not match size of target object.");

  const auto dispatch = [&dst](const nb::ndarray<T, nb::numpy> &src_) {
    if (is_c_contiguous(src_))
      return copy_flattened<convert>(src_, dst);

    switch (src_.ndim()) {
    case 0:
      return copy_array_0d<convert>(src_, dst);
    case 1:
      return copy_array_1d<convert>(src_, dst);
    case 2:
      return copy_array_2d<convert>(src_, dst);
    case 3:
      return copy_array_3d<convert>(src_, dst);
    case 4:
      return copy_array_4d<convert>(src_, dst);
    case 5:
      return copy_array_5d<convert>(src_, dst);
    case 6:
      return copy_array_6d<convert>(src_, dst);
    default:
      throw std::runtime_error(
          "Numpy array with non-c-contiguous memory layout has more "
          "dimensions than supported in the current implementation. "
          "Try making a copy of the array first to get a "
          "c-contiguous layout.");
    }
  };
  if (memory_overlaps(src, dst)) {
    // Make a copy to avoid overlap issues
    nb::module_ numpy = nb::module_::import_("numpy");
    nb::object copy = numpy.attr("array")(src);
    dispatch(nb::cast<nb::ndarray<T, nb::numpy>>(copy));
  } else {
    dispatch(src);
  }
}
} // namespace
} // namespace scipp::detail

template <class SourceDType, class Destination>
void copy_array_into_view(const nb::ndarray<SourceDType, nb::numpy> &src,
                          Destination &&dst, const Dimensions &dims) {
  const auto &shape = dims.shape();
  bool shape_matches = true;
  if (static_cast<size_t>(dims.ndim()) != src.ndim()) {
    shape_matches = false;
  } else {
    for (size_t i = 0; i < src.ndim(); ++i) {
      if (shape[i] != static_cast<scipp::index>(src.shape(i))) {
        shape_matches = false;
        break;
      }
    }
  }
  if (!shape_matches)
    throw except::DimensionError("The shape of the provided data "
                                 "does not match the existing "
                                 "object.");
  scipp::detail::copy_elements<ElementTypeMap<
      typename std::remove_reference_t<Destination>::value_type>::convert>(src,
                                                                           dst);
}

template <class SourceDType, class Destination>
void copy_array_into_view(const std::vector<SourceDType> &src, Destination &dst,
                          const Dimensions &) {
  core::expect::sizeMatches(dst, src);
  std::copy(begin(src), end(src), dst.begin());
}

core::time_point make_time_point(const nb::object &buffer, int64_t scale = 1);
