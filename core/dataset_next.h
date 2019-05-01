// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#ifndef DATASET_NEXT_H
#define DATASET_NEXT_H

#include <optional>
#include <string>
#include <string_view>

#include <boost/iterator/transform_iterator.hpp>

#include "dimension.h"
#include "variable.h"

namespace scipp::core {

namespace next {

class Dataset;

namespace ProxyId {
class Coords;
class Labels;
}
template <class Id, class Key> class ConstProxy;
template <class Base> class MutableProxy;

/// Proxy for accessing coordinates of const Dataset and DataConstProxy.
using CoordsConstProxy = ConstProxy<ProxyId::Coords, Dim>;
/// Proxy for accessing coordinates of Dataset and DataProxy.
using CoordsProxy = MutableProxy<CoordsConstProxy>;
/// Proxy for accessing labels of const Dataset and DataConstProxy.
using LabelsConstProxy = ConstProxy<ProxyId::Labels, std::string_view>;
/// Proxy for accessing labels of Dataset and DataProxy.
using LabelsProxy = MutableProxy<LabelsConstProxy>;

/// Helper for passing slicing parameters.
struct Slice {
  Slice(const Dim dim, const scipp::index begin, const scipp::index end = -1)
      : dim(dim), begin(begin), end(end) {}
  Dim dim;
  scipp::index begin;
  scipp::index end;
};

namespace detail {
/// Helper for holding data items in Dataset.
struct DatasetData {
  /// Optional data values.
  std::optional<Variable> values;
  /// Optional data variance.
  std::optional<Variable> variances;
  /// Dimension coord for the sparse dimension (there can be only 1).
  std::optional<Variable> coord;
  /// Potential labels for the sparse dimension.
  std::map<std::string, Variable> labels;
};

template <class Var>
auto makeSlice(Var &var,
               const std::vector<std::pair<Slice, scipp::index>> &slices) {
  std::conditional_t<std::is_const_v<Var>, ConstVariableSlice, VariableSlice>
      slice(var);
  for (const auto[params, extent] : slices) {
    const auto[dim, begin, end] = params;
    if (slice.dimensions().contains(dim)) {
      // TODO rewrite so we can use +1
      if (slice.dimensions()[dim] == extent)
        slice = slice(dim, begin, end);
      else
        slice = slice(dim, begin, end - 1);
    }
  }
  return slice;
}
} // namespace detail

/// Const proxy for a data item and related coordinates of Dataset.
class DataConstProxy {
public:
  DataConstProxy(const Dataset &dataset, const detail::DatasetData &data)
      : m_dataset(&dataset), m_data(&data) {}

  bool isSparse() const noexcept;
  Dim sparseDim() const noexcept;
  Dimensions dims() const noexcept;
  scipp::span<const index> shape() const noexcept;
  units::Unit unit() const;

  CoordsConstProxy coords() const noexcept;
  LabelsConstProxy labels() const noexcept;

  /// Return true if the proxy contains data values.
  bool hasValues() const noexcept { return m_data->values.has_value(); }
  /// Return true if the proxy contains data variances.
  bool hasVariances() const noexcept { return m_data->variances.has_value(); }

  /// Return untyped or typed const proxy for data values.
  template <class T = void> auto values() const {
    if constexpr (std::is_same_v<T, void>)
      return detail::makeSlice(*m_data->values, slices());
    else
      return detail::makeSlice(*m_data->values, slices()).template span<T>();
  }

  /// Return untyped or typed const proxy for data variances.
  template <class T = void> auto variances() const {
    if constexpr (std::is_same_v<T, void>)
      // TODO slice
      return *m_data->variances;
    else
      return m_data->variances->span<T>();
  }

  DataConstProxy slice(const Slice slice) const {
    DataConstProxy sliced(*this);
    sliced.m_slices.emplace_back(slice, dims()[slice.dim]);
    return sliced;
  }

  const auto &slices() const noexcept { return m_slices; }

private:
  const Dataset *m_dataset;
  const detail::DatasetData *m_data;
  std::vector<std::pair<Slice, scipp::index>> m_slices;
};

/// Proxy for a data item and related coordinates of Dataset.
class DataProxy : public DataConstProxy {
public:
  DataProxy(Dataset &dataset, detail::DatasetData &data)
      : DataConstProxy(dataset, data), m_mutableDataset(&dataset),
        m_mutableData(&data) {}

  CoordsProxy coords() const noexcept;
  LabelsProxy labels() const noexcept;

  /// Return untyped or typed proxy for data values.
  template <class T = void> auto values() const {
    if constexpr (std::is_same_v<T, void>)
      return detail::makeSlice(*m_mutableData->values, slices());
    else
      return detail::makeSlice(*m_mutableData->values, slices())
          .template span<T>();
  }

  /// Return untyped or typed proxy for data variances.
  template <class T = void> auto variances() const {
    if constexpr (std::is_same_v<T, void>)
      // TODO slice
      return *m_mutableData->variances;
    else
      return m_mutableData->variances->span<T>();
  }

private:
  Dataset *m_mutableDataset;
  detail::DatasetData *m_mutableData;
};

namespace detail {
/// Helper for creating iterators of Dataset.
template <class D> struct make_item {
  D *dataset;
  using P = std::conditional_t<std::is_const_v<D>, DataConstProxy, DataProxy>;
  std::pair<std::string_view, P> operator()(auto &item) const {
    return {item.first, P(*dataset, item.second)};
  }
};
template <class D> make_item(D *)->make_item<D>;
} // namespace detail

class DatasetConstSlice;
class DatasetSlice;

/// Collection of data arrays.
class Dataset {
public:
  /// Return the number of data items in the dataset.
  ///
  /// This does not include coordinates or attributes, but only all named
  /// entities (which can consist of various combinations of values, variances,
  /// and sparse coordinates).
  index size() const noexcept { return scipp::size(m_data); }
  /// Return true if there are 0 data items in the dataset.
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  CoordsConstProxy coords() const noexcept;
  CoordsProxy coords() noexcept;

  LabelsConstProxy labels() const noexcept;
  LabelsProxy labels() noexcept;

  DataConstProxy operator[](const std::string &name) const;
  DataProxy operator[](const std::string &name);

  auto begin() const && = delete;
  auto begin() && = delete;
  /// Return const iterator to the beginning of all data items.
  auto begin() const &noexcept {
    return boost::make_transform_iterator(m_data.begin(),
                                          detail::make_item{this});
  }
  /// Return iterator to the beginning of all data items.
  auto begin() & noexcept {
    return boost::make_transform_iterator(m_data.begin(),
                                          detail::make_item{this});
  }
  auto end() const && = delete;
  auto end() && = delete;
  /// Return const iterator to the end of all data items.
  auto end() const &noexcept {
    return boost::make_transform_iterator(m_data.end(),
                                          detail::make_item{this});
  }
  /// Return iterator to the end of all data items.
  auto end() & noexcept {
    return boost::make_transform_iterator(m_data.end(),
                                          detail::make_item{this});
  }

  void setCoord(const Dim dim, Variable coord);
  void setLabels(const std::string &labelName, Variable labels);
  void setValues(const std::string &name, Variable values);
  void setVariances(const std::string &name, Variable variances);
  void setSparseCoord(const std::string &name, Variable coord);
  void setSparseLabels(const std::string &name, const std::string &labelName,
                       Variable labels);

  DatasetConstSlice slice(const Dim dim, const scipp::index begin,
                          const scipp::index end = -1) const;
  DatasetConstSlice slice(const Slice slice) const;
  DatasetConstSlice slice(const Slice slice1, const Slice slice2) const;

private:
  friend class DataConstProxy;
  friend class DataProxy;

  std::map<Dim, Variable> m_coords;
  std::map<std::string, Variable> m_labels;
  std::map<std::string, Variable> m_attrs;
  std::map<std::string, detail::DatasetData> m_data;
};

/// Common functionality for other const-proxy classes.
template <class Id, class Key> class ConstProxy {
private:
  struct make_item {
    const ConstProxy *proxy;
    auto operator()(const auto &item) const {
      return std::pair<Key, ConstVariableSlice>(
          item.first, detail::makeSlice(*item.second.first, proxy->slices()));
    }
  };

public:
  using key_type = Key;

  ConstProxy(std::map<Key, std::pair<const Variable *, Variable *>> &&items,
             const std::vector<std::pair<Slice, scipp::index>> &slices = {})
      : m_items(std::move(items)), m_slices(slices) {}

  /// Return the number of coordinates in the proxy.
  index size() const noexcept { return scipp::size(m_items); }
  /// Return true if there are 0 coordinates in the proxy.
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  /// Return a const proxy to the coordinate for given dimension.
  ConstVariableSlice operator[](const Key key) const {
    return detail::makeSlice(*m_items.at(key).first, m_slices);
  }

  auto begin() const && = delete;
  /// Return const iterator to the beginning of all items.
  auto begin() const &noexcept {
    return boost::make_transform_iterator(m_items.begin(), make_item{this});
  }
  auto end() const && = delete;
  /// Return const iterator to the end of all items.
  auto end() const &noexcept {
    return boost::make_transform_iterator(m_items.end(), make_item{this});
  }

  ConstProxy slice(const Slice slice) const {
    std::map<Key, std::pair<const Variable *, Variable *>> items;
    const auto &coord = *m_items.at(slice.dim).first;
    std::copy_if(m_items.begin(), m_items.end(),
                 std::inserter(items, items.end()),
                 [slice, &coord](const auto &item) {
                   const auto &dims = item.second.first->dimensions();
                   // Delete coords that do not depend on slice dim, unless the
                   // sliced coord depends on this coord's dimension.
                   if (!dims.contains(slice.dim) &&
                       !coord.dimensions().contains(item.first))
                     return false;
                   // Delete coord of sliced dimension if slice is not a range.
                   if ((slice.end == -1) && (item.first == slice.dim))
                     return false;
                   // If sliced dimension is bin edges and slice thickness is
                   // not 2 or larger, delete other coords depending on the
                   // sliced dimension.
                   if (dims.contains(slice.dim) &&
                       coord.dimensions()[slice.dim] == dims[slice.dim] + 1)
                     if (slice.end - slice.begin < 2)
                       return false;
                   return true;
                 });
    ConstProxy sliced(std::move(items));
    sliced.m_slices = m_slices;
    sliced.m_slices.emplace_back(slice, coord.dimensions()[slice.dim]);
    return sliced;
  }

  ConstProxy slice(const Slice slice1, const Slice slice2) const {
    return slice(slice1).slice(slice2);
  }

  ConstProxy slice(const Slice slice1, const Slice slice2,
                   const Slice slice3) const {
    return slice(slice1, slice2).slice(slice3);
  }

  const auto &items() const noexcept { return m_items; }
  const auto &slices() const noexcept { return m_slices; }

protected:
  std::map<Key, std::pair<const Variable *, Variable *>> m_items;
  std::vector<std::pair<Slice, scipp::index>> m_slices;
};

/// Common functionality for other proxy classes.
template <class Base> class MutableProxy : public Base {
private:
  struct make_item {
    const MutableProxy<Base> *proxy;
    auto operator()(const auto &item) const {
      return std::pair<typename Base::key_type, VariableSlice>(
          item.first, detail::makeSlice(*item.second.second, proxy->slices()));
    }
  };

  explicit MutableProxy(Base &&base) : Base(std::move(base)) {}

public:
  using Base::Base;

  /// Return a proxy to the coordinate for given dimension.
  VariableSlice operator[](const typename Base::key_type key) const {
    return detail::makeSlice(*Base::items().at(key).second, Base::slices());
  }

  auto begin() const && = delete;
  /// Return iterator to the beginning of all items.
  auto begin() const &noexcept {
    return boost::make_transform_iterator(Base::items().begin(),
                                          make_item{this});
  }
  auto end() const && = delete;
  /// Return iterator to the end of all items.
  auto end() const &noexcept {
    return boost::make_transform_iterator(Base::items().end(), make_item{this});
  }

  MutableProxy slice(const Slice slice) const {
    return MutableProxy(Base::slice(slice));
  }

  MutableProxy slice(const Slice slice1, const Slice slice2) const {
    return slice(slice1).slice(slice2);
  }

  MutableProxy slice(const Slice slice1, const Slice slice2,
                     const Slice slice3) const {
    return slice(slice1, slice2).slice(slice3);
  }
};

/*
class DatasetConstSlice {
public:
  DatasetConstSlice(const Dataset &dataset,
                    const std::initializer_list<Dataset::Slice> &slices)
      : m_dataset(&dataset), m_slices(slices) {}

  index size() const noexcept { return scipp::size(m_data); }
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  CoordsConstProxy coords() const noexcept;

  LabelsConstProxy labels() const noexcept;

  DataConstProxy operator[](const std::string &name) const;

  auto begin() const && = delete;
  auto begin() const &noexcept {
    return boost::make_transform_iterator(m_data.begin(),
                                          detail::make_item{this});
  }
  auto end() const && = delete;
  auto end() const &noexcept {
    return boost::make_transform_iterator(m_data.end(),
                                          detail::make_item{this});
  }

private:
  const Dataset *m_dataset;
  std::vector<Dataset::Slice> m_slices;
};
*/

} // namespace next
} // namespace scipp::core

#endif // DATASET_NEXT_H
