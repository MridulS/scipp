// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <algorithm>

#include "scipp/dataset/except.h"
#include "scipp/dataset/map_view.h"

namespace scipp::dataset {

template class SCIPP_DATASET_EXPORT Dict<Dim, Variable>;
template class SCIPP_DATASET_EXPORT Dict<std::string, Variable>;

namespace {
template <class T> void expect_writable(const T &dict) {
  if (dict.is_readonly())
    throw except::DataArrayError(
        "Read-only flag is set, cannot mutate metadata dict.");
}
} // namespace

template <class Key, class Value>
Dict<Key, Value>::Dict(const Sizes &sizes,
                       std::initializer_list<std::pair<const Key, Value>> items,
                       const bool readonly)
    : Dict(sizes, holder_type(items), readonly) {}

template <class Key, class Value>
Dict<Key, Value>::Dict(const Sizes &sizes, holder_type items,
                       const bool readonly)
    : m_sizes(sizes) {
  for (auto &&[key, value] : items)
    set(key, std::move(value));
  // `set` requires Dict to be writable, set readonly flag at the end.
  m_readonly = readonly; // NOLINT(cppcoreguidelines-prefer-member-initializer)
}

template <class Key, class Value>
Dict<Key, Value>::Dict(const Dict &other)
    : Dict(other.m_sizes, other.m_items, false) {}

template <class Key, class Value>
Dict<Key, Value>::Dict(Dict &&other) noexcept
    : Dict(std::move(other.m_sizes), std::move(other.m_items),
           other.m_readonly) {}

template <class Key, class Value>
Dict<Key, Value> &Dict<Key, Value>::operator=(const Dict &other) = default;

template <class Key, class Value>
Dict<Key, Value> &Dict<Key, Value>::operator=(Dict &&other) noexcept = default;

template <class Key, class Value>
bool Dict<Key, Value>::operator==(const Dict &other) const {
  if (size() != other.size())
    return false;
  return std::all_of(this->begin(), this->end(), [&other](const auto &item) {
    const auto &[name, data] = item;
    return other.contains(name) && data == other[name];
  });
}

template <class Key, class Value>
bool Dict<Key, Value>::operator!=(const Dict &other) const {
  return !operator==(other);
}

/// Returns whether a given key is present in the view.
template <class Key, class Value>
bool Dict<Key, Value>::contains(const Key &k) const {
  return m_items.find(k) != m_items.cend();
}

/// Returns 1 or 0, depending on whether key is present in the view or not.
template <class Key, class Value>
scipp::index Dict<Key, Value>::count(const Key &k) const {
  return static_cast<scipp::index>(contains(k));
}

/// Const reference to the coordinate for given dimension.
template <class Key, class Value>
const Value &Dict<Key, Value>::operator[](const Key &key) const {
  return at(key);
}

/// Const reference to the coordinate for given dimension.
template <class Key, class Value>
const Value &Dict<Key, Value>::at(const Key &key) const {
  scipp::expect::contains(*this, key);
  return m_items.at(key);
}

/// The coordinate for given dimension.
template <class Key, class Value>
Value Dict<Key, Value>::operator[](const Key &key) {
  return std::as_const(*this).at(key);
}

/// The coordinate for given dimension.
template <class Key, class Value> Value Dict<Key, Value>::at(const Key &key) {
  return std::as_const(*this).at(key);
}

/// Return the dimension for given coord.
/// @param key Key of the coordinate in a coord dict
///
/// Return the dimension of the coord for 1-D coords or Dim::Invalid for 0-D
/// coords. In the special case of multi-dimension coords the following applies,
/// in this order:
/// - For bin-edge coords return the dimension in which the coord dimension
///   exceeds the data dimensions.
/// - Else, for dimension coords (key matching a dimension), return the key.
/// - Else, return Dim::Invalid.
template <class Key, class Value>
Dim Dict<Key, Value>::dim_of(const Key &key) const {
  const auto &var = at(key);
  if (var.dims().ndim() == 0)
    return Dim::Invalid;
  if (var.dims().ndim() == 1)
    return var.dims().inner();
  if constexpr (std::is_same_v<Key, Dim>) {
    for (const auto &dim : var.dims())
      if (is_edges(sizes(), var.dims(), dim))
        return dim;
    if (var.dims().contains(key))
      return key; // dimension coord
  }
  return Dim::Invalid;
}

template <class Key, class Value>
void Dict<Key, Value>::setSizes(const Sizes &sizes) {
  scipp::expect::includes(sizes, m_sizes);
  m_sizes = sizes;
}

template <class Key, class Value> void Dict<Key, Value>::rebuildSizes() {
  Sizes new_sizes = m_sizes;
  for (const auto &dim : m_sizes) {
    bool erase = true;
    for (const auto &item : *this) {
      if (item.second.dims().contains(dim)) {
        erase = false;
        break;
      }
    }
    if (erase)
      new_sizes.erase(dim);
  }
  m_sizes = std::move(new_sizes);
}

namespace {
template <class Key>
void expect_valid_coord_dims(const Key &key, const Dimensions &coord_dims,
                             const Sizes &da_sizes) {
  using core::to_string;
  if (!da_sizes.includes(coord_dims))
    throw except::DimensionError(
        "Cannot add coord '" + to_string(key) + "' of dims " +
        to_string(coord_dims) + " to DataArray with dims " +
        to_string(Dimensions{da_sizes.labels(), da_sizes.sizes()}));
}
} // namespace

template <class Key, class Value>
void Dict<Key, Value>::set(const key_type &key, mapped_type coord) {
  if (contains(key) && at(key).is_same(coord))
    return;
  expect_writable(*this);
  // Is a good definition for things that are allowed: "would be possible to
  // concat along existing dim or extra dim"?
  auto dims = coord.dims();
  for (const auto &dim : coord.dims()) {
    if (!sizes().contains(dim) && dims[dim] == 2) { // bin edge along extra dim
      dims.erase(dim);
      break;
    } else if (dims[dim] == sizes()[dim] + 1) {
      dims.resize(dim, sizes()[dim]);
      break;
    }
  }
  expect_valid_coord_dims(key, dims, m_sizes);
  m_items.insert_or_assign(key, std::move(coord));
}

template <class Key, class Value>
void Dict<Key, Value>::erase(const key_type &key) {
  expect_writable(*this);
  scipp::expect::contains(*this, key);
  m_items.erase(key);
}

template <class Key, class Value>
Value Dict<Key, Value>::extract(const key_type &key) {
  auto extracted = at(key);
  erase(key);
  return extracted;
}

template <class Key, class Value>
Value Dict<Key, Value>::extract(const key_type &key,
                                const mapped_type &default_value) {
  if (contains(key)) {
    return extract(key);
  }
  return default_value;
}

template <class Key, class Value>
Dict<Key, Value> Dict<Key, Value>::slice(const Slice &params) const {
  const bool readonly = true;
  return {m_sizes.slice(params), slice_map(m_sizes, m_items, params), readonly};
}

namespace {
constexpr auto unaligned_by_dim_slice = [](const auto &coords, const auto &item,
                                           const Slice &params) {
  if (params == Slice{} || params.end() != -1)
    return false;
  const Dim dim = params.dim();
  const auto &[key, var] = item;
  return var.dims().contains(dim) && coords.dim_of(key) == dim;
};
}

template <class Key, class Value>
std::tuple<Dict<Key, Value>, Dict<Key, Value>>
Dict<Key, Value>::slice_coords(const Slice &params) const {
  auto coords = slice(params);
  coords.m_readonly = false;
  Dict<Key, Value> attrs(coords.sizes(), {});
  for (auto &coord : *this)
    if (unaligned_by_dim_slice(*this, coord, params))
      attrs.set(coord.first, coords.extract(coord.first));
  coords.m_readonly = true;
  return {std::move(coords), std::move(attrs)};
}

template <class Key, class Value>
void Dict<Key, Value>::validateSlice(const Slice s, const Dict &dict) const {
  using core::to_string;
  using units::to_string;
  for (const auto &[key, item] : dict) {
    const auto it = find(key);
    if (it == end()) {
      throw except::NotFoundError("Cannot insert new meta data '" +
                                  to_string(key) + "' via a slice.");
    } else if ((it->second.is_readonly() ||
                !it->second.dims().contains(s.dim())) &&
               (it->second.dims().contains(s.dim()) ? it->second.slice(s)
                                                    : it->second) != item) {
      throw except::DimensionError("Cannot update meta data '" +
                                   to_string(key) +
                                   "' via slice since it is implicitly "
                                   "broadcast along the slice dimension '" +
                                   to_string(s.dim()) + "'.");
    }
  }
}

template <class Key, class Value>
Dict<Key, Value> &Dict<Key, Value>::setSlice(const Slice s, const Dict &dict) {
  validateSlice(s, dict);
  for (const auto &[key, item] : dict) {
    const auto it = find(key);
    if (it != end() && !it->second.is_readonly() &&
        it->second.dims().contains(s.dim()))
      it->second.setSlice(s, item);
  }
  return *this;
}

template <class Key, class Value>
void Dict<Key, Value>::rename(const Dim from, const Dim to) {
  m_sizes.replace_key(from, to);
  for (auto &item : m_items)
    if (item.second.dims().contains(from))
      item.second.rename(from, to);
}

/// Mark the dict as readonly. Does not imply that items are readonly.
template <class Key, class Value>
void Dict<Key, Value>::set_readonly() noexcept {
  m_readonly = true;
}

/// Return true if the dict is readonly. Does not imply that items are readonly.
template <class Key, class Value>
bool Dict<Key, Value>::is_readonly() const noexcept {
  return m_readonly;
}

template <class Key, class Value>
Dict<Key, Value> Dict<Key, Value>::as_const() const {
  holder_type items;
  std::transform(m_items.begin(), m_items.end(),
                 std::inserter(items, items.end()), [](const auto &item) {
                   return std::pair(item.first, item.second.as_const());
                 });
  const bool readonly = true;
  return {sizes(), std::move(items), readonly};
}

template <class Key, class Value>
Dict<Key, Value> Dict<Key, Value>::merge_from(const Dict &other) const {
  using core::to_string;
  using units::to_string;
  auto out(*this);
  out.m_readonly = false;
  for (const auto &[key, value] : other) {
    if (out.contains(key))
      throw except::DataArrayError(
          "Coord '" + to_string(key) +
          "' shadows attr of the same name. Remove the attr if you are slicing "
          "an array or use the `coords` and `attrs` properties instead of "
          "`meta`.");
    out.set(key, value);
  }
  out.m_readonly = m_readonly;
  return out;
}

template <class Key, class Value>
bool Dict<Key, Value>::item_applies_to(const Key &key,
                                       const Dimensions &dims) const {
  const auto &val = m_items.at(key);
  return std::all_of(val.dims().begin(), val.dims().end(),
                     [&dims](const Dim dim) { return dims.contains(dim); });
}

} // namespace scipp::dataset
