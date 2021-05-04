// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <unordered_map>

#include "scipp-core_export.h"
#include "scipp/common/index.h"
#include "scipp/common/span.h"
#include "scipp/core/slice.h"
#include "scipp/units/dim.h"

namespace scipp::core {

constexpr int32_t NDIM_MAX = 6;

class Dimensions;

template <class Key, class Value, int16_t MaxSize, class Except = int>
class SCIPP_CORE_EXPORT small_map {
public:
  small_map() = default;

  bool operator==(const small_map &other) const noexcept;
  bool operator!=(const small_map &other) const noexcept;

  auto begin() const { return m_keys.begin(); }
  auto end() const { return m_keys.begin() + size(); }
  typename std::array<Key, MaxSize>::const_iterator find(const Key &key) const;
  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }
  constexpr scipp::index size() const noexcept { return m_size; }
  bool contains(const Key &key) const noexcept;
  scipp::index index(const Key &key) const;
  const Value &operator[](const Key &key) const;
  Value &operator[](const Key &key);
  const Value &at(const Key &key) const;
  Value &at(const Key &key);
  void insert_left(const Key &key, const Value &value);
  void insert_right(const Key &key, const Value &value);
  void erase(const Key &key);
  void clear();
  void replace_key(const Key &from, const Key &to);
  constexpr scipp::span<const Key> keys() const &noexcept {
    return {m_keys.data(), static_cast<size_t>(size())};
  }
  constexpr scipp::span<const Value> values() const &noexcept {
    return {m_values.data(), static_cast<size_t>(size())};
  }

private:
  int16_t m_size{0};
  std::array<Key, MaxSize> m_keys{};
  std::array<Value, MaxSize> m_values{};
};

/// Sibling of class Dimensions, but unordered.
class SCIPP_CORE_EXPORT Sizes : public small_map<Dim, scipp::index, NDIM_MAX> {
private:
  using base = small_map<Dim, scipp::index, NDIM_MAX>;

protected:
  using base::insert_left;
  using base::insert_right;

public:
  Sizes() = default;

  scipp::index count(const Dim dim) const noexcept { return contains(dim); }

  void set(const Dim dim, const scipp::index size);
  void relabel(const Dim from, const Dim to);
  bool includes(const Sizes &sizes) const;
  Sizes slice(const Slice &params) const;

  /// Return the labels of the space defined by *this.
  constexpr auto labels() const &noexcept { return keys(); }
  /// Return the shape of the space defined by *this.
  constexpr auto sizes() const &noexcept { return values(); }
};

[[nodiscard]] SCIPP_CORE_EXPORT Sizes concatenate(const Sizes &a,
                                                  const Sizes &b,
                                                  const Dim dim);

[[nodiscard]] SCIPP_CORE_EXPORT Sizes merge(const Sizes &a, const Sizes &b);

SCIPP_CORE_EXPORT bool is_edges(const Sizes &sizes, const Dimensions &dims,
                                const Dim dim);

SCIPP_CORE_EXPORT std::string to_string(const Sizes &sizes);

} // namespace scipp::core

namespace scipp {
using core::Sizes;
}
