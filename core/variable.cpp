// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <cmath>
#include <string>

#include "apply.h"
#include "counts.h"
#include "dataset.h"
#include "except.h"
#include "transform.h"
#include "variable.h"
#include "variable_view.h"

namespace scipp::core {

template <class T, class C> auto &requireT(C &concept) {
  try {
    return dynamic_cast<T &>(concept);
  } catch (const std::bad_cast &) {
    throw except::TypeError("Expected item dtype " +
                            to_string(T::static_dtype()) + ", got " +
                            to_string(concept.dtype()) + '.');
  }
}

template <class T1, class T2> bool equal(const T1 &view1, const T2 &view2) {
  return std::equal(view1.begin(), view1.end(), view2.begin(), view2.end());
}

template <class T> class DataModel;
template <class T> class VariableConceptT;
template <class T> struct RebinHelper {
  // Special rebin version for rebinning inner dimension to a joint new coord.
  static void rebinInner(const Dim dim, const VariableConceptT<T> &oldT,
                         VariableConceptT<T> &newT,
                         const VariableConceptT<T> &oldCoordT,
                         const VariableConceptT<T> &newCoordT) {
    const auto &oldData = oldT.values();
    auto newData = newT.values();
    const auto oldSize = oldT.dims()[dim];
    const auto newSize = newT.dims()[dim];
    const auto count = oldT.dims().volume() / oldSize;
    const auto *xold = &*oldCoordT.values().begin();
    const auto *xnew = &*newCoordT.values().begin();
    // This function assumes that dimensions between coord and data either
    // match, or coord is 1D.
    const bool jointOld = oldCoordT.dims().ndim() == 1;
    const bool jointNew = newCoordT.dims().ndim() == 1;
#pragma omp parallel for
    for (scipp::index c = 0; c < count; ++c) {
      scipp::index iold = 0;
      scipp::index inew = 0;
      const scipp::index oldEdgeOffset = jointOld ? 0 : c * (oldSize + 1);
      const scipp::index newEdgeOffset = jointNew ? 0 : c * (newSize + 1);
      const auto oldOffset = c * oldSize;
      const auto newOffset = c * newSize;
      while ((iold < oldSize) && (inew < newSize)) {
        auto xo_low = xold[oldEdgeOffset + iold];
        auto xo_high = xold[oldEdgeOffset + iold + 1];
        auto xn_low = xnew[newEdgeOffset + inew];
        auto xn_high = xnew[newEdgeOffset + inew + 1];

        if (xn_high <= xo_low)
          inew++; /* old and new bins do not overlap */
        else if (xo_high <= xn_low)
          iold++; /* old and new bins do not overlap */
        else {
          // delta is the overlap of the bins on the x axis
          auto delta = xo_high < xn_high ? xo_high : xn_high;
          delta -= xo_low > xn_low ? xo_low : xn_low;

          auto owidth = xo_high - xo_low;
          newData[newOffset + inew] +=
              oldData[oldOffset + iold] * delta / owidth;

          if (xn_high > xo_high) {
            iold++;
          } else {
            inew++;
          }
        }
      }
    }
  }
};

template <typename T> struct RebinGeneralHelper {
  static void rebin(const Dim dim, const Variable &oldT, Variable &newT,
                    const Variable &oldCoordT, const Variable &newCoordT) {
    const auto oldSize = oldT.dims()[dim];
    const auto newSize = newT.dims()[dim];

    const auto *xold = oldCoordT.values<T>().data();
    const auto *xnew = newCoordT.values<T>().data();
    // This function assumes that dimensions between coord and data
    // coord is 1D.
    int iold = 0;
    int inew = 0;
    while ((iold < oldSize) && (inew < newSize)) {
      auto xo_low = xold[iold];
      auto xo_high = xold[iold + 1];
      auto xn_low = xnew[inew];
      auto xn_high = xnew[inew + 1];

      if (xn_high <= xo_low)
        inew++; /* old and new bins do not overlap */
      else if (xo_high <= xn_low)
        iold++; /* old and new bins do not overlap */
      else {
        // delta is the overlap of the bins on the x axis
        auto delta = xo_high < xn_high ? xo_high : xn_high;
        delta -= xo_low > xn_low ? xo_low : xn_low;

        auto owidth = xo_high - xo_low;
        newT(dim, inew) += oldT(dim, iold) * delta / owidth;
        if (xn_high > xo_high) {
          iold++;
        } else {
          inew++;
        }
      }
    }
  }
};

template <class T> class ViewModel;

VariableConcept::VariableConcept(const Dimensions &dimensions)
    : m_dimensions(dimensions){};

bool isMatchingOr1DBinEdge(const Dim dim, Dimensions edges,
                           const Dimensions &toMatch) {
  if (edges.ndim() == 1)
    return true;
  edges.resize(dim, edges[dim] - 1);
  return edges == toMatch;
}

template <class T>
auto makeSpan(T &model, const Dimensions &dims, const Dim dim,
              const scipp::index begin, const scipp::index end) {
  if (!dims.contains(dim) && (begin != 0 || end != 1))
    throw std::runtime_error("VariableConcept: Slice index out of range.");
  if (!dims.contains(dim) || dims[dim] == end - begin) {
    return scipp::span(model.data(), model.data() + model.size());
  }
  const scipp::index beginOffset = begin * dims.offset(dim);
  const scipp::index endOffset = end * dims.offset(dim);
  return scipp::span(model.data() + beginOffset, model.data() + endOffset);
}

template <class T, class... Args>
auto optionalVariancesView(T &concept, Args &&... args) {
  return concept.hasVariances()
             ? std::optional(concept.variancesView(std::forward<Args>(args)...))
             : std::nullopt;
}

template <class T> VariableConceptHandle VariableConceptT<T>::makeView() const {
  auto &dims = this->dims();
  return std::make_unique<ViewModel<decltype(valuesView(dims))>>(
      dims, valuesView(dims), optionalVariancesView(*this, dims));
}

template <class T> VariableConceptHandle VariableConceptT<T>::makeView() {
  if (this->isConstView())
    return const_cast<const VariableConceptT &>(*this).makeView();
  auto &dims = this->dims();
  return std::make_unique<ViewModel<decltype(valuesView(dims))>>(
      dims, valuesView(dims), optionalVariancesView(*this, dims));
}

template <class T>
VariableConceptHandle
VariableConceptT<T>::makeView(const Dim dim, const scipp::index begin,
                              const scipp::index end) const {
  auto dims = this->dims();
  if (end == -1)
    dims.erase(dim);
  else
    dims.resize(dim, end - begin);
  return std::make_unique<ViewModel<decltype(valuesView(dims, dim, begin))>>(
      dims, valuesView(dims, dim, begin),
      optionalVariancesView(*this, dims, dim, begin));
}

template <class T>
VariableConceptHandle VariableConceptT<T>::makeView(const Dim dim,
                                                    const scipp::index begin,
                                                    const scipp::index end) {
  if (this->isConstView())
    return const_cast<const VariableConceptT &>(*this).makeView(dim, begin,
                                                                end);
  auto dims = this->dims();
  if (end == -1)
    dims.erase(dim);
  else
    dims.resize(dim, end - begin);
  return std::make_unique<ViewModel<decltype(valuesView(dims, dim, begin))>>(
      dims, valuesView(dims, dim, begin),
      optionalVariancesView(*this, dims, dim, begin));
}

template <class T>
VariableConceptHandle
VariableConceptT<T>::reshape(const Dimensions &dims) const {
  if (this->dims().volume() != dims.volume())
    throw std::runtime_error(
        "Cannot reshape to dimensions with different volume");
  return std::make_unique<ViewModel<decltype(getReshaped(dims))>>(
      dims, getReshaped(dims));
}

template <class T>
VariableConceptHandle VariableConceptT<T>::reshape(const Dimensions &dims) {
  if (this->dims().volume() != dims.volume())
    throw std::runtime_error(
        "Cannot reshape to dimensions with different volume");
  return std::make_unique<ViewModel<decltype(getReshaped(dims))>>(
      dims, getReshaped(dims));
}

template <class T>
bool VariableConceptT<T>::operator==(const VariableConcept &other) const {
  const auto &dims = this->dims();
  if (dims != other.dims())
    return false;
  if (this->dtype() != other.dtype())
    return false;
  if (this->hasVariances() != other.hasVariances())
    return false;
  const auto &otherT = requireT<const VariableConceptT>(other);
  if (dims.volume() == 0 && dims == other.dims())
    return true;
  if (this->isContiguous()) {
    if (other.isContiguous() && dims.isContiguousIn(other.dims())) {
      return equal(values(), otherT.values()) &&
             (!this->hasVariances() || equal(variances(), otherT.variances()));
    } else {
      return equal(values(), otherT.valuesView(dims)) &&
             (!this->hasVariances() ||
              equal(variances(), otherT.variancesView(dims)));
    }
  } else {
    if (other.isContiguous() && dims.isContiguousIn(other.dims())) {
      return equal(valuesView(dims), otherT.values()) &&
             (!this->hasVariances() ||
              equal(variancesView(dims), otherT.variances()));
    } else {
      return equal(valuesView(dims), otherT.valuesView(dims)) &&
             (!this->hasVariances() ||
              equal(variancesView(dims), otherT.variancesView(dims)));
    }
  }
}

template <class T>
void VariableConceptT<T>::copy(const VariableConcept &other, const Dim dim,
                               const scipp::index offset,
                               const scipp::index otherBegin,
                               const scipp::index otherEnd) {
  auto iterDims = this->dims();
  const scipp::index delta = otherEnd - otherBegin;
  if (iterDims.contains(dim))
    iterDims.resize(dim, delta);

  const auto &otherT = requireT<const VariableConceptT>(other);
  auto otherView = otherT.valuesView(iterDims, dim, otherBegin);
  // Four cases for minimizing use of VariableView --- just copy contiguous
  // range where possible.
  if (this->isContiguous() && iterDims.isContiguousIn(this->dims())) {
    auto target = values(dim, offset, offset + delta);
    if (other.isContiguous() && iterDims.isContiguousIn(other.dims())) {
      auto source = otherT.values(dim, otherBegin, otherEnd);
      std::copy(source.begin(), source.end(), target.begin());
    } else {
      std::copy(otherView.begin(), otherView.end(), target.begin());
    }
  } else {
    auto view = valuesView(iterDims, dim, offset);
    if (other.isContiguous() && iterDims.isContiguousIn(other.dims())) {
      auto source = otherT.values(dim, otherBegin, otherEnd);
      std::copy(source.begin(), source.end(), view.begin());
    } else {
      std::copy(otherView.begin(), otherView.end(), view.begin());
    }
  }
  // TODO Avoid code duplication for variances.
  if (this->hasVariances()) {
    auto otherView = otherT.variancesView(iterDims, dim, otherBegin);
    if (this->isContiguous() && iterDims.isContiguousIn(this->dims())) {
      auto target = variances(dim, offset, offset + delta);
      if (other.isContiguous() && iterDims.isContiguousIn(other.dims())) {
        auto source = otherT.variances(dim, otherBegin, otherEnd);
        std::copy(source.begin(), source.end(), target.begin());
      } else {
        std::copy(otherView.begin(), otherView.end(), target.begin());
      }
    } else {
      auto view = variancesView(iterDims, dim, offset);
      if (other.isContiguous() && iterDims.isContiguousIn(other.dims())) {
        auto source = otherT.variances(dim, otherBegin, otherEnd);
        std::copy(source.begin(), source.end(), view.begin());
      } else {
        std::copy(otherView.begin(), otherView.end(), view.begin());
      }
    }
  }
}

/// Implementation of VariableConcept that holds data.
template <class T> class DataModel : public conceptT_t<typename T::value_type> {
public:
  using value_type = std::remove_const_t<typename T::value_type>;

  DataModel(const Dimensions &dimensions, T model,
            std::optional<T> variances = std::nullopt)
      : conceptT_t<typename T::value_type>(std::move(dimensions)),
        m_model(std::move(model)), m_variances(std::move(variances)) {
    if (this->dims().volume() != scipp::size(m_model))
      throw std::runtime_error("Creating Variable: data size does not match "
                               "volume given by dimension extents");
  }

  scipp::span<value_type> values() override {
    return scipp::span(m_model.data(), m_model.data() + size());
  }
  scipp::span<value_type> values(const Dim dim, const scipp::index begin,
                                 const scipp::index end) override {
    return makeSpan(m_model, this->dims(), dim, begin, end);
  }

  scipp::span<const value_type> values() const override {
    return scipp::span(m_model.data(), m_model.data() + size());
  }
  scipp::span<const value_type> values(const Dim dim, const scipp::index begin,
                                       const scipp::index end) const override {
    return makeSpan(m_model, this->dims(), dim, begin, end);
  }

  scipp::span<value_type> variances() override {
    return scipp::span(m_variances->data(), m_variances->data() + size());
  }
  scipp::span<value_type> variances(const Dim dim, const scipp::index begin,
                                    const scipp::index end) override {
    return makeSpan(*m_variances, this->dims(), dim, begin, end);
  }

  scipp::span<const value_type> variances() const override {
    return scipp::span(m_variances->data(), m_variances->data() + size());
  }
  scipp::span<const value_type>
  variances(const Dim dim, const scipp::index begin,
            const scipp::index end) const override {
    return makeSpan(*m_variances, this->dims(), dim, begin, end);
  }

  VariableView<value_type> valuesView(const Dimensions &dims) override {
    return makeVariableView(m_model.data(), 0, dims, this->dims());
  }
  VariableView<value_type> valuesView(const Dimensions &dims, const Dim dim,
                                      const scipp::index begin) override {
    scipp::index beginOffset = this->dims().contains(dim)
                                   ? begin * this->dims().offset(dim)
                                   : begin * this->dims().volume();
    return makeVariableView(m_model.data(), beginOffset, dims, this->dims());
  }

  VariableView<const value_type>
  valuesView(const Dimensions &dims) const override {
    return makeVariableView(m_model.data(), 0, dims, this->dims());
  }
  VariableView<const value_type>
  valuesView(const Dimensions &dims, const Dim dim,
             const scipp::index begin) const override {
    scipp::index beginOffset = this->dims().contains(dim)
                                   ? begin * this->dims().offset(dim)
                                   : begin * this->dims().volume();
    return makeVariableView(m_model.data(), beginOffset, dims, this->dims());
  }

  VariableView<value_type> variancesView(const Dimensions &dims) override {
    return makeVariableView(m_variances->data(), 0, dims, this->dims());
  }
  VariableView<value_type> variancesView(const Dimensions &dims, const Dim dim,
                                         const scipp::index begin) override {
    scipp::index beginOffset = this->dims().contains(dim)
                                   ? begin * this->dims().offset(dim)
                                   : begin * this->dims().volume();
    return makeVariableView(m_variances->data(), beginOffset, dims,
                            this->dims());
  }

  VariableView<const value_type>
  variancesView(const Dimensions &dims) const override {
    return makeVariableView(m_variances->data(), 0, dims, this->dims());
  }
  VariableView<const value_type>
  variancesView(const Dimensions &dims, const Dim dim,
                const scipp::index begin) const override {
    scipp::index beginOffset = this->dims().contains(dim)
                                   ? begin * this->dims().offset(dim)
                                   : begin * this->dims().volume();
    return makeVariableView(m_variances->data(), beginOffset, dims,
                            this->dims());
  }

  VariableView<const value_type>
  getReshaped(const Dimensions &dims) const override {
    return makeVariableView(m_model.data(), 0, dims, dims);
  }
  VariableView<value_type> getReshaped(const Dimensions &dims) override {
    return makeVariableView(m_model.data(), 0, dims, dims);
  }

  VariableConceptHandle clone() const override {
    return std::make_unique<DataModel<T>>(this->dims(), m_model, m_variances);
  }

  VariableConceptHandle clone(const Dimensions &dims) const override {
    if (hasVariances())
      return std::make_unique<DataModel<T>>(dims, T(dims.volume()),
                                            T(dims.volume()));
    else
      return std::make_unique<DataModel<T>>(dims, T(dims.volume()));
  }

  bool isContiguous() const override { return true; }
  bool isView() const override { return false; }
  bool isConstView() const override { return false; }
  bool hasVariances() const noexcept override {
    return m_variances.has_value();
  }

  scipp::index size() const override { return m_model.size(); }

  T m_model;
  std::optional<T> m_variances;
};

namespace detail {
template <class T>
std::unique_ptr<VariableConceptT<T>>
makeVariableConceptT(const Dimensions &dims) {
  return std::make_unique<DataModel<Vector<T>>>(dims, Vector<T>(dims.volume()));
}
template <class T>
std::unique_ptr<VariableConceptT<T>>
makeVariableConceptT(const Dimensions &dims, Vector<T> data) {
  return std::make_unique<DataModel<Vector<T>>>(dims, std::move(data));
}
template std::unique_ptr<VariableConceptT<double>>
makeVariableConceptT<double>(const Dimensions &);
template std::unique_ptr<VariableConceptT<float>>
makeVariableConceptT<float>(const Dimensions &);
template std::unique_ptr<VariableConceptT<double>>
makeVariableConceptT<double>(const Dimensions &, Vector<double>);
template std::unique_ptr<VariableConceptT<float>>
makeVariableConceptT<float>(const Dimensions &, Vector<float>);
} // namespace detail

/// Implementation of VariableConcept that represents a view onto data.
template <class T>
class ViewModel
    : public conceptT_t<std::remove_const_t<typename T::element_type>> {
  void requireMutable() const {
    if (isConstView())
      throw std::runtime_error(
          "View is const, cannot get mutable range of data.");
  }
  void requireContiguous() const {
    if (!isContiguous())
      throw std::runtime_error(
          "View is not contiguous, cannot get contiguous range of data.");
  }

public:
  using value_type = typename T::value_type;

  ViewModel(const Dimensions &dimensions, T model,
            std::optional<T> variances = std::nullopt)
      : conceptT_t<value_type>(std::move(dimensions)),
        m_model(std::move(model)), m_variances(std::move(variances)) {
    if (this->dims().volume() != m_model.size())
      throw std::runtime_error("Creating Variable: data size does not match "
                               "volume given by dimension extents");
  }

  scipp::span<value_type> values() override {
    requireMutable();
    requireContiguous();
    if constexpr (std::is_const<typename T::element_type>::value)
      return scipp::span<value_type>();
    else
      return scipp::span(m_model.data(), m_model.data() + size());
  }
  scipp::span<value_type> values(const Dim dim, const scipp::index begin,
                                 const scipp::index end) override {
    requireMutable();
    requireContiguous();
    if constexpr (std::is_const<typename T::element_type>::value) {
      static_cast<void>(dim);
      static_cast<void>(begin);
      static_cast<void>(end);
      return scipp::span<value_type>();
    } else {
      return makeSpan(m_model, this->dims(), dim, begin, end);
    }
  }

  scipp::span<const value_type> values() const override {
    requireContiguous();
    return scipp::span(m_model.data(), m_model.data() + size());
  }
  scipp::span<const value_type> values(const Dim dim, const scipp::index begin,
                                       const scipp::index end) const override {
    requireContiguous();
    return makeSpan(m_model, this->dims(), dim, begin, end);
  }

  scipp::span<value_type> variances() override {
    requireMutable();
    requireContiguous();
    if constexpr (std::is_const<typename T::element_type>::value)
      return scipp::span<value_type>();
    else
      return scipp::span(m_variances->data(), m_variances->data() + size());
  }
  scipp::span<value_type> variances(const Dim dim, const scipp::index begin,
                                    const scipp::index end) override {
    requireMutable();
    requireContiguous();
    if constexpr (std::is_const<typename T::element_type>::value) {
      static_cast<void>(dim);
      static_cast<void>(begin);
      static_cast<void>(end);
      return scipp::span<value_type>();
    } else {
      return makeSpan(*m_variances, this->dims(), dim, begin, end);
    }
  }

  scipp::span<const value_type> variances() const override {
    requireContiguous();
    return scipp::span(m_variances->data(), m_variances->data() + size());
  }
  scipp::span<const value_type>
  variances(const Dim dim, const scipp::index begin,
            const scipp::index end) const override {
    requireContiguous();
    return makeSpan(*m_variances, this->dims(), dim, begin, end);
  }

  VariableView<value_type> valuesView(const Dimensions &dims) override {
    requireMutable();
    if constexpr (std::is_const<typename T::element_type>::value) {
      static_cast<void>(dims);
      return VariableView<value_type>(nullptr, 0, {}, {});
    } else {
      return {m_model, dims};
    }
  }
  VariableView<value_type> valuesView(const Dimensions &dims, const Dim dim,
                                      const scipp::index begin) override {
    requireMutable();
    if constexpr (std::is_const<typename T::element_type>::value) {
      static_cast<void>(dim);
      static_cast<void>(begin);
      return VariableView<value_type>(nullptr, 0, {}, {});
    } else {
      return {m_model, dims, dim, begin};
    }
  }

  VariableView<const value_type>
  valuesView(const Dimensions &dims) const override {
    return {m_model, dims};
  }
  VariableView<const value_type>
  valuesView(const Dimensions &dims, const Dim dim,
             const scipp::index begin) const override {
    return {m_model, dims, dim, begin};
  }

  VariableView<value_type> variancesView(const Dimensions &dims) override {
    requireMutable();
    if constexpr (std::is_const<typename T::element_type>::value) {
      static_cast<void>(dims);
      return VariableView<value_type>(nullptr, 0, {}, {});
    } else {
      return {*m_variances, dims};
    }
  }
  VariableView<value_type> variancesView(const Dimensions &dims, const Dim dim,
                                         const scipp::index begin) override {
    requireMutable();
    if constexpr (std::is_const<typename T::element_type>::value) {
      static_cast<void>(dim);
      static_cast<void>(begin);
      return VariableView<value_type>(nullptr, 0, {}, {});
    } else {
      return {*m_variances, dims, dim, begin};
    }
  }

  VariableView<const value_type>
  variancesView(const Dimensions &dims) const override {
    return {*m_variances, dims};
  }
  VariableView<const value_type>
  variancesView(const Dimensions &dims, const Dim dim,
                const scipp::index begin) const override {
    return {*m_variances, dims, dim, begin};
  }

  VariableView<const value_type>
  getReshaped(const Dimensions &dims) const override {
    return {m_model, dims};
  }
  VariableView<value_type> getReshaped(const Dimensions &dims) override {
    requireMutable();
    if constexpr (std::is_const<typename T::element_type>::value) {
      static_cast<void>(dims);
      return VariableView<value_type>(nullptr, 0, {}, {});
    } else {
      return {m_model, dims};
    }
  }

  VariableConceptHandle clone() const override {
    return std::make_unique<ViewModel<T>>(this->dims(), m_model, m_variances);
  }

  VariableConceptHandle clone(const Dimensions &) const override {
    throw std::runtime_error("Cannot resize view.");
  }

  bool isContiguous() const override {
    return this->dims().isContiguousIn(m_model.parentDimensions());
  }
  bool isView() const override { return true; }
  bool isConstView() const override {
    return std::is_const<typename T::element_type>::value;
  }
  bool hasVariances() const noexcept override {
    return m_variances.has_value();
  }

  scipp::index size() const override { return m_model.size(); }

  T m_model;
  std::optional<T> m_variances;
};

Variable::Variable(const ConstVariableSlice &slice)
    : Variable(*slice.m_variable) {
  if (slice.m_view) {
    setUnit(slice.unit());
    setDimensions(slice.dims());
    // There is a bug in the implementation of MultiIndex used in VariableView
    // in case one of the dimensions has extent 0.
    if (dims().volume() != 0)
      data().copy(slice.data(), Dim::Invalid, 0, 0, 1);
  }
}
Variable::Variable(const Variable &parent, const Dimensions &dims)
    : m_unit(parent.unit()), m_object(parent.m_object->clone(dims)) {}

Variable::Variable(const ConstVariableSlice &parent, const Dimensions &dims)
    : m_unit(parent.unit()), m_object(parent.data().clone(dims)) {}

Variable::Variable(const Variable &parent, VariableConceptHandle data)
    : m_unit(parent.unit()), m_object(std::move(data)) {}

template <class T>
Variable::Variable(const units::Unit unit, const Dimensions &dimensions,
                   T object, const Dim sparseDim)
    : m_sparseDim(sparseDim), m_unit{unit},
      m_object(std::make_unique<DataModel<T>>(std::move(dimensions),
                                              std::move(object))) {}
template <class T>
Variable::Variable(const units::Unit unit, const Dimensions &dimensions,
                   T values, T variances, const Dim sparseDim)
    : m_sparseDim(sparseDim), m_unit{unit},
      m_object(std::make_unique<DataModel<T>>(
          std::move(dimensions), std::move(values), std::move(variances))) {}

void Variable::setDimensions(const Dimensions &dimensions) {
  if (dimensions.volume() == m_object->dims().volume()) {
    if (dimensions != m_object->dims())
      data().m_dimensions = dimensions;
    return;
  }
  m_object = m_object->clone(dimensions);
}

template <class T>
const Vector<underlying_type_t<T>> &Variable::cast(const bool variances) const {
  auto &dm = requireT<const DataModel<Vector<underlying_type_t<T>>>>(*m_object);
  if (!variances)
    return dm.m_model;
  else {
    if (!hasVariances())
      throw std::runtime_error("No variances");
    return *dm.m_variances;
  }
}

template <class T>
Vector<underlying_type_t<T>> &Variable::cast(const bool variances) {
  auto &dm = requireT<DataModel<Vector<underlying_type_t<T>>>>(*m_object);
  if (!variances)
    return dm.m_model;
  else {
    if (!hasVariances())
      throw std::runtime_error("No variances");
    return *dm.m_variances;
  }
}

#define INSTANTIATE(...)                                                       \
  template Variable::Variable(const units::Unit, const Dimensions &,           \
                              Vector<underlying_type_t<__VA_ARGS__>>,          \
                              const Dim);                                      \
  template Variable::Variable(const units::Unit, const Dimensions &,           \
                              Vector<underlying_type_t<__VA_ARGS__>>,          \
                              Vector<underlying_type_t<__VA_ARGS__>>,          \
                              const Dim);                                      \
  template Vector<underlying_type_t<__VA_ARGS__>>                              \
      &Variable::cast<__VA_ARGS__>(const bool);                                \
  template const Vector<underlying_type_t<__VA_ARGS__>>                        \
      &Variable::cast<__VA_ARGS__>(const bool) const;

INSTANTIATE(std::string)
INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(int64_t)
INSTANTIATE(int32_t)
INSTANTIATE(char)
INSTANTIATE(bool)
INSTANTIATE(std::pair<int64_t, int64_t>)
#if defined(_WIN32) || defined(__clang__) && defined(__APPLE__)
INSTANTIATE(scipp::index)
INSTANTIATE(std::pair<scipp::index, scipp::index>)
#endif
INSTANTIATE(boost::container::small_vector<scipp::index, 1>)
INSTANTIATE(boost::container::small_vector<double, 8>)
INSTANTIATE(std::vector<double>)
INSTANTIATE(std::vector<std::string>)
INSTANTIATE(std::vector<scipp::index>)
INSTANTIATE(Dataset)
INSTANTIATE(std::array<double, 3>)
INSTANTIATE(std::array<double, 4>)
INSTANTIATE(Eigen::Vector3d)

template <class T1, class T2> bool equals(const T1 &a, const T2 &b) {
  if (a.unit() != b.unit())
    return false;
  if (!(a.dims() == b.dims()))
    return false;
  return a.data() == b.data();
}

bool Variable::operator==(const Variable &other) const {
  return equals(*this, other);
}

bool Variable::operator==(const ConstVariableSlice &other) const {
  return equals(*this, other);
}

bool Variable::operator!=(const Variable &other) const {
  return !(*this == other);
}

bool Variable::operator!=(const ConstVariableSlice &other) const {
  return !(*this == other);
}

template <class T1, class T2> T1 &plus_equals(T1 &variable, const T2 &other) {
  // Addition with different Variable type is supported, mismatch of underlying
  // element types is handled in DataModel::operator+=.
  // Different name is ok for addition.
  expect::equals(variable.unit(), other.unit());
  expect::contains(variable.dims(), other.dims());
  // Note: This will broadcast/transpose the RHS if required. We do not support
  // changing the dimensions of the LHS though!
  transform_in_place<
      pair_self_t<double, float, int64_t, Eigen::Vector3d>,
      pair_custom_t<std::pair<sparse_container<double>, double>>>(
      other, variable, [](auto &&a, auto &&b) { return a + b; });
  return variable;
}

Variable Variable::operator-() const {
  // TODO This implementation only works for variables containing doubles and
  // will throw, e.g., for ints.
  auto copy(*this);
  copy *= -1.0;
  return copy;
}

Variable &Variable::operator+=(const Variable &other) & {
  return plus_equals(*this, other);
}
Variable &Variable::operator+=(const ConstVariableSlice &other) & {
  return plus_equals(*this, other);
}
Variable &Variable::operator+=(const double value) & {
  // TODO By not setting a unit here this operator is only usable if the
  // variable is dimensionless. Should we ignore the unit for scalar operations,
  // i.e., set the same unit as *this.unit()?
  return plus_equals(*this, makeVariable<double>({}, {value}));
}

template <class T1, class T2> T1 &minus_equals(T1 &variable, const T2 &other) {
  expect::equals(variable.unit(), other.unit());
  expect::contains(variable.dims(), other.dims());
  transform_in_place<pair_self_t<double, float, int64_t, Eigen::Vector3d>>(
      other, variable, [](auto &&a, auto &&b) { return a - b; });
  return variable;
}

Variable &Variable::operator-=(const Variable &other) & {
  return minus_equals(*this, other);
}
Variable &Variable::operator-=(const ConstVariableSlice &other) & {
  return minus_equals(*this, other);
}
Variable &Variable::operator-=(const double value) & {
  return minus_equals(*this, makeVariable<double>({}, {value}));
}

template <class T1, class T2> T1 &times_equals(T1 &variable, const T2 &other) {
  expect::contains(variable.dims(), other.dims());
  // setUnit is catching bad cases of changing units (if `variable` is a slice).
  variable.setUnit(variable.unit() * other.unit());
  transform_in_place<pair_self_t<double, float, int64_t>,
                     pair_custom_t<std::pair<Eigen::Vector3d, double>>>(
      other, variable, [](auto &&a, auto &&b) { return a * b; });
  return variable;
}

Variable &Variable::operator*=(const Variable &other) & {
  return times_equals(*this, other);
}
Variable &Variable::operator*=(const ConstVariableSlice &other) & {
  return times_equals(*this, other);
}
Variable &Variable::operator*=(const double value) & {
  auto other = makeVariable<double>({}, {value});
  other.setUnit(units::dimensionless);
  return times_equals(*this, other);
}

template <class T1, class T2> T1 &divide_equals(T1 &variable, const T2 &other) {
  expect::contains(variable.dims(), other.dims());
  // setUnit is catching bad cases of changing units (if `variable` is a slice).
  variable.setUnit(variable.unit() / other.unit());
  transform_in_place<pair_self_t<double, float, int64_t>,
                     pair_custom_t<std::pair<Eigen::Vector3d, double>>>(
      other, variable, [](auto &&a, auto &&b) { return a / b; });
  return variable;
}

Variable &Variable::operator/=(const Variable &other) & {
  return divide_equals(*this, other);
}
Variable &Variable::operator/=(const ConstVariableSlice &other) & {
  return divide_equals(*this, other);
}
Variable &Variable::operator/=(const double value) & {
  return divide_equals(*this, makeVariable<double>({}, {value}));
}

template <class T> VariableSlice VariableSlice::assign(const T &other) const {
  if (unit() != other.unit())
    throw std::runtime_error("Cannot assign to slice: Unit mismatch.");
  if (dims() != other.dims())
    throw except::DimensionMismatchError(dims(), other.dims());
  data().copy(other.data(), Dim::Invalid, 0, 0, 1);
  return *this;
}

template VariableSlice VariableSlice::assign(const Variable &) const;
template VariableSlice VariableSlice::assign(const ConstVariableSlice &) const;

VariableSlice VariableSlice::operator+=(const Variable &other) const {
  return plus_equals(*this, other);
}
VariableSlice VariableSlice::operator+=(const ConstVariableSlice &other) const {
  return plus_equals(*this, other);
}
VariableSlice VariableSlice::operator+=(const double value) const {
  return plus_equals(*this, makeVariable<double>({}, {value}));
}

VariableSlice VariableSlice::operator-=(const Variable &other) const {
  return minus_equals(*this, other);
}
VariableSlice VariableSlice::operator-=(const ConstVariableSlice &other) const {
  return minus_equals(*this, other);
}
VariableSlice VariableSlice::operator-=(const double value) const {
  return minus_equals(*this, makeVariable<double>({}, {value}));
}

VariableSlice VariableSlice::operator*=(const Variable &other) const {
  return times_equals(*this, other);
}
VariableSlice VariableSlice::operator*=(const ConstVariableSlice &other) const {
  return times_equals(*this, other);
}
VariableSlice VariableSlice::operator*=(const double value) const {
  return times_equals(*this, makeVariable<double>({}, {value}));
}

VariableSlice VariableSlice::operator/=(const Variable &other) const {
  return divide_equals(*this, other);
}
VariableSlice VariableSlice::operator/=(const ConstVariableSlice &other) const {
  return divide_equals(*this, other);
}
VariableSlice VariableSlice::operator/=(const double value) const {
  return divide_equals(*this, makeVariable<double>({}, {value}));
}

bool ConstVariableSlice::operator==(const Variable &other) const {
  // Always use deep comparison (pointer comparison does not make sense since we
  // may be looking at a different section).
  return equals(*this, other);
}
bool ConstVariableSlice::operator==(const ConstVariableSlice &other) const {
  return equals(*this, other);
}

bool ConstVariableSlice::operator!=(const Variable &other) const {
  return !(*this == other);
}
bool ConstVariableSlice::operator!=(const ConstVariableSlice &other) const {
  return !(*this == other);
}

Variable ConstVariableSlice::operator-() const {
  Variable copy(*this);
  return -copy;
}

void VariableSlice::setUnit(const units::Unit &unit) const {
  // TODO Should we forbid setting the unit altogether? I think it is useful in
  // particular since views onto subsets of dataset do not imply slicing of
  // variables but return slice views.
  if ((this->unit() != unit) && (dims() != m_mutableVariable->dims()))
    throw std::runtime_error("Partial view on data of variable cannot be used "
                             "to change the unit.\n");
  m_mutableVariable->setUnit(unit);
}

template <class T>
const VariableView<const underlying_type_t<T>>
ConstVariableSlice::cast() const {
  using TT = underlying_type_t<T>;
  if (!m_view)
    return requireT<const DataModel<Vector<TT>>>(data()).valuesView(dims());
  if (m_view->isConstView())
    return requireT<const ViewModel<VariableView<const TT>>>(data()).m_model;
  // Make a const view from the mutable one.
  return {requireT<const ViewModel<VariableView<TT>>>(data()).m_model, dims()};
}

template <class T>
const VariableView<const underlying_type_t<T>>
ConstVariableSlice::castVariances() const {
  using TT = underlying_type_t<T>;
  if (!m_view)
    return requireT<const DataModel<Vector<TT>>>(data()).variancesView(dims());
  if (m_view->isConstView())
    return *requireT<const ViewModel<VariableView<const TT>>>(data())
                .m_variances;
  // Make a const view from the mutable one.
  return {*requireT<const ViewModel<VariableView<TT>>>(data()).m_variances,
          dims()};
}

template <class T>
VariableView<underlying_type_t<T>> VariableSlice::cast() const {
  using TT = underlying_type_t<T>;
  if (m_view)
    return requireT<const ViewModel<VariableView<TT>>>(data()).m_model;
  return requireT<DataModel<Vector<TT>>>(data()).valuesView(dims());
}

template <class T>
VariableView<underlying_type_t<T>> VariableSlice::castVariances() const {
  using TT = underlying_type_t<T>;
  if (m_view)
    return *requireT<const ViewModel<VariableView<TT>>>(data()).m_variances;
  return requireT<DataModel<Vector<TT>>>(data()).variancesView(dims());
}

#define INSTANTIATE_SLICEVIEW(...)                                             \
  template const VariableView<const underlying_type_t<__VA_ARGS__>>            \
  ConstVariableSlice::cast<__VA_ARGS__>() const;                               \
  template const VariableView<const underlying_type_t<__VA_ARGS__>>            \
  ConstVariableSlice::castVariances<__VA_ARGS__>() const;                      \
  template VariableView<underlying_type_t<__VA_ARGS__>>                        \
  VariableSlice::cast<__VA_ARGS__>() const;                                    \
  template VariableView<underlying_type_t<__VA_ARGS__>>                        \
  VariableSlice::castVariances<__VA_ARGS__>() const;

INSTANTIATE_SLICEVIEW(double);
INSTANTIATE_SLICEVIEW(float);
INSTANTIATE_SLICEVIEW(int64_t);
INSTANTIATE_SLICEVIEW(int32_t);
INSTANTIATE_SLICEVIEW(char);
INSTANTIATE_SLICEVIEW(bool);
INSTANTIATE_SLICEVIEW(std::string);
INSTANTIATE_SLICEVIEW(boost::container::small_vector<double, 8>);
INSTANTIATE_SLICEVIEW(Dataset);
INSTANTIATE_SLICEVIEW(Eigen::Vector3d);

ConstVariableSlice Variable::slice(const Slice slice) const & {
  return {*this, slice.dim, slice.begin, slice.end};
}

Variable Variable::slice(const Slice slice) const && {
  return {this->slice(slice)};
}

VariableSlice Variable::slice(const Slice slice) & {
  return {*this, slice.dim, slice.begin, slice.end};
}

Variable Variable::slice(const Slice slice) && { return {this->slice(slice)}; }

ConstVariableSlice Variable::operator()(const Dim dim, const scipp::index begin,
                                        const scipp::index end) const & {
  return slice({dim, begin, end});
}

VariableSlice Variable::operator()(const Dim dim, const scipp::index begin,
                                   const scipp::index end) & {
  return slice({dim, begin, end});
}

ConstVariableSlice Variable::reshape(const Dimensions &dims) const & {
  return {*this, dims};
}

VariableSlice Variable::reshape(const Dimensions &dims) & {
  return {*this, dims};
}

Variable Variable::reshape(const Dimensions &dims) && {
  Variable reshaped(std::move(*this));
  reshaped.setDimensions(dims);
  return reshaped;
}

Variable ConstVariableSlice::reshape(const Dimensions &dims) const {
  // In general a variable slice is not contiguous. Therefore we cannot reshape
  // without making a copy (except for special cases).
  Variable reshaped(*this);
  reshaped.setDimensions(dims);
  return reshaped;
}

// Note: The std::move here is necessary because RVO does not work for variables
// that are function parameters.
Variable operator+(Variable a, const Variable &b) {
  auto result = broadcast(std::move(a), b.dims());
  return result += b;
}
Variable operator-(Variable a, const Variable &b) {
  auto result = broadcast(std::move(a), b.dims());
  return result -= b;
}
Variable operator*(Variable a, const Variable &b) {
  auto result = broadcast(std::move(a), b.dims());
  return result *= b;
}
Variable operator/(Variable a, const Variable &b) {
  auto result = broadcast(std::move(a), b.dims());
  return result /= b;
}
Variable operator+(Variable a, const ConstVariableSlice &b) {
  auto result = broadcast(std::move(a), b.dims());
  return result += b;
}
Variable operator-(Variable a, const ConstVariableSlice &b) {
  auto result = broadcast(std::move(a), b.dims());
  return result -= b;
}
Variable operator*(Variable a, const ConstVariableSlice &b) {
  auto result = broadcast(std::move(a), b.dims());
  return result *= b;
}
Variable operator/(Variable a, const ConstVariableSlice &b) {
  auto result = broadcast(std::move(a), b.dims());
  return result /= b;
}
Variable operator+(Variable a, const double b) { return std::move(a += b); }
Variable operator-(Variable a, const double b) { return std::move(a -= b); }
Variable operator*(Variable a, const double b) { return std::move(a *= b); }
Variable operator/(Variable a, const double b) { return std::move(a /= b); }
Variable operator+(const double a, Variable b) { return std::move(b += a); }
Variable operator-(const double a, Variable b) { return -(b -= a); }
Variable operator*(const double a, Variable b) { return std::move(b *= a); }
Variable operator/(const double a, Variable b) {
  b.setUnit(units::Unit(units::dimensionless) / b.unit());
  transform_in_place<double, float>(
      b, overloaded{[a](const double b) { return a / b; },
                    [a](const float b) { return a / b; }});
  return std::move(b);
}

// Example of a "derived" operation: Implementation does not require adding a
// virtual function to VariableConcept.
std::vector<Variable> split(const Variable &var, const Dim dim,
                            const std::vector<scipp::index> &indices) {
  if (indices.empty())
    return {var};
  std::vector<Variable> vars;
  vars.emplace_back(var(dim, 0, indices.front()));
  for (scipp::index i = 0; i < scipp::size(indices) - 1; ++i)
    vars.emplace_back(var(dim, indices[i], indices[i + 1]));
  vars.emplace_back(var(dim, indices.back(), var.dims()[dim]));
  return vars;
}

Variable concatenate(const Variable &a1, const Variable &a2, const Dim dim) {
  if (a1.dtype() != a2.dtype())
    throw std::runtime_error(
        "Cannot concatenate Variables: Data types do not match.");
  if (a1.unit() != a2.unit())
    throw std::runtime_error(
        "Cannot concatenate Variables: Units do not match.");

  if (a1.sparseDim() == dim && a2.sparseDim() == dim) {
    Variable out(a1);
    // TODO Sanitize transform_in_place implementation so the functor signature
    // is more reasonable.
    transform_in_place<pair_self_t<sparse_container<double>>>(
        a2, out, [](auto a, const auto &b) {
          a.insert(a.end(), b.begin(), b.end());
          return a;
        });
    return out;
  }

  const auto &dims1 = a1.dims();
  const auto &dims2 = a2.dims();
  // TODO Many things in this function should be refactored and moved in class
  // Dimensions.
  // TODO Special handling for edge variables.
  for (const auto &dim1 : dims1.labels()) {
    if (dim1 != dim) {
      if (!dims2.contains(dim1))
        throw std::runtime_error(
            "Cannot concatenate Variables: Dimensions do not match.");
      if (dims2[dim1] != dims1[dim1])
        throw std::runtime_error(
            "Cannot concatenate Variables: Dimension extents do not match.");
    }
  }
  auto size1 = dims1.count();
  auto size2 = dims2.count();
  if (dims1.contains(dim))
    size1--;
  if (dims2.contains(dim))
    size2--;
  // This check covers the case of dims2 having extra dimensions not present in
  // dims1.
  // TODO Support broadcast of dimensions?
  if (size1 != size2)
    throw std::runtime_error(
        "Cannot concatenate Variables: Dimensions do not match.");

  auto out(a1);
  auto dims(dims1);
  scipp::index extent1 = 1;
  scipp::index extent2 = 1;
  if (dims1.contains(dim))
    extent1 += dims1[dim] - 1;
  if (dims2.contains(dim))
    extent2 += dims2[dim] - 1;
  if (dims.contains(dim))
    dims.resize(dim, extent1 + extent2);
  else
    dims.add(dim, extent1 + extent2);
  out.setDimensions(dims);

  out.data().copy(a1.data(), dim, 0, 0, extent1);
  out.data().copy(a2.data(), dim, extent1, 0, extent2);

  return out;
}

Variable rebin(const Variable &var, const Variable &oldCoord,
               const Variable &newCoord) {

  expect::countsOrCountsDensity(var);
  Dim dim = Dim::Invalid;
  for (const auto d : oldCoord.dims().labels())
    if (oldCoord.dims()[d] == var.dims()[d] + 1) {
      dim = d;
      break;
    }

  auto do_rebin = [dim](auto &&out, auto &&old, auto &&oldCoord,
                        auto &&newCoord) {
    // Dimensions of *this and old are guaranteed to be the same.
    const auto &oldT = *old;
    const auto &oldCoordT = *oldCoord;
    const auto &newCoordT = *newCoord;
    auto &outT = *out;
    const auto &dims = outT.dims();
    if (dims.inner() == dim &&
        isMatchingOr1DBinEdge(dim, oldCoordT.dims(), oldT.dims()) &&
        isMatchingOr1DBinEdge(dim, newCoordT.dims(), dims)) {
      RebinHelper<typename std::remove_reference_t<decltype(
          outT)>::value_type>::rebinInner(dim, oldT, outT, oldCoordT,
                                          newCoordT);
    } else {
      throw std::runtime_error(
          "TODO the new coord should be 1D or the same dim as newCoord.");
    }
  };

  if (var.unit() == units::counts ||
      var.unit() == units::counts * units::counts) {
    auto dims = var.dims();
    dims.resize(dim, newCoord.dims()[dim] - 1);
    Variable rebinned(var, dims);
    if (rebinned.dims().inner() == dim) {
      apply_in_place<double, float>(do_rebin, rebinned, var, oldCoord,
                                    newCoord);
    } else {
      if (newCoord.dims().ndim() > 1)
        throw std::runtime_error(
            "Not inner rebin works only for 1d coordinates for now.");
      switch (rebinned.dtype()) {
      case dtype<double>:
        RebinGeneralHelper<double>::rebin(dim, var, rebinned, oldCoord,
                                          newCoord);
        break;
      case dtype<float>:
        RebinGeneralHelper<float>::rebin(dim, var, rebinned, oldCoord,
                                         newCoord);
        break;
      default:
        throw std::runtime_error(
            "Rebinning is possible only for double and float types.");
      }
    }
    return rebinned;
  } else {
    // TODO This will currently fail if the data is a multi-dimensional density.
    // Would need a conversion that converts only the rebinned dimension.
    // TODO This could be done more efficiently without a temporary Dataset.
    throw std::runtime_error("Temporarily disabled for refactor");
    /*
    Dataset density;
    density.insert(dimensionCoord(dim), oldCoord);
    density.insert(Data::Value, var);
    auto cnts = counts::fromDensity(std::move(density), dim).erase(Data::Value);
    Dataset rebinnedCounts;
    rebinnedCounts.insert(dimensionCoord(dim), newCoord);
    rebinnedCounts.insert(Data::Value,
                          rebin(std::get<Variable>(cnts), oldCoord, newCoord));
    return std::get<Variable>(
        counts::toDensity(std::move(rebinnedCounts), dim).erase(Data::Value));
    */
  }
}

Variable permute(const Variable &var, const Dim dim,
                 const std::vector<scipp::index> &indices) {
  auto permuted(var);
  for (size_t i = 0; i < indices.size(); ++i)
    permuted.data().copy(var.data(), dim, i, indices[i], indices[i] + 1);
  return permuted;
}

Variable filter(const Variable &var, const Variable &filter) {
  if (filter.dims().ndim() != 1)
    throw std::runtime_error(
        "Cannot filter variable: The filter must by 1-dimensional.");
  const auto dim = filter.dims().labels()[0];
  auto mask = filter.values<bool>();

  const scipp::index removed = std::count(mask.begin(), mask.end(), 0);
  if (removed == 0)
    return var;

  auto out(var);
  auto dims = out.dims();
  dims.resize(dim, dims[dim] - removed);
  out.setDimensions(dims);

  scipp::index iOut = 0;
  // Note: Could copy larger chunks of applicable for better(?) performance.
  // Note: This implementation is inefficient, since we need to cast to concrete
  // type for *every* slice. Should be combined into a single virtual call.
  for (scipp::index iIn = 0; iIn < mask.size(); ++iIn)
    if (mask[iIn])
      out.data().copy(var.data(), dim, iOut++, iIn, iIn + 1);
  return out;
}

Variable sum(const Variable &var, const Dim dim) {
  auto summed(var);
  auto dims = summed.dims();
  dims.erase(dim);
  // setDimensions zeros the data
  summed.setDimensions(dims);
  transform_in_place<pair_self_t<double, float, int64_t, Eigen::Vector3d>>(
      var, summed, [](auto &&a, auto &&b) { return a + b; });
  return summed;
}

Variable mean(const Variable &var, const Dim dim) {
  auto summed = sum(var, dim);
  double scale = 1.0 / static_cast<double>(var.dims()[dim]);
  return summed * makeVariable<double>({}, {scale});
}

Variable abs(const Variable &var) {
  return transform<double, float>(var, [](const auto x) { return ::abs(x); });
}

Variable norm(const Variable &var) {
  return transform<Eigen::Vector3d>(var, [](auto &&x) { return x.norm(); });
}

Variable sqrt(const Variable &var) {
  Variable result =
      transform<double, float>(var, [](const auto x) { return std::sqrt(x); });
  result.setUnit(sqrt(var.unit()));
  return result;
}

Variable broadcast(Variable var, const Dimensions &dims) {
  if (var.dims().contains(dims))
    return std::move(var);
  auto newDims = var.dims();
  const auto labels = dims.labels();
  for (auto it = labels.end(); it != labels.begin();) {
    --it;
    const auto label = *it;
    if (newDims.contains(label))
      expect::dimensionMatches(newDims, label, dims[label]);
    else
      newDims.add(label, dims[label]);
  }
  Variable result(var);
  result.setDimensions(newDims);
  result.data().copy(var.data(), Dim::Invalid, 0, 0, 1);
  return result;
}

void swap(Variable &var, const Dim dim, const scipp::index a,
          const scipp::index b) {
  const Variable tmp = var(dim, a);
  var(dim, a).assign(var(dim, b));
  var(dim, b).assign(tmp);
}

Variable reverse(Variable var, const Dim dim) {
  const auto size = var.dims()[dim];
  for (scipp::index i = 0; i < size / 2; ++i)
    swap(var, dim, i, size - i - 1);
  return std::move(var);
}

template <>
VariableView<const double> getView<double>(const Variable &var,
                                           const Dimensions &dims) {
  return requireT<const VariableConceptT<double>>(var.data()).valuesView(dims);
}

} // namespace scipp::core
