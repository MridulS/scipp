#ifndef WORKSPACE_H
#define WORKSPACE_H

#include <cstdint>

#include "data.h"
#include "instrument.h"
#include "metadata.h"

using IndexSet = std::vector<size_t>;

template <class Data, class Instrument = SpectrumInfo> class Workspace {
public:
  using value_type = Data;
  Workspace() = default;

  // Create with different (or same) data item type. This is in a sense similar
  // to our current workspace factory creating from a parent?
  template <class OtherData>
  explicit Workspace(const Workspace<OtherData> &other)
      : m_data(other.m_spectrumDefinitions.size()),
        m_spectrumDefinitions(other.m_spectrumDefinitions),
        m_spectrumNumbers(other.m_spectrumNumbers),
        m_instrument(other.m_instrument), m_logs(other.m_logs) {}
  template <class OtherData>
  Workspace(const Workspace<OtherData> &other, const IndexSet &indexSet)
      : m_data(other.size()),
        m_spectrumDefinitions(other.size()),
        m_spectrumNumbers(other.size()),
        m_instrument(other.m_instrument), m_logs(other.m_logs) {
    for (size_t i = 0; i < indexSet.size(); ++i) {
      m_spectrumDefinitions[i] = other.m_spectrumDefinitions[indexSet[i]];
      m_spectrumNumbers[i] = other.m_spectrumNumbers[indexSet[i]];
      // TODO similar for m_instrument
    }
  }
  typename std::vector<Data>::iterator begin() { return m_data.begin(); }
  typename std::vector<Data>::iterator end() { return m_data.end(); }
  size_t size() const { return m_data.size(); }
  Data &operator[](const size_t i) { return m_data[i]; }
  const Data &operator[](const size_t i) const { return m_data[i]; }
  Logs &logs() { return m_logs; }
  const Logs &logs() const { return m_logs; }

  template <class OtherData, class Instrument2> friend class Workspace;

private:
  std::vector<Data> m_data;
  std::vector<SpectrumDefinition> m_spectrumDefinitions;
  std::vector<int32_t> m_spectrumNumbers;
  Instrument m_instrument;
  Logs m_logs;
};

#endif // WORKSPACE_H
