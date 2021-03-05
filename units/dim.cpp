// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <limits>
#include <mutex>

#include "scipp/common/index.h"
#include "scipp/units/dim.h"

namespace scipp::units {

std::unordered_map<std::string, Dim::Id> Dim::builtin_ids{
    {"detector", Dim::Detector},
    {"dspacing", Dim::DSpacing},
    {"energy", Dim::Energy},
    {"energy_transfer", Dim::EnergyTransfer},
    {"event", Dim::Event},
    {"final_energy", Dim::FinalEnergy},
    {"group", Dim::Group},
    {"incident_energy", Dim::IncidentEnergy},
    {"<invalid>", Dim::Invalid},
    {"position", Dim::Position},
    {"pulse_time", Dim::PulseTime},
    {"Q^2", Dim::QSquared},
    {"Q", Dim::Q},
    {"Q_x", Dim::Qx},
    {"Q_y", Dim::Qy},
    {"Q_z", Dim::Qz},
    {"row", Dim::Row},
    {"scattering_angle", Dim::ScatteringAngle},
    {"spectrum", Dim::Spectrum},
    {"temperature", Dim::Temperature},
    {"time", Dim::Time},
    {"tof", Dim::Tof},
    {"wavelength", Dim::Wavelength},
    {"x", Dim::X},
    {"y", Dim::Y},
    {"z", Dim::Z}};

std::unordered_map<std::string, Dim::Id> Dim::custom_ids;
std::shared_mutex Dim::mutex;

Dim::Dim(const std::string &label) {
  if (const auto it = builtin_ids.find(label); it != builtin_ids.end()) {
    m_id = it->second;
    return;
  }
  std::shared_lock read_lock(mutex);
  if (const auto it = custom_ids.find(label); it != custom_ids.end()) {
    m_id = it->second;
    return;
  }
  const auto id = scipp::size(custom_ids) + 1000;
  if (id > std::numeric_limits<std::underlying_type<Id>::type>::max())
    throw std::runtime_error(
        "Exceeded maximum number of different dimension labels.");
  m_id = static_cast<Id>(id);
  read_lock.unlock();
  const std::unique_lock write_lock(mutex);
  custom_ids[label] = m_id;
}

std::string Dim::name() const {
  if (static_cast<int64_t>(m_id) < 1000)
    for (const auto &item : builtin_ids)
      if (item.second == m_id)
        return item.first;
  const std::shared_lock read_lock(mutex);
  for (const auto &item : custom_ids)
    if (item.second == m_id)
      return item.first;
  return "unreachable"; // throw or terminate?
}

std::string to_string(const Dim dim) { return dim.name(); }

} // namespace scipp::units
