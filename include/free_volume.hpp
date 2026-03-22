/**
 * @file free_volume.hpp
 * @brief Translational and rotational free-volume measurement algorithms.
 */
#pragma once

#include "vec3.hpp"
#include "rod.hpp"
#include <vector>

namespace fvol {

/// Parameters for the measurement algorithms
struct MeasureParams {
    int n_samples       = 360;   ///< Number of angular samples (ψ or φ)
    int theta_coarse    = 48;    ///< Coarse θ steps for rotational scan
    int bisection_steps = 16;    ///< Bisection refinement iterations
    double max_search_dist = -1; ///< Max translation search dist (auto if ≤0)
};

/// Per-rod summary
struct RodFreeVolume {
    double free_translation_area;   ///< ∫ ½ d(ψ)² dψ
    double free_solid_angle;        ///< ∫ (1 - cos θ(φ)) dφ
    double min_translation_dist;    ///< min over ψ of d(ψ)
    double min_rotation_angle;      ///< min over φ of θ(φ)
};

/// Per-direction profile (for --measure-single-rod)
struct DirectionalProfile {
    std::vector<double> angles;     ///< ψ or φ values
    std::vector<double> distances;  ///< d(ψ) or θ(φ) values
};

/// Measure translational free distance in all directions for one rod
DirectionalProfile measure_translation_profile(
    int rod_idx,
    const std::vector<Rod>& rods,
    const std::vector<int>& neighbors,
    const MeasureParams& params);

/// Measure rotational free angle in all azimuthal directions for one rod
DirectionalProfile measure_rotation_profile(
    int rod_idx,
    const std::vector<Rod>& rods,
    const std::vector<int>& neighbors,
    const MeasureParams& params);

/// Compute summary from profiles
RodFreeVolume summarize_profiles(
    const DirectionalProfile& trans,
    const DirectionalProfile& rot);

/// High-level: measure free volume for all rods (OpenMP-parallelized)
std::vector<RodFreeVolume> measure_all_rods(
    const std::vector<Rod>& rods,
    const MeasureParams& params,
    bool verbose = false);

} // namespace fvol
