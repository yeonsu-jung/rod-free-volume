/**
 * @file free_volume.cpp
 * @brief Implementation of translational and rotational free-volume measurement.
 */

#include "free_volume.hpp"
#include "collision.hpp"
#include "aabb.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
namespace fvol {
namespace cuda {
void launch_free_volume_cuda(
    const void* host_rods, int n_rods,
    int n_samples, int theta_coarse, int bisection_steps, double max_search_dist,
    double* h_trans_area, double* h_solid_angle,
    double* h_min_trans, double* h_min_rot);
} // namespace cuda
} // namespace fvol
#endif

namespace fvol {

static constexpr double kPi  = 3.14159265358979323846;
static constexpr double kInf = std::numeric_limits<double>::infinity();

// ────────────────────────────────────────────────────────────────────────────
// Translation: for each direction ψ, find the max distance d(ψ) the rod
// can slide in the perpendicular plane before it collides with any neighbor.
// ────────────────────────────────────────────────────────────────────────────

DirectionalProfile measure_translation_profile(
    int rod_idx,
    const std::vector<Rod>& rods,
    const std::vector<int>& neighbors,
    const MeasureParams& params)
{
    const Rod& test = rods[rod_idx];
    Vec3 ax = test.axis();
    auto [u, v] = build_perp_frame(ax);

    int N = params.n_samples;
    double max_dist = params.max_search_dist;
    if (max_dist <= 0.0) {
        max_dist = test.length();              // sensible upper bound
    }

    DirectionalProfile prof;
    prof.angles.resize(N);
    prof.distances.resize(N);

    for (int i = 0; i < N; ++i) {
        double psi = 2.0 * kPi * i / N;
        prof.angles[i] = psi;

        Vec3 dir = u * std::cos(psi) + v * std::sin(psi);

        // Coarse scan: step through distances to find the bracket
        double lo = 0.0;
        double hi = max_dist;
        bool found = false;

        int coarse_steps = params.theta_coarse;  // reuse param
        for (int step = 1; step <= coarse_steps; ++step) {
            double d = max_dist * step / coarse_steps;
            Vec3 offset = dir * d;
            bool collided = false;
            for (int nj : neighbors) {
                if (capsule_collide_translated(test, offset, rods[nj])) {
                    collided = true;
                    break;
                }
            }
            if (collided) {
                hi = d;
                lo = max_dist * (step - 1) / coarse_steps;
                found = true;
                break;
            }
        }

        if (!found) {
            prof.distances[i] = max_dist;
            continue;
        }

        // Bisection refinement
        for (int iter = 0; iter < params.bisection_steps; ++iter) {
            double mid = 0.5 * (lo + hi);
            Vec3 offset = dir * mid;
            bool collided = false;
            for (int nj : neighbors) {
                if (capsule_collide_translated(test, offset, rods[nj])) {
                    collided = true;
                    break;
                }
            }
            if (collided) hi = mid;
            else          lo = mid;
        }
        prof.distances[i] = hi;
    }

    return prof;
}

// ────────────────────────────────────────────────────────────────────────────
// Rotation: for each azimuthal angle φ, find the polar angle θ(φ) at which
// the rod first collides with any neighbor when rotated about its center.
// ────────────────────────────────────────────────────────────────────────────

DirectionalProfile measure_rotation_profile(
    int rod_idx,
    const std::vector<Rod>& rods,
    const std::vector<int>& neighbors,
    const MeasureParams& params)
{
    const Rod& test = rods[rod_idx];
    Vec3 ax = test.axis();
    auto [u, v] = build_perp_frame(ax);

    int N = params.n_samples;

    DirectionalProfile prof;
    prof.angles.resize(N);
    prof.distances.resize(N);

    for (int i = 0; i < N; ++i) {
        double phi = 2.0 * kPi * i / N;
        prof.angles[i] = phi;

        // Coarse scan over θ ∈ [0, π]
        double lo = 0.0;
        double hi = kPi;
        bool found = false;

        for (int step = 1; step < params.theta_coarse; ++step) {
            double theta = kPi * step / (params.theta_coarse - 1);

            // New direction at (θ, φ) relative to original axis
            double sin_t = std::sin(theta);
            double cos_t = std::cos(theta);
            Vec3 new_dir = u * (sin_t * std::cos(phi))
                         + v * (sin_t * std::sin(phi))
                         + ax * cos_t;

            bool collided = false;
            for (int nj : neighbors) {
                if (capsule_collide_rotated(test, new_dir, rods[nj])) {
                    collided = true;
                    break;
                }
            }
            if (collided) {
                hi = theta;
                lo = kPi * (step - 1) / (params.theta_coarse - 1);
                found = true;
                break;
            }
        }

        if (!found) {
            prof.distances[i] = kPi;
            continue;
        }

        // Bisection refinement
        for (int iter = 0; iter < params.bisection_steps; ++iter) {
            double mid = 0.5 * (lo + hi);
            double sin_t = std::sin(mid);
            double cos_t = std::cos(mid);
            Vec3 new_dir = u * (sin_t * std::cos(phi))
                         + v * (sin_t * std::sin(phi))
                         + ax * cos_t;

            bool collided = false;
            for (int nj : neighbors) {
                if (capsule_collide_rotated(test, new_dir, rods[nj])) {
                    collided = true;
                    break;
                }
            }
            if (collided) hi = mid;
            else          lo = mid;
        }
        prof.distances[i] = hi;
    }

    return prof;
}

// ────────────────────────────────────────────────────────────────────────────
// Summarize profiles into scalar metrics
// ────────────────────────────────────────────────────────────────────────────

RodFreeVolume summarize_profiles(
    const DirectionalProfile& trans,
    const DirectionalProfile& rot)
{
    RodFreeVolume rv{};

    // Translation area = ∫ ½ d(ψ)² dψ  (via trapezoidal / uniform spacing)
    int Nt = static_cast<int>(trans.angles.size());
    if (Nt > 0) {
        double dpsi = 2.0 * kPi / Nt;
        double area = 0.0;
        double min_d = kInf;
        for (int i = 0; i < Nt; ++i) {
            area += 0.5 * trans.distances[i] * trans.distances[i] * dpsi;
            min_d = std::min(min_d, trans.distances[i]);
        }
        rv.free_translation_area = area;
        rv.min_translation_dist = min_d;
    }

    // Solid angle = ∫ (1 - cos θ(φ)) dφ
    int Nr = static_cast<int>(rot.angles.size());
    if (Nr > 0) {
        double dphi = 2.0 * kPi / Nr;
        double sa = 0.0;
        double min_theta = kInf;
        for (int i = 0; i < Nr; ++i) {
            sa += (1.0 - std::cos(rot.distances[i])) * dphi;
            min_theta = std::min(min_theta, rot.distances[i]);
        }
        rv.free_solid_angle = sa;
        rv.min_rotation_angle = min_theta;
    }

    return rv;
}

// ────────────────────────────────────────────────────────────────────────────
// Batch: measure all rods with OpenMP
// ────────────────────────────────────────────────────────────────────────────

std::vector<RodFreeVolume> measure_all_rods(
    const std::vector<Rod>& rods,
    const MeasureParams& params,
    bool verbose)
{
    int N = static_cast<int>(rods.size());
    std::vector<RodFreeVolume> results(N);

    // Determine max displacement for broadphase AABB expansion.
    // For translation probes, the rod can move at most max_search_dist.
    // For rotation probes, the center stays fixed but endpoints sweep a sphere
    // of radius = half_length.
    // We take the maximum rod length as a conservative upper bound.
    double max_len = 0.0;
    for (const auto& r : rods) {
        max_len = std::max(max_len, r.length());
    }
    double max_disp = (params.max_search_dist > 0)
                      ? std::max(params.max_search_dist, max_len)
                      : max_len;

#ifdef USE_CUDA
    if (verbose) {
        std::cerr << "  Running with CUDA acceleration...\n";
    }
    
    std::vector<double> h_trans_area(N);
    std::vector<double> h_solid_angle(N);
    std::vector<double> h_min_trans(N);
    std::vector<double> h_min_rot(N);

    cuda::launch_free_volume_cuda(
        rods.data(), N,
        params.n_samples, params.theta_coarse, params.bisection_steps, params.max_search_dist,
        h_trans_area.data(), h_solid_angle.data(),
        h_min_trans.data(), h_min_rot.data());

    for (int i = 0; i < N; ++i) {
        results[i].free_translation_area = h_trans_area[i];
        results[i].free_solid_angle      = h_solid_angle[i];
        results[i].min_translation_dist  = h_min_trans[i];
        results[i].min_rotation_angle    = h_min_rot[i];
    }
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < N; ++i) {
        if (verbose) {
#ifdef USE_OPENMP
            if (omp_get_thread_num() == 0)
#endif
            {
                std::cerr << "\r  Processing rod " << (i + 1) << "/" << N
                          << std::flush;
            }
        }

        // Broadphase neighbor list
        auto neighbors = find_broadphase_neighbors(i, rods, max_disp);

        auto trans = measure_translation_profile(i, rods, neighbors, params);
        auto rot   = measure_rotation_profile(i, rods, neighbors, params);
        results[i] = summarize_profiles(trans, rot);
    }

    if (verbose) {
        std::cerr << "\n";
    }
#endif
    return results;
}

} // namespace fvol
