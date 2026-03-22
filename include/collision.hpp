/**
 * @file collision.hpp
 * @brief Segment-segment distance and capsule collision tests.
 */
#pragma once

#include "vec3.hpp"
#include "rod.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

namespace fvol {

/**
 * @brief Find closest-approach parameters (t, u) ∈ [0,1]² for two segments.
 *        Segment 1: p1s + t*(p1e - p1s), Segment 2: p2s + u*(p2e - p2s)
 */
inline std::pair<double, double> find_closest_parameters(
    const Vec3& d1, const Vec3& d2, const Vec3& d12)
{
    double D1 = d1.dot(d1);
    double D2 = d2.dot(d2);
    double S1 = d1.dot(d12);
    double S2 = d2.dot(d12);
    double R  = d1.dot(d2);
    double den = D1 * D2 - R * R;

    auto clamp01 = [](double x) { return std::clamp(x, 0.0, 1.0); };
    constexpr double TOL = 1e-30;

    double t, u;
    if (D1 < TOL || D2 < TOL) {
        if (D1 > TOL) { u = 0.0; t = clamp01(S1 / D1); }
        else if (D2 > TOL) { t = 0.0; u = clamp01(-S2 / D2); }
        else { t = u = 0.0; }
    } else if (std::abs(den) < 1e-12) {
        t = 0.0; u = clamp01(-S2 / D2);
        double uf = u;
        if (std::abs(uf - u) > 1e-12) {
            t = clamp01((uf * R + S1) / D1);
            u = uf;
        }
    } else {
        t = clamp01((S1 * D2 - S2 * R) / den);
        u = (t * R - S2) / D2;
        double uf = clamp01(u);
        if (std::abs(uf - u) > 1e-12) {
            t = clamp01((uf * R + S1) / D1);
            u = uf;
        }
    }
    return {t, u};
}

/**
 * @brief Minimum distance between two line segments.
 */
inline double segment_distance(
    const Vec3& p1s, const Vec3& p1e,
    const Vec3& p2s, const Vec3& p2e)
{
    Vec3 d1 = p1e - p1s;
    Vec3 d2 = p2e - p2s;
    Vec3 d12 = p2s - p1s;
    auto [t, u] = find_closest_parameters(d1, d2, d12);
    Vec3 diff = d1 * t - d2 * u - d12;
    return diff.norm();
}

/**
 * @brief Test if two capsules (rods with diameter) overlap.
 * @return true if surface-to-surface distance ≤ 0
 */
inline bool capsules_collide(const Rod& a, const Rod& b) {
    double dist = segment_distance(a.p1, a.p2, b.p1, b.p2);
    double contact_dist = a.radius() + b.radius();
    return dist <= contact_dist + 1e-12;
}

/**
 * @brief Test if a translated test rod collides with a neighbor.
 */
inline bool capsule_collide_translated(
    const Rod& test, const Vec3& offset,
    const Rod& neighbor)
{
    Vec3 p1 = test.p1 + offset;
    Vec3 p2 = test.p2 + offset;
    double dist = segment_distance(p1, p2, neighbor.p1, neighbor.p2);
    return dist <= (test.radius() + neighbor.radius() + 1e-12);
}

/**
 * @brief Test if a rotated test rod collides with a neighbor.
 *        The rod is rotated about its center to a new direction.
 */
inline bool capsule_collide_rotated(
    const Rod& test, const Vec3& new_direction,
    const Rod& neighbor)
{
    Vec3 c = test.center();
    double hl = test.half_length();
    Vec3 d = new_direction.normalized();
    Vec3 p1 = c - d * hl;
    Vec3 p2 = c + d * hl;
    double dist = segment_distance(p1, p2, neighbor.p1, neighbor.p2);
    return dist <= (test.radius() + neighbor.radius() + 1e-12);
}

} // namespace fvol
