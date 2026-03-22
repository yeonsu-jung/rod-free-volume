/**
 * @file aabb.hpp
 * @brief Axis-Aligned Bounding Box broadphase for neighbor pruning.
 */
#pragma once

#include "vec3.hpp"
#include "rod.hpp"
#include <vector>

namespace fvol {

struct AABB {
    Vec3 lo;  ///< Minimum corner
    Vec3 hi;  ///< Maximum corner

    AABB() : lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30} {}
    AABB(const Vec3& lo, const Vec3& hi) : lo(lo), hi(hi) {}

    /// Build AABB from a rod, expanded by margin on all sides
    static AABB from_rod(const Rod& rod, double margin = 0.0) {
        Vec3 lo = Vec3::min(rod.p1, rod.p2);
        Vec3 hi = Vec3::max(rod.p1, rod.p2);
        double expand = rod.radius() + margin;
        lo -= Vec3{expand, expand, expand};
        hi += Vec3{expand, expand, expand};
        return {lo, hi};
    }

    /// Expand this AABB by a uniform margin
    AABB expanded(double margin) const {
        Vec3 m{margin, margin, margin};
        return {lo - m, hi + m};
    }

    /// Test overlap with another AABB
    bool overlaps(const AABB& other) const {
        return lo.x <= other.hi.x && hi.x >= other.lo.x &&
               lo.y <= other.hi.y && hi.y >= other.lo.y &&
               lo.z <= other.hi.z && hi.z >= other.lo.z;
    }
};

/**
 * @brief Broadphase: for a given test rod, find candidate neighbors
 *        whose AABBs overlap with the test rod's expanded AABB.
 *
 * @param test_idx    Index of the test rod
 * @param rods        All rods
 * @param max_disp    Maximum displacement to consider (expands AABB)
 * @return Vector of neighbor indices
 */
inline std::vector<int> find_broadphase_neighbors(
    int test_idx,
    const std::vector<Rod>& rods,
    double max_disp)
{
    // Build an expanded AABB for the test rod that covers all possible
    // positions during translation and rotation probes.
    const Rod& test = rods[test_idx];
    double expand = test.half_length() + max_disp + test.radius();
    AABB test_aabb = AABB::from_rod(test, expand);

    std::vector<int> neighbors;
    for (int j = 0; j < static_cast<int>(rods.size()); ++j) {
        if (j == test_idx) continue;
        AABB rod_aabb = AABB::from_rod(rods[j], 0.0);
        if (test_aabb.overlaps(rod_aabb)) {
            neighbors.push_back(j);
        }
    }
    return neighbors;
}

} // namespace fvol
