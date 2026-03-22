/**
 * @file rod.hpp
 * @brief Rod (capsule) representation using two endpoints and a diameter.
 */
#pragma once

#include "vec3.hpp"
#include <utility>

namespace fvol {

struct Rod {
    Vec3 p1;        ///< First endpoint
    Vec3 p2;        ///< Second endpoint
    double diameter; ///< Rod diameter (capsule thickness)

    Rod() : p1{}, p2{}, diameter(0.0) {}
    Rod(const Vec3& p1, const Vec3& p2, double diameter = 0.0)
        : p1(p1), p2(p2), diameter(diameter) {}

    Vec3 center() const { return (p1 + p2) * 0.5; }
    Vec3 axis() const { return (p2 - p1).normalized(); }
    double length() const { return (p2 - p1).norm(); }
    double half_length() const { return length() * 0.5; }
    double radius() const { return diameter * 0.5; }

    /// Get endpoints as pair
    std::pair<Vec3, Vec3> endpoints() const { return {p1, p2}; }

    /// Create a translated copy
    Rod translated(const Vec3& offset) const {
        return Rod(p1 + offset, p2 + offset, diameter);
    }

    /// Create a copy with new orientation (keep center, set new direction)
    Rod reoriented(const Vec3& new_direction) const {
        Vec3 c = center();
        double hl = half_length();
        Vec3 d = new_direction.normalized();
        return Rod(c - d * hl, c + d * hl, diameter);
    }
};

/// Build a local orthonormal frame perpendicular to a rod axis.
/// Returns (u, v) where {u, v, axis} form a right-handed frame.
inline std::pair<Vec3, Vec3> build_perp_frame(const Vec3& axis) {
    Vec3 ref = (std::abs(axis.x) < 0.9) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 u = axis.cross(ref).normalized();
    Vec3 v = axis.cross(u).normalized();
    return {u, v};
}

} // namespace fvol
