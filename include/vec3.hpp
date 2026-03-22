/**
 * @file vec3.hpp
 * @brief Self-contained 3D vector type for the free-volume tool.
 */
#pragma once

#include <cmath>
#include <algorithm>

namespace fvol {

struct Vec3 {
    double x, y, z;

    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3& operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }

    double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const {
        return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
    }

    double norm() const { return std::sqrt(x * x + y * y + z * z); }
    double norm2() const { return x * x + y * y + z * z; }

    Vec3 normalized() const {
        double n = norm();
        return (n > 1e-15) ? (*this / n) : Vec3{0, 0, 0};
    }

    /// Component-wise min
    static Vec3 min(const Vec3& a, const Vec3& b) {
        return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
    }
    /// Component-wise max
    static Vec3 max(const Vec3& a, const Vec3& b) {
        return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
    }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

} // namespace fvol
