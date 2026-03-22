/**
 * @file test_basic.cpp
 * @brief Basic validation tests for rod free-volume measurement.
 *
 * Compile & run:
 *   cd tools/rod-free-volume/build
 *   cmake .. && make
 *   ./rod_free_volume_test
 */

#include "collision.hpp"
#include "free_volume.hpp"
#include "rod.hpp"
#include "aabb.hpp"
#include "vec3.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace fvol;

static constexpr double kPi  = 3.14159265358979323846;
static constexpr double kTol = 1e-2;  // 1% tolerance for integration tests

static int tests_passed = 0;
static int tests_total  = 0;

#define CHECK(cond, msg) do { \
    ++tests_total; \
    if (!(cond)) { \
        std::cerr << "FAIL [" << __LINE__ << "]: " << msg << "\n"; \
    } else { \
        ++tests_passed; \
        std::cout << "  OK: " << msg << "\n"; \
    } \
} while(0)

// ──────────────────────────────────────────────────────────────────────────
// Test 1: Segment distance basics
// ──────────────────────────────────────────────────────────────────────────
void test_segment_distance() {
    std::cout << "\n=== Segment Distance ===\n";

    // Parallel segments offset in y
    double d = segment_distance(
        {0,0,0}, {1,0,0},
        {0,0.5,0}, {1,0.5,0}
    );
    CHECK(std::abs(d - 0.5) < 1e-10, "Parallel segments at distance 0.5");

    // Crossing segments
    d = segment_distance(
        {-1,0,0}, {1,0,0},
        {0,-1,1}, {0,1,1}
    );
    // Closest points are (0,0,0) and (0,0,1) → distance = 1.0
    CHECK(std::abs(d - 1.0) < 1e-10, "Crossing segments at distance 1.0");

    // Coincident (touching)
    d = segment_distance(
        {0,0,0}, {1,0,0},
        {0.5,0,0}, {0.5,1,0}
    );
    CHECK(d < 1e-10, "Touching segments distance ≈ 0");
}

// ──────────────────────────────────────────────────────────────────────────
// Test 2: Capsule collision
// ──────────────────────────────────────────────────────────────────────────
void test_capsule_collision() {
    std::cout << "\n=== Capsule Collision ===\n";

    Rod a({0,0,0}, {1,0,0}, 0.1);   // diameter 0.1, radius 0.05
    Rod b({0,0.04,0}, {1,0.04,0}, 0.1);  // gap = 0.04 < 0.05+0.05 → collide
    CHECK(capsules_collide(a, b), "Overlapping capsules collide");

    Rod c({0,0.11,0}, {1,0.11,0}, 0.1);  // gap = 0.11 > 0.10 → no collision
    CHECK(!capsules_collide(a, c), "Separated capsules don't collide");
}

// ──────────────────────────────────────────────────────────────────────────
// Test 3: Single rod in empty space → full freedom
// ──────────────────────────────────────────────────────────────────────────
void test_single_rod_free() {
    std::cout << "\n=== Single Rod (no neighbors) ===\n";

    Rod rod({0,0,0}, {0,0,1}, 0.01);
    std::vector<Rod> rods = {rod};
    std::vector<int> neighbors = {};  // no neighbors

    MeasureParams params;
    params.n_samples = 72;
    params.theta_coarse = 12;
    params.bisection_steps = 8;

    auto rot = measure_rotation_profile(0, rods, neighbors, params);
    auto summary = summarize_profiles(
        measure_translation_profile(0, rods, neighbors, params), rot);

    // Free solid angle should be 4π (full sphere)
    CHECK(std::abs(summary.free_solid_angle - 4.0 * kPi) < kTol,
          "Free solid angle = 4π for isolated rod (got " +
          std::to_string(summary.free_solid_angle) + ")");

    // Min rotation angle should be π
    CHECK(std::abs(summary.min_rotation_angle - kPi) < kTol,
          "Min rotation angle = π for isolated rod");
}

// ──────────────────────────────────────────────────────────────────────────
// Test 4: Two parallel rods – known translational constraint
// ──────────────────────────────────────────────────────────────────────────
void test_two_parallel_rods() {
    std::cout << "\n=== Two Parallel Rods ===\n";

    double gap = 0.1;
    double diam = 0.02;
    Rod rod_a({0, 0, 0}, {0, 0, 1}, diam);
    Rod rod_b({gap, 0, 0}, {gap, 0, 1}, diam);
    std::vector<Rod> rods = {rod_a, rod_b};

    // For rod 0, neighbor is rod 1
    std::vector<int> neighbors = {1};
    MeasureParams params;
    params.n_samples = 360;
    params.theta_coarse = 48;
    params.bisection_steps = 16;

    auto trans = measure_translation_profile(0, rods, neighbors, params);
    auto summary = summarize_profiles(trans,
        measure_rotation_profile(0, rods, neighbors, params));

    // The minimum translation distance across all ψ should be ≈ gap - diameter
    // (in the direction toward the neighbor rod)
    double expected_min = gap - diam;  // 0.08
    double actual_min = summary.min_translation_dist;
    CHECK(std::abs(actual_min - expected_min) < 0.01,
          "Min translation distance: expected ~" +
          std::to_string(expected_min) + ", got " + std::to_string(actual_min));
}

// ──────────────────────────────────────────────────────────────────────────
// Test 5: AABB broadphase
// ──────────────────────────────────────────────────────────────────────────
void test_aabb_broadphase() {
    std::cout << "\n=== AABB Broadphase ===\n";

    double diam = 0.01;
    Rod rod_a({0,0,0}, {0,0,1}, diam);
    Rod rod_b({0.05,0,0}, {0.05,0,1}, diam);   // close
    Rod rod_c({100,100,100}, {100,100,101}, diam);  // very far
    std::vector<Rod> rods = {rod_a, rod_b, rod_c};

    auto neighbors = find_broadphase_neighbors(0, rods, 1.0);
    bool has_b = false, has_c = false;
    for (int n : neighbors) {
        if (n == 1) has_b = true;
        if (n == 2) has_c = true;
    }
    CHECK(has_b, "Broadphase includes nearby rod");
    CHECK(!has_c, "Broadphase excludes distant rod");
}

// ──────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "Rod Free-Volume Basic Tests\n";

    test_segment_distance();
    test_capsule_collision();
    test_single_rod_free();
    test_two_parallel_rods();
    test_aabb_broadphase();

    std::cout << "\n────────────────────────────────\n"
              << tests_passed << "/" << tests_total << " tests passed.\n";
    return (tests_passed == tests_total) ? 0 : 1;
}
