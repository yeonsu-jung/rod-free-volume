/**
 * @file free_volume_cuda.cu
 * @brief Optimized CUDA kernels for rod free-volume computation.
 */

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace fvol {
namespace cuda {

static constexpr double kPi  = 3.14159265358979323846;

struct d_Vec3 { double x, y, z; };
struct d_Rod { d_Vec3 p1, p2; double diameter; };

// Math helpers
__device__ d_Vec3 d_add(d_Vec3 a, d_Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ d_Vec3 d_sub(d_Vec3 a, d_Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ d_Vec3 d_scale(d_Vec3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
__device__ double d_dot(d_Vec3 a, d_Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ d_Vec3 d_cross(d_Vec3 a, d_Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
__device__ double d_norm(d_Vec3 a) { return sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }
__device__ d_Vec3 d_normalized(d_Vec3 a) {
    double n = d_norm(a);
    if (n > 1e-15) return d_scale(a, 1.0/n);
    return {0,0,0};
}
__device__ double d_clamp01(double x) { return fmin(fmax(x, 0.0), 1.0); }

// Rod helpers
__device__ d_Vec3 d_center(const d_Rod& r) { return d_scale(d_add(r.p1, r.p2), 0.5); }
__device__ d_Vec3 d_axis(const d_Rod& r) { return d_normalized(d_sub(r.p2, r.p1)); }
__device__ double d_length(const d_Rod& r) { return d_norm(d_sub(r.p2, r.p1)); }

__device__ void d_build_perp_frame(d_Vec3 axis, d_Vec3* u, d_Vec3* v) {
    d_Vec3 ref = (fabs(axis.x) < 0.9) ? d_Vec3{1,0,0} : d_Vec3{0,1,0};
    *u = d_normalized(d_cross(axis, ref));
    *v = d_normalized(d_cross(axis, *u));
}

// Distance & Collision
__device__ double d_segment_distance(d_Vec3 p1s, d_Vec3 p1e, d_Vec3 p2s, d_Vec3 p2e) {
    d_Vec3 d1 = d_sub(p1e, p1s);
    d_Vec3 d2 = d_sub(p2e, p2s);
    d_Vec3 d12 = d_sub(p2s, p1s);

    double D1 = d_dot(d1, d1);
    double D2 = d_dot(d2, d2);
    double S1 = d_dot(d1, d12);
    double S2 = d_dot(d2, d12);
    double R  = d_dot(d1, d2);
    double den = D1 * D2 - R * R;
    double t, u;

    if (D1 < 1e-30 || D2 < 1e-30) {
        if (D1 > 1e-30) { u = 0.0; t = d_clamp01(S1 / D1); }
        else if (D2 > 1e-30) { t = 0.0; u = d_clamp01(-S2 / D2); }
        else { t = u = 0.0; }
    } else if (fabs(den) < 1e-12) {
        t = 0.0; u = d_clamp01(-S2 / D2);
        double uf = d_clamp01(u);
        if (fabs(uf - u) > 1e-12) { t = d_clamp01((uf * R + S1) / D1); u = uf; }
    } else {
        t = d_clamp01((S1 * D2 - S2 * R) / den);
        u = (t * R - S2) / D2;
        double uf = d_clamp01(u);
        if (fabs(uf - u) > 1e-12) { t = d_clamp01((uf * R + S1) / D1); u = uf; }
    }

    d_Vec3 diff = d_sub(d_sub(d_scale(d1, t), d_scale(d2, u)), d12);
    return sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
}

__device__ bool d_capsule_collide_translated(const d_Rod& test, d_Vec3 offset, const d_Rod& neighbor) {
    d_Vec3 p1 = d_add(test.p1, offset);
    d_Vec3 p2 = d_add(test.p2, offset);
    return d_segment_distance(p1, p2, neighbor.p1, neighbor.p2) <= (test.diameter*0.5 + neighbor.diameter*0.5 + 1e-12);
}

__device__ bool d_capsule_collide_rotated(const d_Rod& test, d_Vec3 new_dir, const d_Rod& neighbor) {
    d_Vec3 c = d_center(test);
    double hl = d_length(test) * 0.5;
    d_Vec3 d = d_normalized(new_dir);
    d_Vec3 shift = d_scale(d, hl);
    d_Vec3 p1 = d_sub(c, shift);
    d_Vec3 p2 = d_add(c, shift);
    return d_segment_distance(p1, p2, neighbor.p1, neighbor.p2) <= (test.diameter*0.5 + neighbor.diameter*0.5 + 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel over (Rods * Samples) - Phase 1: Translation Distances
// ─────────────────────────────────────────────────────────────────────────────
__global__ void compute_trans_dists_kernel(
    const d_Rod* __restrict__ rods, int n_rods, int n_samples,
    int theta_coarse, int bisection_steps, double max_search_dist,
    double* __restrict__ out_dists)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rods * n_samples) return;

    int rod_idx = tid / n_samples;
    int sample_idx = tid % n_samples;

    d_Rod test = rods[rod_idx];
    d_Vec3 ax = d_axis(test);
    d_Vec3 u, v; d_build_perp_frame(ax, &u, &v);
    
    double default_max = max_search_dist > 0 ? max_search_dist : d_length(test);
    double expand = test.diameter*0.5 + fmax(default_max, d_length(test));
    
    double t_lo_x = fmin(test.p1.x, test.p2.x) - expand; double t_hi_x = fmax(test.p1.x, test.p2.x) + expand;
    double t_lo_y = fmin(test.p1.y, test.p2.y) - expand; double t_hi_y = fmax(test.p1.y, test.p2.y) + expand;
    double t_lo_z = fmin(test.p1.z, test.p2.z) - expand; double t_hi_z = fmax(test.p1.z, test.p2.z) + expand;

    double psi = 2.0 * kPi * sample_idx / n_samples;
    d_Vec3 dir = d_add(d_scale(u, cos(psi)), d_scale(v, sin(psi)));
    double lo = 0.0, hi = default_max;
    bool found = false;
    
    for (int step = 1; step <= theta_coarse; ++step) {
        double d = default_max * step / theta_coarse;
        d_Vec3 offset = d_scale(dir, d);
        bool collided = false;
        for (int j = 0; j < n_rods; ++j) {
            if (j == rod_idx) continue;
            d_Rod nj = rods[j];
            double r2 = nj.diameter * 0.5;
            if (t_hi_x < fmin(nj.p1.x, nj.p2.x) - r2) continue;
            if (t_lo_x > fmax(nj.p1.x, nj.p2.x) + r2) continue;
            if (t_hi_y < fmin(nj.p1.y, nj.p2.y) - r2) continue;
            if (t_lo_y > fmax(nj.p1.y, nj.p2.y) + r2) continue;
            if (t_hi_z < fmin(nj.p1.z, nj.p2.z) - r2) continue;
            if (t_lo_z > fmax(nj.p1.z, nj.p2.z) + r2) continue;

            if (d_capsule_collide_translated(test, offset, nj)) { collided = true; break; }
        }
        if (collided) {
            hi = d; lo = default_max * (step - 1) / theta_coarse;
            found = true; break;
        }
    }
    
    if (found) {
        for (int iter = 0; iter < bisection_steps; ++iter) {
            double mid = 0.5 * (lo + hi);
            d_Vec3 offset = d_scale(dir, mid);
            bool collided = false;
            for (int j = 0; j < n_rods; ++j) {
                if (j == rod_idx) continue;
                d_Rod nj = rods[j];
                double r2 = nj.diameter*0.5;
                if (t_hi_x < fmin(nj.p1.x, nj.p2.x) - r2) continue;
                if (t_lo_x > fmax(nj.p1.x, nj.p2.x) + r2) continue;
                if (t_hi_y < fmin(nj.p1.y, nj.p2.y) - r2) continue;
                if (t_lo_y > fmax(nj.p1.y, nj.p2.y) + r2) continue;
                if (t_hi_z < fmin(nj.p1.z, nj.p2.z) - r2) continue;
                if (t_lo_z > fmax(nj.p1.z, nj.p2.z) + r2) continue;

                if (d_capsule_collide_translated(test, offset, nj)) { collided = true; break; }
            }
            if (collided) hi = mid; else lo = mid;
        }
    }
    out_dists[tid] = hi;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel over (Rods * Samples) - Phase 2: Rotation Angles
// ─────────────────────────────────────────────────────────────────────────────
__global__ void compute_rot_angles_kernel(
    const d_Rod* __restrict__ rods, int n_rods, int n_samples,
    int theta_coarse, int bisection_steps, double max_search_dist,
    double* __restrict__ out_angles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rods * n_samples) return;

    int rod_idx = tid / n_samples;
    int sample_idx = tid % n_samples;

    d_Rod test = rods[rod_idx];
    d_Vec3 ax = d_axis(test);
    d_Vec3 u, v; d_build_perp_frame(ax, &u, &v);
    
    double default_max = max_search_dist > 0 ? max_search_dist : d_length(test);
    double expand = test.diameter*0.5 + fmax(default_max, d_length(test));
    
    double t_lo_x = fmin(test.p1.x, test.p2.x) - expand; double t_hi_x = fmax(test.p1.x, test.p2.x) + expand;
    double t_lo_y = fmin(test.p1.y, test.p2.y) - expand; double t_hi_y = fmax(test.p1.y, test.p2.y) + expand;
    double t_lo_z = fmin(test.p1.z, test.p2.z) - expand; double t_hi_z = fmax(test.p1.z, test.p2.z) + expand;

    double phi = 2.0 * kPi * sample_idx / n_samples;
    double lo = 0.0, hi = kPi;
    bool found = false;
    
    for (int step = 1; step < theta_coarse; ++step) {
        double theta = kPi * step / (theta_coarse - 1);
        double sin_t = sin(theta), cos_t = cos(theta);
        d_Vec3 new_dir = d_add(d_add(d_scale(u, sin_t * cos(phi)), 
                                     d_scale(v, sin_t * sin(phi))), d_scale(ax, cos_t));
        bool collided = false;
        for (int j = 0; j < n_rods; ++j) {
            if (j == rod_idx) continue;
            d_Rod nj = rods[j];
            double r2 = nj.diameter * 0.5;
            if (t_hi_x < fmin(nj.p1.x, nj.p2.x) - r2) continue;
            if (t_lo_x > fmax(nj.p1.x, nj.p2.x) + r2) continue;
            if (t_hi_y < fmin(nj.p1.y, nj.p2.y) - r2) continue;
            if (t_lo_y > fmax(nj.p1.y, nj.p2.y) + r2) continue;
            if (t_hi_z < fmin(nj.p1.z, nj.p2.z) - r2) continue;
            if (t_lo_z > fmax(nj.p1.z, nj.p2.z) + r2) continue;

            if (d_capsule_collide_rotated(test, new_dir, nj)) { collided = true; break; }
        }
        if (collided) {
            hi = theta; lo = kPi * (step - 1) / (theta_coarse - 1);
            found = true; break;
        }
    }
    
    if (found) {
        for (int iter = 0; iter < bisection_steps; ++iter) {
            double mid = 0.5 * (lo + hi);
            double sin_t = sin(mid), cos_t = cos(mid);
            d_Vec3 new_dir = d_add(d_add(d_scale(u, sin_t * cos(phi)), 
                                         d_scale(v, sin_t * sin(phi))), d_scale(ax, cos_t));
            bool collided = false;
            for (int j = 0; j < n_rods; ++j) {
                if (j == rod_idx) continue;
                d_Rod nj = rods[j];
                double r2 = nj.diameter * 0.5;
                if (t_hi_x < fmin(nj.p1.x, nj.p2.x) - r2) continue;
                if (t_lo_x > fmax(nj.p1.x, nj.p2.x) + r2) continue;
                if (t_hi_y < fmin(nj.p1.y, nj.p2.y) - r2) continue;
                if (t_lo_y > fmax(nj.p1.y, nj.p2.y) + r2) continue;
                if (t_hi_z < fmin(nj.p1.z, nj.p2.z) - r2) continue;
                if (t_lo_z > fmax(nj.p1.z, nj.p2.z) + r2) continue;

                if (d_capsule_collide_rotated(test, new_dir, nj)) { collided = true; break; }
            }
            if (collided) hi = mid; else lo = mid;
        }
    }
    out_angles[tid] = hi;
}

// ─────────────────────────────────────────────────────────────────────────────
// Summarization Kernel (1 thread per rod)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void summarize_profiles_kernel(
    int n_rods, int n_samples,
    const double* __restrict__ trans_dists,
    const double* __restrict__ rot_angles,
    double* __restrict__ out_trans_area, double* __restrict__ out_solid_angle,
    double* __restrict__ out_min_trans, double* __restrict__ out_min_rot)
{
    int rod_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rod_idx >= n_rods) return;

    double area = 0.0, min_trans = 1e30;
    double sa = 0.0, min_rot = 1e30;
    
    double dpsi = 2.0 * kPi / n_samples;
    int base_idx = rod_idx * n_samples;
    
    for (int i = 0; i < n_samples; ++i) {
        double d = trans_dists[base_idx + i];
        area += 0.5 * d * d * dpsi;
        if (d < min_trans) min_trans = d;
        
        double t = rot_angles[base_idx + i];
        sa += (1.0 - cos(t)) * dpsi;  // dphi == dpsi here
        if (t < min_rot) min_rot = t;
    }

    out_trans_area[rod_idx] = area;
    out_min_trans[rod_idx] = min_trans;
    out_solid_angle[rod_idx] = sa;
    out_min_rot[rod_idx] = min_rot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host Entry Point
// ─────────────────────────────────────────────────────────────────────────────
void launch_free_volume_cuda(
    const void* host_rods, int n_rods,
    int n_samples, int theta_coarse, int bisection_steps, double max_search_dist,
    double* h_trans_area, double* h_solid_angle,
    double* h_min_trans, double* h_min_rot)
{
    d_Rod* d_rods;
    cudaMalloc(&d_rods, n_rods * sizeof(d_Rod));
    cudaMemcpy(d_rods, host_rods, n_rods * sizeof(d_Rod), cudaMemcpyHostToDevice);

    double *d_trans_dists, *d_rot_angles;
    int total_samples = n_rods * n_samples;
    cudaMalloc(&d_trans_dists, total_samples * sizeof(double));
    cudaMalloc(&d_rot_angles, total_samples * sizeof(double));

    double *d_ta, *d_sa, *d_mt, *d_mr;
    cudaMalloc(&d_ta, n_rods * sizeof(double));
    cudaMalloc(&d_sa, n_rods * sizeof(double));
    cudaMalloc(&d_mt, n_rods * sizeof(double));
    cudaMalloc(&d_mr, n_rods * sizeof(double));

    int block = 256;
    int grid_samples = (total_samples + block - 1) / block;
    
    compute_trans_dists_kernel<<<grid_samples, block>>>(
        d_rods, n_rods, n_samples, theta_coarse, bisection_steps, max_search_dist, 
        d_trans_dists);
        
    compute_rot_angles_kernel<<<grid_samples, block>>>(
        d_rods, n_rods, n_samples, theta_coarse, bisection_steps, max_search_dist, 
        d_rot_angles);

    int grid_rods = (n_rods + block - 1) / block;
    summarize_profiles_kernel<<<grid_rods, block>>>(
        n_rods, n_samples, d_trans_dists, d_rot_angles,
        d_ta, d_sa, d_mt, d_mr);

    cudaMemcpy(h_trans_area, d_ta, n_rods * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_solid_angle, d_sa, n_rods * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min_trans, d_mt, n_rods * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min_rot, d_mr, n_rods * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_rods);
    cudaFree(d_trans_dists);
    cudaFree(d_rot_angles);
    cudaFree(d_ta); cudaFree(d_sa);
    cudaFree(d_mt); cudaFree(d_mr);
}

} // namespace cuda
} // namespace fvol

#endif // USE_CUDA
