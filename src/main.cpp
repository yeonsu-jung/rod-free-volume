/**
 * @file main.cpp
 * @brief CLI entry point for the rod free-volume measurement tool.
 *
 * Usage:
 *   rod_free_volume [options] <input_file>
 *
 * Reads an N×6 endpoint file, measures translational and rotational
 * free volume for each rod, and writes a summary table.
 */

#include "free_volume.hpp"
#include "aabb.hpp"
#include "rod.hpp"
#include "vec3.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fvol;

// ──────────────────────────────────────────────────────────────────────────
// CLI option struct
// ──────────────────────────────────────────────────────────────────────────

struct CLIOptions {
    std::string input_file;
    std::string output_file;           // empty = stdout
    double diameter          = -1.0;   // override from header
    double radius_override   = -1.0;
    int    n_samples         = 360;
    int    bisection_steps   = 16;
    int    theta_coarse      = 48;
    int    measure_single    = -1;     // -1 = off
    bool   verbose           = false;
};

// ──────────────────────────────────────────────────────────────────────────
// Parse CLI
// ──────────────────────────────────────────────────────────────────────────

static void print_usage() {
    std::cout <<
R"(Usage: rod_free_volume [options] <input_file>

Measures translational and rotational free volume for rods in a packing.

Input file: N lines of "x1 y1 z1 x2 y2 z2" (rod endpoints).
            Diameter can be specified in a header comment:
              # diameter = 0.005
              # radius = 0.0025
              # rod_radius = 0.0025

Options:
  --diameter <d>           Rod diameter (overrides header)
  --radius <r>             Rod radius (overrides header)
  --samples <n>            Angular sample count [default: 360]
  --bisection-steps <n>    Bisection refinement steps [default: 16]
  --theta-coarse <n>       Coarse theta scan steps [default: 48]
  --measure-single-rod <i> Output detailed per-direction profiles for rod i
  --output <file>          Output file path [default: stdout]
  --threads <n>            OpenMP thread count (if compiled with OpenMP)
  -v, --verbose            Print progress to stderr
  -h, --help               Show this help
)";
}

static CLIOptions parse_args(int argc, char** argv) {
    CLIOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };

        if (arg == "--diameter")           { opts.diameter = std::stod(next()); }
        else if (arg == "--radius")        { opts.radius_override = std::stod(next()); }
        else if (arg == "--samples")       { opts.n_samples = std::stoi(next()); }
        else if (arg == "--bisection-steps") { opts.bisection_steps = std::stoi(next()); }
        else if (arg == "--theta-coarse")  { opts.theta_coarse = std::stoi(next()); }
        else if (arg == "--measure-single-rod") { opts.measure_single = std::stoi(next()); }
        else if (arg == "--output")        { opts.output_file = next(); }
        else if (arg == "--threads") {
            int nt = std::stoi(next());
#ifdef USE_OPENMP
            omp_set_num_threads(nt);
#else
            (void)nt;
            std::cerr << "Warning: --threads ignored (not compiled with OpenMP)\n";
#endif
        }
        else if (arg == "-v" || arg == "--verbose") { opts.verbose = true; }
        else if (arg == "-h" || arg == "--help") { print_usage(); std::exit(0); }
        else if (arg[0] == '-') { throw std::runtime_error("Unknown option: " + arg); }
        else { opts.input_file = arg; }
    }

    if (opts.input_file.empty()) {
        print_usage();
        throw std::runtime_error("No input file specified");
    }
    return opts;
}

// ──────────────────────────────────────────────────────────────────────────
// Load rods from endpoint file
// ──────────────────────────────────────────────────────────────────────────

static std::vector<Rod> load_rods(const std::string& filename,
                                  double& diameter_out)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    double parsed_diameter = -1.0;
    std::vector<Rod> rods;
    std::string line;

    while (std::getline(file, line)) {
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        // Handle comment/header lines
        if (line[0] == '#') {
            // Try to parse diameter or radius from header
            auto try_parse = [&](const std::string& key) -> double {
                auto pos = line.find(key);
                if (pos == std::string::npos) return -1.0;
                pos = line.find('=', pos);
                if (pos == std::string::npos) return -1.0;
                return std::stod(line.substr(pos + 1));
            };

            double d = try_parse("diameter");
            if (d > 0) { parsed_diameter = d; continue; }

            double r = try_parse("rod_radius");
            if (r > 0) { parsed_diameter = 2.0 * r; continue; }

            r = try_parse("radius");
            if (r > 0) { parsed_diameter = 2.0 * r; continue; }

            continue;
        }

        // Parse endpoint data
        std::stringstream ss(line);
        double x1, y1, z1, x2, y2, z2;
        // Handle both space and comma separated
        char c;
        if (line.find(',') != std::string::npos) {
            // CSV format
            if (!(ss >> x1 >> c >> y1 >> c >> z1 >> c >> x2 >> c >> y2 >> c >> z2))
                continue;
        } else {
            if (!(ss >> x1 >> y1 >> z1 >> x2 >> y2 >> z2))
                continue;
        }

        rods.emplace_back(Vec3{x1, y1, z1}, Vec3{x2, y2, z2}, 0.0);
    }

    diameter_out = parsed_diameter;
    return rods;
}

// ──────────────────────────────────────────────────────────────────────────
// Output helpers
// ──────────────────────────────────────────────────────────────────────────

static void write_summary_table(
    std::ostream& out,
    const std::vector<RodFreeVolume>& results)
{
    out << "rod_index,free_translation_area,free_solid_angle,"
           "min_translation_dist,min_rotation_angle\n";
    out << std::setprecision(10);
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << i << ','
            << r.free_translation_area << ','
            << r.free_solid_angle << ','
            << r.min_translation_dist << ','
            << r.min_rotation_angle << '\n';
    }
}

static void write_single_rod_profiles(
    std::ostream& out,
    int rod_idx,
    const DirectionalProfile& trans,
    const DirectionalProfile& rot)
{
    out << std::setprecision(10);

    out << "# Translation profile for rod " << rod_idx << "\n";
    out << "psi,free_distance\n";
    for (size_t i = 0; i < trans.angles.size(); ++i) {
        out << trans.angles[i] << ',' << trans.distances[i] << '\n';
    }

    out << "\n# Rotation profile for rod " << rod_idx << "\n";
    out << "phi,free_theta\n";
    for (size_t i = 0; i < rot.angles.size(); ++i) {
        out << rot.angles[i] << ',' << rot.distances[i] << '\n';
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    try {
        CLIOptions opts = parse_args(argc, argv);

        // Load rods
        double header_diameter = -1.0;
        auto rods = load_rods(opts.input_file, header_diameter);
        if (rods.empty()) {
            throw std::runtime_error("No rods loaded from " + opts.input_file);
        }

        // Determine diameter
        double diameter = 0.0;
        if (opts.diameter > 0) {
            diameter = opts.diameter;
        } else if (opts.radius_override > 0) {
            diameter = 2.0 * opts.radius_override;
        } else if (header_diameter > 0) {
            diameter = header_diameter;
        } else {
            throw std::runtime_error(
                "Rod diameter not specified. Use --diameter, --radius, "
                "or include '# diameter = ...' / '# radius = ...' in the input file.");
        }

        // Apply diameter to all rods
        for (auto& r : rods) {
            r.diameter = diameter;
        }

        if (opts.verbose) {
            std::cerr << "Loaded " << rods.size() << " rods from "
                      << opts.input_file << "\n"
                      << "  Diameter: " << diameter << "\n"
                      << "  Rod length (first): " << rods[0].length() << "\n"
                      << "  Samples: " << opts.n_samples << "\n"
                      << "  Bisection steps: " << opts.bisection_steps << "\n"
                      << "  Coarse theta: " << opts.theta_coarse << "\n";
        }

        MeasureParams params;
        params.n_samples       = opts.n_samples;
        params.bisection_steps = opts.bisection_steps;
        params.theta_coarse    = opts.theta_coarse;

        // Choose output stream
        std::ofstream out_file;
        std::ostream* out = &std::cout;
        if (!opts.output_file.empty()) {
            out_file.open(opts.output_file);
            if (!out_file.is_open()) {
                throw std::runtime_error("Cannot open output file: " + opts.output_file);
            }
            out = &out_file;
        }

        if (opts.measure_single >= 0) {
            // ── Single-rod detailed mode ──
            int idx = opts.measure_single;
            if (idx < 0 || idx >= static_cast<int>(rods.size())) {
                throw std::runtime_error("Rod index " + std::to_string(idx) +
                    " out of range [0, " + std::to_string(rods.size() - 1) + "]");
            }

            if (opts.verbose) {
                std::cerr << "Measuring single rod " << idx << " ...\n";
            }

            // Get broadphase neighbors (use rod length as max displacement)
            double max_disp = rods[idx].length();
            auto neighbors = find_broadphase_neighbors(idx, rods, max_disp);

            if (opts.verbose) {
                std::cerr << "  Broadphase neighbors: " << neighbors.size() << "\n";
            }

            auto trans = measure_translation_profile(idx, rods, neighbors, params);
            auto rot   = measure_rotation_profile(idx, rods, neighbors, params);

            write_single_rod_profiles(*out, idx, trans, rot);

            // Also print summary line to stderr
            auto summary = summarize_profiles(trans, rot);
            std::cerr << "  Free translation area: " << summary.free_translation_area << "\n"
                      << "  Free solid angle:      " << summary.free_solid_angle << "\n"
                      << "  Min translation dist:  " << summary.min_translation_dist << "\n"
                      << "  Min rotation angle:    " << summary.min_rotation_angle << "\n";
        } else {
            // ── Batch mode: all rods ──
            if (opts.verbose) {
                std::cerr << "Measuring free volume for all " << rods.size()
                          << " rods ...\n";
            }

            auto results = measure_all_rods(rods, params, opts.verbose);
            write_summary_table(*out, results);

            if (opts.verbose) {
                // Print aggregate statistics
                double sum_ta = 0, sum_sa = 0;
                for (const auto& r : results) {
                    sum_ta += r.free_translation_area;
                    sum_sa += r.free_solid_angle;
                }
                double N = static_cast<double>(results.size());
                std::cerr << "\nSummary:\n"
                          << "  Mean free translation area: " << (sum_ta / N) << "\n"
                          << "  Mean free solid angle:      " << (sum_sa / N) << "\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
