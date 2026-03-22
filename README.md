# Rod Free-Volume Measurement Tool

Standalone C++ tool that measures translational and rotational free volume for each rod in a dense packing.

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

OpenMP is auto-detected. CUDA is optional (auto-detected if `nvcc` is available).

If CUDA is installed but `nvcc` is not on `PATH`, point CMake at the toolkit explicitly:

```bash
mkdir -p build && cd build
cmake .. -DROD_FREE_VOLUME_ENABLE_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build . -j$(nproc)
```

On cluster environments that use environment modules, load CUDA first and then configure with the module-provided compiler:

```bash
module load cuda/12.9.1-fasrc01
mkdir -p build-cuda && cd build-cuda
cmake .. -DROD_FREE_VOLUME_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"
cmake --build . -j$(nproc)
```

If you want a CPU-only build, disable CUDA explicitly:

```bash
mkdir -p build && cd build
cmake .. -DROD_FREE_VOLUME_ENABLE_CUDA=OFF
cmake --build . -j$(nproc)
```

## Usage

```bash
# Measure all rods in a packing
./rod_free_volume --verbose ../../examples/1,1,1/x_relaxed_AR100.txt

# Override diameter via CLI
./rod_free_volume --diameter 0.01 packing.txt

# Write output to file
./rod_free_volume --output results.csv --diameter 0.005 packing.txt

# Detailed per-direction profile for rod 5
./rod_free_volume --measure-single-rod 5 --diameter 0.005 packing.txt

# Adjust sampling resolution
./rod_free_volume --samples 720 --bisection-steps 20 --diameter 0.005 packing.txt
```

## Input Format

N lines of whitespace-separated endpoint coordinates:
```
# diameter = 0.005
# (or: # radius = 0.0025, or: # rod_radius = 0.0025)
x1 y1 z1 x2 y2 z2
x1 y1 z1 x2 y2 z2
...
```

CSV format (comma-separated) is also supported.

## Output

### Summary Mode (default)

CSV table with one row per rod:
```csv
rod_index,free_translation_area,free_solid_angle,min_translation_dist,min_rotation_angle
0,0.0234,1.456,0.012,0.034
1,0.0189,1.123,0.009,0.028
...
```

### Single-Rod Mode (`--measure-single-rod <i>`)

Per-direction profiles:
```csv
# Translation profile for rod 0
psi,free_distance
0.000000,0.0234
0.017453,0.0231
...

# Rotation profile for rod 0
phi,free_theta
0.000000,0.456
0.017453,0.448
...
```

## Algorithm

For each test rod:

1. **AABB broadphase**: prune distant rods that can never collide
2. **Translational free path**: for each direction ψ in the perpendicular plane, binary-search the maximum translation distance before capsule collision
3. **Rotational free path**: for each azimuthal angle φ, binary-search the polar angle θ at which the rotated rod first collides with a neighbor

Both use coarse scan + bisection refinement (configurable via `--theta-coarse` and `--bisection-steps`).

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--diameter <d>` | from header | Rod diameter |
| `--radius <r>` | from header | Rod radius (= diameter/2) |
| `--samples <n>` | 360 | Angular sample count |
| `--bisection-steps <n>` | 16 | Bisection refinement steps |
| `--theta-coarse <n>` | 48 | Coarse θ scan steps |
| `--measure-single-rod <i>` | off | Detailed profile for rod i |
| `--output <file>` | stdout | Output file path |
| `--threads <n>` | auto | OpenMP thread count |
| `-v, --verbose` | off | Progress to stderr |
