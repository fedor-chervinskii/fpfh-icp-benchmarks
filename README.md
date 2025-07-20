# FPFH-ICP Benchmarks

Benchmarking classic 6D pose estimation algorithms (FPFH + ICP) on BOP benchmark datasets using Open3D.

## Overview

This repository implements and benchmarks Fast Point Feature Histogram (FPFH) feature matching followed by Iterative Closest Point (ICP) refinement for 6D object pose estimation. The benchmarking is performed against the [BOP (Benchmark for 6D Object Pose Estimation)](https://bop.felk.cvut.cz/) datasets.

## Features

- **FPFH-ICP Algorithm**: Implementation of classic feature-based 6D pose estimation
- **BOP Integration**: Direct integration with BOP toolkit for standardized evaluation
- **Multiple Datasets**: Support for various BOP datasets (LM-O, YCB-V, T-LESS, etc.)
- **Easy Benchmarking**: Simple command-line interface for running benchmarks

## Quick Start

### Installation

1. **Install UV** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS with Homebrew:
# brew install uv
```

2. **Clone the repository with submodules:**
```bash
git clone --recursive https://github.com/fedor-chervinskii/fpfh-icp-benchmarks.git
cd fpfh-icp-benchmarks
```

3. **Install dependencies with UV:**
```bash
# Create virtual environment and install dependencies
uv sync

# Add BOP toolkit dependencies
uv add imageio pypng scikit-image vispy pyopengl
```

4. **Setup BOP toolkit:**
```bash
cd bop_toolkit
uv pip install -e .
cd ..
```

### Download BOP Data

Download a BOP dataset (e.g., LM-O):
```bash
uv run python scripts/download_data.py --dataset lmo --path ./data
```

### Run Benchmark

**One-liner to run benchmarking on a single dataset:**
```bash
uv run python scripts/run_benchmark.py --dataset lmo --data_path ./data --output_dir ./results
```

**Run on a subset of objects:**
```bash
uv run python scripts/run_benchmark.py --dataset lmo --data_path ./data --objects 1,2,5 --output_dir ./results
```

**Run with custom configuration:**
```bash
uv run python scripts/run_benchmark.py --config config/datasets/lmo.yaml --data_path ./data --output_dir ./results
```

## Supported Datasets

- **LM-O** (Linemod-Occluded): `--dataset lmo`
- **YCB-V** (YCB-Video): `--dataset ycbv`
- **T-LESS**: `--dataset tless`
- **ITODD**: `--dataset itodd`
- **HB** (HomebrewedDB): `--dataset hb`

## Algorithm Details

The implemented algorithm follows these steps:
1. **Preprocessing**: Load scene and template point clouds
2. **Feature Extraction**: Compute FPFH features for both scene and template
3. **Correspondence Matching**: Find correspondences using feature similarity
4. **Initial Pose Estimation**: RANSAC-based pose estimation from correspondences
5. **Pose Refinement**: ICP refinement of the initial pose estimate
6. **Evaluation**: Compare results against ground truth using BOP metrics

## Configuration

Algorithm parameters can be configured via YAML files in the `config/` directory:

- `config/default.yaml`: Default algorithm parameters
- `config/datasets/`: Dataset-specific configurations

## Results

Results are saved in BOP-compatible format in the specified output directory:
- Pose predictions in BOP CSV format
- Evaluation metrics
- Visualization images (optional)

## Development

### Running Tests

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=html
```

### Code Formatting

```bash
# Format code with black
uv run black src scripts tests

# Sort imports with isort
uv run isort src scripts tests

# Run linting
uv run flake8 src scripts tests
```

### Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update dependencies
uv sync
```

## Project Structure

```
fpfh-icp-benchmarks/
├── src/                    # Source code
│   ├── algorithms/         # Algorithm implementations
│   ├── utils/             # Utility functions
│   └── benchmark/         # Benchmarking framework
├── config/                # Configuration files
├── scripts/               # Executable scripts
├── bop_toolkit/           # BOP toolkit submodule
└── results/               # Output directory
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{fpfh-icp-benchmarks,
    title={FPFH-ICP Benchmarks: Classic 6D Pose Estimation on BOP Datasets},
    author={Your Name},
    year={2025},
    url={https://github.com/your-username/fpfh-icp-benchmarks}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [BOP Benchmark](https://bop.felk.cvut.cz/)
- [Open3D Documentation](http://www.open3d.org/docs/)
- [FPFH Paper](https://ieeexplore.ieee.org/document/5152473)
