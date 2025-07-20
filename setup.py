from setuptools import setup, find_packages

setup(
    name="fpfh-icp-benchmarks",
    version="0.1.0",
    description="Benchmarking classic 6D pose estimation algorithms on BOP datasets",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "open3d>=0.17.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pyyaml>=6.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "fpfh-benchmark=scripts.run_benchmark:main",
        ],
    },
)
