#!/usr/bin/env python3
"""Main benchmark script for 6D pose estimation algorithms."""

import os
import sys
import argparse
import yaml
from typing import List

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from benchmark.runner import BenchmarkRunner


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_object_ids(obj_str: str) -> List[int]:
    """Parse object IDs from string.
    
    Args:
        obj_str: Comma-separated string of object IDs
        
    Returns:
        List of object IDs
    """
    return [int(x.strip()) for x in obj_str.split(',')]


def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(
        description='Run 6D pose estimation benchmark on BOP datasets'
    )
    
    # Dataset options
    parser.add_argument('--dataset', type=str, 
                       choices=['lmo', 'ycbv', 'tless', 'itodd', 'hb'],
                       help='BOP dataset name')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to BOP dataset root directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    
    # Configuration options
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (overrides dataset default)')
    
    # Subset options
    parser.add_argument('--scenes', type=str,
                       help='Comma-separated list of scene IDs to evaluate')
    parser.add_argument('--objects', type=str,
                       help='Comma-separated list of object IDs to evaluate')
    
    # Algorithm parameters (override config)
    parser.add_argument('--voxel_size', type=float,
                       help='Point cloud voxel size for downsampling')
    parser.add_argument('--fpfh_radius', type=float,
                       help='Radius for FPFH feature computation')
    parser.add_argument('--ransac_threshold', type=float,
                       help='RANSAC inlier threshold')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load custom config
        config = load_config(args.config)
    elif args.dataset:
        # Load dataset-specific config
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'config', 'datasets', 
            f'{args.dataset}.yaml'
        )
        if os.path.exists(config_path):
            config = load_config(config_path)
        else:
            # Fall back to default config
            default_config_path = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'default.yaml'
            )
            config = load_config(default_config_path)
            config['dataset'] = args.dataset
    else:
        parser.error('Either --dataset or --config must be specified')
    
    # Override config with command line arguments
    config['data_path'] = args.data_path
    config['output_dir'] = args.output_dir
    
    if args.dataset:
        config['dataset'] = args.dataset
    
    # Override algorithm parameters if provided
    if 'algorithm' not in config:
        config['algorithm'] = {}
    
    if args.voxel_size:
        config['algorithm']['voxel_size'] = args.voxel_size
    if args.fpfh_radius:
        config['algorithm']['fpfh_radius'] = args.fpfh_radius
    if args.ransac_threshold:
        config['algorithm']['ransac_threshold'] = args.ransac_threshold
    
    # Parse subset options
    scene_ids = None
    if args.scenes:
        scene_ids = parse_object_ids(args.scenes)
    
    object_ids = None
    if args.objects:
        object_ids = parse_object_ids(args.objects)
    elif 'object_ids' in config:
        object_ids = config['object_ids']
    
    if args.verbose:
        print("Configuration:")
        print(yaml.dump(config, default_flow_style=False))
        print(f"Scene IDs: {scene_ids}")
        print(f"Object IDs: {object_ids}")
    
    # Initialize and run benchmark
    print(f"Starting benchmark for dataset: {config['dataset']}")
    print(f"Data path: {config['data_path']}")
    print(f"Output directory: {config['output_dir']}")
    
    try:
        runner = BenchmarkRunner(config)
        results = runner.run_benchmark(scene_ids=scene_ids, object_ids=object_ids, verbose=args.verbose)
        runner.save_results(results)
        
        print("\nBenchmark completed successfully!")
        summary = results.get('summary', {})
        print(f"Total scenes processed: {summary.get('total_scenes', 0)}")
        print(f"Successful scenes: {summary.get('successful_scenes', 0)}")
        print(f"Success rate: {summary.get('success_rate', 0.0):.3f}")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
