"""Simple test runner."""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add bop_toolkit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../bop_toolkit'))

try:
    from bop_toolkit_lib import inout
    print("BOP toolkit loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import BOP toolkit: {e}")
    inout = None

# Test our relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms.fpfh_icp import FPFHICPAlgorithm

class BenchmarkRunner:
    """Simple test runner."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize benchmark runner."""
        print("BenchmarkRunner initialized successfully!")
        self.config = config
        self.dataset_name = config['dataset']
        self.data_path = config['data_path']
        self.output_dir = config['output_dir']
        
        # Initialize algorithm
        algorithm_config = config.get('algorithm', {})
        self.algorithm = FPFHICPAlgorithm(algorithm_config)
        
        # Initialize BOP dataset paths
        self.dataset_path = Path(self.data_path) / self.dataset_name
        self.models_path = self.dataset_path / "models"
        self.test_path = self.dataset_path / "test"
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Models path: {self.models_path}")
        print(f"Test path: {self.test_path}")
        
        # Load object models info (the problematic part)
        models_info_path = self.models_path / "models_info.json"
        if models_info_path.exists() and inout is not None:
            try:
                self.models_info = inout.load_json(str(models_info_path))
                print("Models info loaded successfully")
            except Exception as e:
                print(f"Error loading models info: {e}")
                self.models_info = {}
        else:
            self.models_info = {}
            print("Models info not loaded (file doesn't exist or inout not available)")

    def run_benchmark(self, scene_ids: Optional[List[int]] = None, 
                     object_ids: Optional[List[int]] = None,
                     verbose: bool = False) -> Dict[str, Any]:
        """Run a simple benchmark test.
        
        Args:
            scene_ids: List of scene IDs to evaluate (None for all)
            object_ids: List of object IDs to evaluate (None for all)
            verbose: Whether to show detailed metrics
            
        Returns:
            Dictionary containing benchmark results
        """
        print("Starting simple benchmark evaluation...")
        
        # Get available scenes if not specified
        if scene_ids is None:
            scene_dirs = [d for d in self.test_path.iterdir() if d.is_dir() and d.name.isdigit()]
            scene_ids = [int(d.name) for d in scene_dirs]
        
        # Get available objects if not specified  
        if object_ids is None:
            if self.models_info:
                object_ids = [int(k) for k in self.models_info.keys()]
            else:
                # Fallback: just use object 1 for testing
                object_ids = [1]
        
        print(f"Available scenes: {scene_ids}")
        print(f"Target objects: {object_ids}")
        
        results = {
            'scene_results': {},
            'summary': {
                'total_scenes': len(scene_ids),
                'successful_scenes': 0,
                'success_rate': 0.0
            }
        }
        
        successful_scenes = 0
        
        # For now, just test that we can access the data structure
        for scene_id in scene_ids:
            try:
                scene_dir = self.test_path / f"{scene_id:06d}"
                print(f"Checking scene {scene_id} at {scene_dir}")
                
                if scene_dir.exists():
                    print(f"✓ Scene {scene_id} directory exists")
                    
                    # Check for required files
                    camera_file = scene_dir / "scene_camera.json"
                    gt_file = scene_dir / "scene_gt.json"
                    
                    if camera_file.exists():
                        print(f"✓ Camera file exists: {camera_file}")
                    else:
                        print(f"✗ Camera file missing: {camera_file}")
                    
                    if gt_file.exists():
                        print(f"✓ GT file exists: {gt_file}")
                    else:
                        print(f"✗ GT file missing: {gt_file}")
                    
                    # Check for depth/rgb directories
                    depth_dir = scene_dir / "depth"
                    rgb_dir = scene_dir / "rgb"
                    
                    if depth_dir.exists():
                        depth_files = list(depth_dir.glob("*.png"))
                        print(f"✓ Found {len(depth_files)} depth files")
                    else:
                        print(f"✗ Depth directory missing: {depth_dir}")
                    
                    if rgb_dir.exists():
                        rgb_files = list(rgb_dir.glob("*.png"))
                        print(f"✓ Found {len(rgb_files)} RGB files")
                    else:
                        print(f"✗ RGB directory missing: {rgb_dir}")
                    
                    successful_scenes += 1
                    results['scene_results'][scene_id] = {'status': 'success', 'files_found': True}
                else:
                    print(f"✗ Scene {scene_id} directory does not exist")
                    results['scene_results'][scene_id] = {'status': 'missing', 'files_found': False}
                    
            except Exception as e:
                print(f"✗ Error processing scene {scene_id}: {e}")
                results['scene_results'][scene_id] = {'status': 'error', 'error': str(e)}
        
        # Check models
        print("\nChecking object models...")
        for obj_id in object_ids:
            model_path = self.models_path / f"obj_{obj_id:06d}.ply"
            if model_path.exists():
                print(f"✓ Object {obj_id} model exists: {model_path}")
                
                # If verbose, run FPFH-ICP demo on this object
                if verbose:
                    try:
                        self._run_fpfh_icp_demo(model_path, obj_id)
                    except Exception as e:
                        print(f"  Demo failed: {e}")
            else:
                print(f"✗ Object {obj_id} model missing: {model_path}")
        
        results['summary']['successful_scenes'] = successful_scenes
        results['summary']['success_rate'] = successful_scenes / len(scene_ids) if scene_ids else 0.0
        
        print("\nBenchmark test completed!")
        print(f"Scenes processed: {len(scene_ids)}")
        print(f"Successful scenes: {successful_scenes}")
        print(f"Success rate: {results['summary']['success_rate']:.2%}")
        
        return results
    
    def _run_fpfh_icp_demo(self, model_path: Path, obj_id: int) -> None:
        """Run a demo of FPFH-ICP algorithm with detailed metrics."""
        import open3d as o3d
        import time
        
        print(f"  Running FPFH-ICP demo for object {obj_id}...")
        
        # Load model
        model_pcd = o3d.io.read_point_cloud(str(model_path))
        model_points = np.asarray(model_pcd.points)
        
        if len(model_points) == 0:
            print("  ✗ Empty model point cloud")
            return
        
        print(f"  Model loaded: {len(model_points)} points")
        
        # Create synthetic scene (in real use, this would be actual scene data)
        scene_points = self._create_synthetic_scene(model_points)
        print(f"  Synthetic scene created: {len(scene_points)} points")
        
        # Run FPFH-ICP with timing
        start_time = time.time()
        
        # Preprocess point clouds
        scene_pcd = self.algorithm.preprocess_point_cloud(scene_points)
        model_pcd_processed = self.algorithm.preprocess_point_cloud(model_points)
        preprocessing_time = time.time() - start_time
        
        print(f"  Preprocessing: {preprocessing_time:.3f}s")
        print(f"    Scene points after downsampling: {len(scene_pcd.points)}")
        print(f"    Model points after downsampling: {len(model_pcd_processed.points)}")
        print(f"    Voxel size: {self.algorithm.voxel_size}")
        
        # Compute FPFH features
        fpfh_start = time.time()
        scene_features = self.algorithm.compute_fpfh_features(scene_pcd)
        model_features = self.algorithm.compute_fpfh_features(model_pcd_processed)
        fpfh_time = time.time() - fpfh_start
        
        print(f"  FPFH feature extraction: {fpfh_time:.3f}s")
        print(f"    Scene features: {scene_features.data.shape}")
        print(f"    Model features: {model_features.data.shape}")
        print(f"    FPFH radius: {self.algorithm.fpfh_radius}")
        
        # Initial registration (FPFH + RANSAC)
        ransac_start = time.time()
        initial_transform, initial_fitness = self.algorithm.initial_registration(
            model_pcd_processed, scene_pcd, model_features, scene_features
        )
        ransac_time = time.time() - ransac_start
        
        print(f"  FPFH+RANSAC registration: {ransac_time:.3f}s")
        print(f"    Initial fitness: {initial_fitness:.4f}")
        print(f"    RANSAC threshold: {self.algorithm.ransac_threshold}")
        print(f"    Max iterations: {self.algorithm.max_iterations}")
        
        # ICP refinement
        icp_start = time.time()
        final_transform, final_fitness = self.algorithm.refine_registration(
            model_pcd_processed, scene_pcd, initial_transform
        )
        icp_time = time.time() - icp_start
        
        print(f"  ICP refinement: {icp_time:.3f}s")
        print(f"    Final fitness: {final_fitness:.4f}")
        print(f"    Fitness improvement: {final_fitness - initial_fitness:.4f}")
        print(f"    ICP threshold: {self.algorithm.icp_threshold}")
        
        # Calculate final alignment metrics
        model_transformed = model_pcd_processed.transform(final_transform)
        distances = model_transformed.compute_point_cloud_distance(scene_pcd)
        mean_distance = np.mean(distances)
        
        total_time = time.time() - start_time
        print(f"  Total processing time: {total_time:.3f}s")
        print(f"  Final alignment metrics:")
        print(f"    Mean point-to-point distance: {mean_distance:.6f}")
        print(f"    Algorithm confidence: {final_fitness:.4f}")
        
    def _create_synthetic_scene(self, model_points: np.ndarray) -> np.ndarray:
        """Create a synthetic scene with the model transformed and with noise."""
        # Apply a known transformation
        transform = np.array([
            [0.9, -0.1, 0.05, 0.02],
            [0.1, 0.95, -0.03, 0.01],
            [-0.05, 0.03, 0.98, 0.05],
            [0, 0, 0, 1]
        ])
        
        # Transform model points
        homogeneous = np.hstack([model_points, np.ones((len(model_points), 1))])
        transformed = (transform @ homogeneous.T).T
        scene_points = transformed[:, :3]
        
        # Add noise
        noise = np.random.normal(0, 0.0005, scene_points.shape)
        scene_points += noise
        
        # Add some clutter points
        n_clutter = len(model_points) // 6
        clutter = np.random.uniform(-0.1, 0.1, (n_clutter, 3))
        scene_points = np.vstack([scene_points, clutter])
        
        return scene_points

    def save_results(self, results: Dict[str, Any], output_path: str = None) -> None:
        """Save benchmark results to file.
        
        Args:
            results: Results dictionary
            output_path: Output file path (optional)
        """
        import yaml
        import os
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'benchmark_results.yaml')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as YAML for human readability
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)
        
        print(f"Results saved to: {output_path}")

print("BenchmarkRunner class defined successfully!")
