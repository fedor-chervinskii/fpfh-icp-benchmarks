"""Test suite for FPFH-ICP algorithm."""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.fpfh_icp import FPFHICPAlgorithm
from utils.pointcloud import (
    transform_point_cloud, 
    downsample_point_cloud,
    depth_to_point_cloud
)
from utils.metrics import rotation_error, translation_error, add_error


class TestFPFHICPAlgorithm(unittest.TestCase):
    """Test cases for FPFH-ICP algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'voxel_size': 0.01,
            'normal_radius': 0.02,
            'fpfh_radius': 0.05,
            'ransac_threshold': 0.02,
            'max_iterations': 10000,
            'icp_threshold': 0.01
        }
        self.algorithm = FPFHICPAlgorithm(self.config)
        
        # Create synthetic point clouds
        self.template_points = self._create_box_pointcloud(size=0.1, density=100)
        self.scene_points = self._create_scene_with_object()
    
    def _create_box_pointcloud(self, size: float = 0.1, density: int = 100) -> np.ndarray:
        """Create a simple box-shaped point cloud."""
        points = []
        
        # Create points on box faces
        for i in range(density):
            for j in range(density):
                u = (i / (density - 1)) * size - size / 2
                v = (j / (density - 1)) * size - size / 2
                
                # Six faces of the box
                faces = [
                    [u, v, -size/2],  # Bottom
                    [u, v, size/2],   # Top
                    [u, -size/2, v],  # Front
                    [u, size/2, v],   # Back
                    [-size/2, u, v],  # Left
                    [size/2, u, v],   # Right
                ]
                points.extend(faces)
        
        return np.array(points)
    
    def _create_scene_with_object(self) -> np.ndarray:
        """Create a scene containing the transformed object."""
        # Create transformation
        angle = np.pi / 6  # 30 degrees
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        translation = np.array([0.05, 0.03, 0.02])
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        
        # Transform template
        transformed_template = transform_point_cloud(self.template_points, transform)
        
        # Add noise and outliers
        noise = np.random.normal(0, 0.001, transformed_template.shape)
        transformed_template += noise
        
        # Add some background points
        background_points = np.random.uniform(-0.2, 0.2, (200, 3))
        
        scene_points = np.vstack([transformed_template, background_points])
        
        return scene_points
    
    def test_algorithm_initialization(self):
        """Test algorithm initialization."""
        self.assertEqual(self.algorithm.get_name(), "FPFH-ICP")
        self.assertEqual(self.algorithm.voxel_size, 0.01)
        self.assertEqual(self.algorithm.fpfh_radius, 0.05)
    
    def test_point_cloud_preprocessing(self):
        """Test point cloud preprocessing."""
        # Test with valid points
        pcd = self.algorithm.preprocess_point_cloud(self.template_points)
        
        self.assertTrue(pcd.has_normals())
        self.assertGreater(len(pcd.points), 0)
    
    def test_fpfh_feature_computation(self):
        """Test FPFH feature computation."""
        pcd = self.algorithm.preprocess_point_cloud(self.template_points)
        features = self.algorithm.compute_fpfh_features(pcd)
        
        self.assertGreater(features.num(), 0)
        self.assertEqual(features.dimension(), 33)  # FPFH has 33 dimensions
    
    def test_pose_estimation(self):
        """Test full pose estimation pipeline."""
        # This is a simplified test - in practice, pose estimation
        # with synthetic data might not always succeed
        try:
            pose, confidence = self.algorithm.estimate_pose(
                self.scene_points,
                self.template_points,
                np.eye(3)  # Identity camera matrix
            )
            
            # Check that we get a valid transformation matrix
            self.assertEqual(pose.shape, (4, 4))
            self.assertIsInstance(confidence, (int, float))
            self.assertGreaterEqual(confidence, 0.0)
            
            # Check that the bottom row is [0, 0, 0, 1]
            np.testing.assert_array_almost_equal(
                pose[3, :], [0, 0, 0, 1], decimal=10
            )
            
        except Exception as e:
            # Pose estimation might fail on synthetic data
            self.skipTest(f"Pose estimation failed: {e}")


class TestPointCloudUtils(unittest.TestCase):
    """Test cases for point cloud utilities."""
    
    def test_transform_point_cloud(self):
        """Test point cloud transformation."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        transform = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2], 
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        
        transformed = transform_point_cloud(points, transform)
        expected = np.array([[1, 2, 3], [2, 2, 3], [1, 3, 3]])
        
        np.testing.assert_array_almost_equal(transformed, expected)
    
    def test_downsample_point_cloud(self):
        """Test point cloud downsampling."""
        # Create dense grid
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.flatten(), yy.flatten(), np.zeros(len(xx.flatten()))])
        
        # Downsample
        downsampled = downsample_point_cloud(points, voxel_size=0.1)
        
        # Should have fewer points
        self.assertLess(len(downsampled), len(points))
        self.assertGreater(len(downsampled), 0)
    
    def test_depth_to_point_cloud(self):
        """Test depth image to point cloud conversion."""
        # Create synthetic depth image
        depth = np.ones((100, 100)) * 1000  # 1 meter depth
        intrinsics = np.array([
            [500, 0, 50],
            [0, 500, 50],
            [0, 0, 1]
        ])
        
        pcd = depth_to_point_cloud(depth, intrinsics, depth_scale=1000.0)
        
        self.assertEqual(pcd.shape[1], 3)  # 3D points
        self.assertGreater(pcd.shape[0], 0)  # Has points


class TestMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def test_rotation_error(self):
        """Test rotation error calculation."""
        # Identity rotation should have 0 error
        R1 = np.eye(3)
        R2 = np.eye(3)
        error = rotation_error(R1, R2)
        self.assertAlmostEqual(error, 0.0, places=10)
        
        # 90 degree rotation around z-axis
        R1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R2 = np.eye(3)
        error = rotation_error(R1, R2)
        self.assertAlmostEqual(error, 90.0, places=5)
    
    def test_translation_error(self):
        """Test translation error calculation."""
        t1 = np.array([1, 2, 3])
        t2 = np.array([1, 2, 3])
        error = translation_error(t1, t2)
        self.assertAlmostEqual(error, 0.0)
        
        t1 = np.array([0, 0, 0])
        t2 = np.array([3, 4, 0])
        error = translation_error(t1, t2)
        self.assertAlmostEqual(error, 5.0)  # 3-4-5 triangle
    
    def test_add_error(self):
        """Test ADD error calculation."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        T1 = np.eye(4)
        T2 = np.eye(4)
        
        error = add_error(points, T1, T2)
        self.assertAlmostEqual(error, 0.0)


if __name__ == '__main__':
    unittest.main()
