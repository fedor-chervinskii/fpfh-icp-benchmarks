"""FPFH + ICP algorithm implementation."""

import numpy as np
import open3d as o3d
from typing import Dict, Any, Tuple
from .base_algorithm import BaseAlgorithm


class FPFHICPAlgorithm(BaseAlgorithm):
    """FPFH feature matching followed by ICP refinement."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FPFH-ICP algorithm.
        
        Args:
            config: Configuration dictionary with parameters:
                - voxel_size: Downsampling voxel size
                - normal_radius: Radius for normal estimation
                - fpfh_radius: Radius for FPFH feature computation
                - ransac_threshold: RANSAC inlier threshold
                - max_iterations: Maximum RANSAC iterations
                - icp_threshold: ICP convergence threshold
        """
        super().__init__(config)
        self.voxel_size = config.get('voxel_size', 0.005)
        self.normal_radius = config.get('normal_radius', 0.01)
        self.fpfh_radius = config.get('fpfh_radius', 0.025)
        self.ransac_threshold = config.get('ransac_threshold', 0.01)
        self.max_iterations = config.get('max_iterations', 100000)
        self.icp_threshold = config.get('icp_threshold', 0.002)
        
    def get_name(self) -> str:
        """Get algorithm name."""
        return "FPFH-ICP"
    
    def preprocess_point_cloud(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        """Preprocess point cloud: downsample and compute normals.
        
        Args:
            points: Raw point cloud (N, 3)
            
        Returns:
            Preprocessed Open3D point cloud with normals
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Downsample
        pcd = pcd.voxel_down_sample(self.voxel_size)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius, max_nn=30
            )
        )
        
        return pcd
    
    def compute_fpfh_features(self, pcd: o3d.geometry.PointCloud) -> o3d.pipelines.registration.Feature:
        """Compute FPFH features for point cloud.
        
        Args:
            pcd: Point cloud with normals
            
        Returns:
            FPFH features
        """
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.fpfh_radius, max_nn=100
            )
        )
    
    def initial_registration(self, 
                           source: o3d.geometry.PointCloud,
                           target: o3d.geometry.PointCloud,
                           source_features: o3d.pipelines.registration.Feature,
                           target_features: o3d.pipelines.registration.Feature) -> Tuple[np.ndarray, float]:
        """Perform initial registration using FPFH features and RANSAC.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            source_features: Source FPFH features
            target_features: Target FPFH features
            
        Returns:
            Tuple of (transformation_matrix, fitness_score)
        """
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_features, target_features,
            mutual_filter=True,
            max_correspondence_distance=self.ransac_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.ransac_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(self.max_iterations, 0.999)
        )
        
        return result.transformation, result.fitness
    
    def refine_registration(self, 
                          source: o3d.geometry.PointCloud,
                          target: o3d.geometry.PointCloud,
                          initial_transform: np.ndarray) -> Tuple[np.ndarray, float]:
        """Refine registration using ICP.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            initial_transform: Initial transformation from FPFH matching
            
        Returns:
            Tuple of (refined_transformation, fitness_score)
        """
        result = o3d.pipelines.registration.registration_icp(
            source, target, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        return result.transformation, result.fitness
    
    def estimate_pose(self, 
                     scene_pcd: np.ndarray, 
                     template_pcd: np.ndarray,
                     camera_intrinsics: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """Estimate 6D pose using FPFH + ICP.
        
        Args:
            scene_pcd: Scene point cloud (N, 3)
            template_pcd: Template point cloud (M, 3)
            camera_intrinsics: Camera intrinsics (not used in this algorithm)
            
        Returns:
            Tuple of (pose_matrix, confidence_score)
        """
        # Preprocess point clouds
        scene = self.preprocess_point_cloud(scene_pcd)
        template = self.preprocess_point_cloud(template_pcd)
        
        # Compute FPFH features
        scene_features = self.compute_fpfh_features(scene)
        template_features = self.compute_fpfh_features(template)
        
        # Initial registration with FPFH + RANSAC
        initial_transform, initial_fitness = self.initial_registration(
            template, scene, template_features, scene_features
        )
        
        # Refine with ICP
        final_transform, final_fitness = self.refine_registration(
            template, scene, initial_transform
        )
        
        return final_transform, final_fitness
