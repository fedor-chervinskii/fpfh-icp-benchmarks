"""Point cloud utility functions."""

import numpy as np
import open3d as o3d
from typing import Tuple


def load_point_cloud(file_path: str) -> np.ndarray:
    """Load point cloud from file.
    
    Args:
        file_path: Path to point cloud file (.ply, .pcd, etc.)
        
    Returns:
        Point cloud as numpy array (N, 3)
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def save_point_cloud(points: np.ndarray, file_path: str):
    """Save point cloud to file.
    
    Args:
        points: Point cloud as numpy array (N, 3)
        file_path: Output file path
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)


def depth_to_point_cloud(depth_image: np.ndarray, 
                        intrinsics: np.ndarray,
                        depth_scale: float = 1000.0) -> np.ndarray:
    """Convert depth image to point cloud.
    
    Args:
        depth_image: Depth image (H, W)
        intrinsics: Camera intrinsic matrix (3, 3)
        depth_scale: Scale factor for depth values
        
    Returns:
        Point cloud as numpy array (N, 3)
    """
    h, w = depth_image.shape
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Get valid depth points
    valid_mask = depth_image > 0
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth_image[valid_mask] / depth_scale
    
    # Convert to camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid
    
    return np.column_stack((x, y, z))


def transform_point_cloud(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply transformation to point cloud.
    
    Args:
        points: Point cloud (N, 3)
        transform: 4x4 transformation matrix
        
    Returns:
        Transformed point cloud (N, 3)
    """
    # Convert to homogeneous coordinates
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply transformation
    transformed_homo = (transform @ points_homo.T).T
    
    return transformed_homo[:, :3]


def downsample_point_cloud(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample point cloud using voxel grid.
    
    Args:
        points: Point cloud (N, 3)
        voxel_size: Voxel size for downsampling
        
    Returns:
        Downsampled point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled.points)


def remove_outliers(points: np.ndarray, 
                   nb_neighbors: int = 20,
                   std_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Remove statistical outliers from point cloud.
    
    Args:
        points: Point cloud (N, 3)
        nb_neighbors: Number of neighbors to analyze
        std_ratio: Standard deviation ratio threshold
        
    Returns:
        Tuple of (filtered_points, inlier_indices)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    filtered_pcd, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    
    return np.asarray(filtered_pcd.points), np.array(inlier_indices)


def compute_point_cloud_bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box of point cloud.
    
    Args:
        points: Point cloud (N, 3)
        
    Returns:
        Tuple of (min_bounds, max_bounds)
    """
    return np.min(points, axis=0), np.max(points, axis=0)
