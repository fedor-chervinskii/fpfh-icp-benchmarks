"""Visualization utilities."""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def visualize_point_clouds(point_clouds: List[np.ndarray], 
                          colors: Optional[List[Tuple[float, float, float]]] = None,
                          window_name: str = "Point Clouds") -> None:
    """Visualize multiple point clouds.
    
    Args:
        point_clouds: List of point clouds to visualize
        colors: Optional list of RGB colors for each point cloud
        window_name: Window title
    """
    if colors is None:
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 
                 (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
    
    geometries = []
    for i, points in enumerate(point_clouds):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if i < len(colors):
            pcd.paint_uniform_color(colors[i])
        geometries.append(pcd)
    
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def visualize_correspondences(source_points: np.ndarray,
                            target_points: np.ndarray,
                            correspondences: np.ndarray,
                            window_name: str = "Correspondences") -> None:
    """Visualize point correspondences.
    
    Args:
        source_points: Source point cloud
        target_points: Target point cloud
        correspondences: Correspondence indices (N, 2)
        window_name: Window title
    """
    # Create point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    source_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    target_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green
    
    # Create lines for correspondences
    lines = []
    for i, (src_idx, tgt_idx) in enumerate(correspondences):
        lines.append([i * 2, i * 2 + 1])
    
    # Combine points for line set
    line_points = []
    for src_idx, tgt_idx in correspondences:
        line_points.append(source_points[src_idx])
        line_points.append(target_points[tgt_idx])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    
    o3d.visualization.draw_geometries([source_pcd, target_pcd, line_set], 
                                     window_name=window_name)


def visualize_registration_result(source: np.ndarray,
                                target: np.ndarray,
                                transformation: np.ndarray,
                                window_name: str = "Registration Result") -> None:
    """Visualize registration result.
    
    Args:
        source: Source point cloud
        target: Target point cloud
        transformation: 4x4 transformation matrix
        window_name: Window title
    """
    # Transform source
    source_transformed = transform_point_cloud(source, transformation)
    
    # Create point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_transformed)
    source_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)
    target_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green
    
    o3d.visualization.draw_geometries([source_pcd, target_pcd], 
                                     window_name=window_name)


def plot_metrics_over_time(metrics: dict, save_path: Optional[str] = None) -> None:
    """Plot evaluation metrics over time/iterations.
    
    Args:
        metrics: Dictionary with metric names as keys and lists of values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name, marker='o')
    
    plt.xlabel('Iteration/Time')
    plt.ylabel('Metric Value')
    plt.title('Evaluation Metrics Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_pose_visualization(poses: List[np.ndarray], 
                            labels: Optional[List[str]] = None,
                            coordinate_frame_size: float = 0.1) -> List[o3d.geometry.TriangleMesh]:
    """Create coordinate frames for pose visualization.
    
    Args:
        poses: List of 4x4 transformation matrices
        labels: Optional labels for each pose
        coordinate_frame_size: Size of coordinate frames
        
    Returns:
        List of coordinate frame geometries
    """
    geometries = []
    
    for i, pose in enumerate(poses):
        # Create coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=coordinate_frame_size
        )
        frame.transform(pose)
        
        geometries.append(frame)
    
    return geometries


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
