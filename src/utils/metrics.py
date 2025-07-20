"""Evaluation metrics for 6D pose estimation."""

import numpy as np
from typing import List, Dict, Tuple


def rotation_error(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Calculate rotation error in degrees.
    
    Args:
        R_est: Estimated rotation matrix (3x3)
        R_gt: Ground truth rotation matrix (3x3)
        
    Returns:
        Rotation error in degrees
    """
    # Compute relative rotation
    R_rel = R_gt.T @ R_est
    
    # Calculate angle from rotation matrix
    trace = np.trace(R_rel)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    return np.degrees(angle)


def translation_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Calculate translation error in meters.
    
    Args:
        t_est: Estimated translation vector (3,)
        t_gt: Ground truth translation vector (3,)
        
    Returns:
        Translation error in meters
    """
    return np.linalg.norm(t_est - t_gt)


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    """Calculate both rotation and translation error.
    
    Args:
        T_est: Estimated 4x4 transformation matrix
        T_gt: Ground truth 4x4 transformation matrix
        
    Returns:
        Tuple of (rotation_error_deg, translation_error_m)
    """
    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]
    
    rot_err = rotation_error(R_est, R_gt)
    trans_err = translation_error(t_est, t_gt)
    
    return rot_err, trans_err


def add_error(points: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray) -> float:
    """Calculate Average Distance of Model Points (ADD) error.
    
    Args:
        points: Model points (N, 3)
        T_est: Estimated 4x4 transformation matrix
        T_gt: Ground truth 4x4 transformation matrix
        
    Returns:
        ADD error in meters
    """
    # Transform points with estimated and ground truth poses
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    
    points_est = (T_est @ points_homo.T).T[:, :3]
    points_gt = (T_gt @ points_homo.T).T[:, :3]
    
    # Calculate average distance
    distances = np.linalg.norm(points_est - points_gt, axis=1)
    return np.mean(distances)


def adds_error(points: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray) -> float:
    """Calculate Average Distance of Model Points Symmetric (ADD-S) error.
    
    For symmetric objects, finds closest points instead of corresponding points.
    
    Args:
        points: Model points (N, 3)
        T_est: Estimated 4x4 transformation matrix
        T_gt: Ground truth 4x4 transformation matrix
        
    Returns:
        ADD-S error in meters
    """
    from scipy.spatial.distance import cdist
    
    # Transform points with estimated and ground truth poses
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    
    points_est = (T_est @ points_homo.T).T[:, :3]
    points_gt = (T_gt @ points_homo.T).T[:, :3]
    
    # Find closest points (symmetric matching)
    distances = cdist(points_est, points_gt)
    min_distances = np.min(distances, axis=1)
    
    return np.mean(min_distances)


def projection_error(points: np.ndarray, 
                    T_est: np.ndarray, 
                    T_gt: np.ndarray,
                    K: np.ndarray) -> float:
    """Calculate 2D projection error.
    
    Args:
        points: 3D model points (N, 3)
        T_est: Estimated 4x4 transformation matrix
        T_gt: Ground truth 4x4 transformation matrix
        K: Camera intrinsic matrix (3, 3)
        
    Returns:
        Average 2D projection error in pixels
    """
    # Transform points to camera coordinates
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    
    points_est_cam = (T_est @ points_homo.T).T[:, :3]
    points_gt_cam = (T_gt @ points_homo.T).T[:, :3]
    
    # Project to image plane
    def project_points(points_3d, intrinsics):
        points_2d_homo = (intrinsics @ points_3d.T).T
        return points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    proj_est = project_points(points_est_cam, K)
    proj_gt = project_points(points_gt_cam, K)
    
    # Calculate 2D distances
    distances_2d = np.linalg.norm(proj_est - proj_gt, axis=1)
    return np.mean(distances_2d)


def calculate_bop_metrics(poses_est: List[np.ndarray],
                         poses_gt: List[np.ndarray],
                         model_points: np.ndarray,
                         thresholds: Dict[str, float] = None) -> Dict[str, float]:
    """Calculate BOP-style evaluation metrics.
    
    Args:
        poses_est: List of estimated poses
        poses_gt: List of ground truth poses
        model_points: 3D model points
        thresholds: Thresholds for different metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    if thresholds is None:
        thresholds = {
            'add': 0.1,  # 10% of object diameter
            'adds': 0.1,
            'rot': 15.0,  # degrees
            'trans': 0.05  # meters
        }
    
    results = {
        'add_errors': [],
        'adds_errors': [],
        'rot_errors': [],
        'trans_errors': []
    }
    
    for T_est, T_gt in zip(poses_est, poses_gt):
        # Calculate errors
        add_err = add_error(model_points, T_est, T_gt)
        adds_err = adds_error(model_points, T_est, T_gt)
        rot_err, trans_err = pose_error(T_est, T_gt)
        
        results['add_errors'].append(add_err)
        results['adds_errors'].append(adds_err)
        results['rot_errors'].append(rot_err)
        results['trans_errors'].append(trans_err)
    
    # Calculate success rates
    results['add_auc'] = np.mean(np.array(results['add_errors']) < thresholds['add'])
    results['adds_auc'] = np.mean(np.array(results['adds_errors']) < thresholds['adds'])
    results['rot_acc'] = np.mean(np.array(results['rot_errors']) < thresholds['rot'])
    results['trans_acc'] = np.mean(np.array(results['trans_errors']) < thresholds['trans'])
    
    # Mean errors
    results['mean_add'] = np.mean(results['add_errors'])
    results['mean_adds'] = np.mean(results['adds_errors'])
    results['mean_rot'] = np.mean(results['rot_errors'])
    results['mean_trans'] = np.mean(results['trans_errors'])
    
    return results
