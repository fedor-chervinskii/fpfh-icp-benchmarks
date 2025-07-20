"""Base algorithm interface."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple


class BaseAlgorithm(ABC):
    """Base class for 6D pose estimation algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize algorithm with configuration.
        
        Args:
            config: Algorithm configuration parameters
        """
        self.config = config
        
    @abstractmethod
    def estimate_pose(self, 
                     scene_pcd: np.ndarray, 
                     template_pcd: np.ndarray,
                     camera_intrinsics: np.ndarray) -> Tuple[np.ndarray, float]:
        """Estimate 6D pose of template in scene.
        
        Args:
            scene_pcd: Scene point cloud (N, 3)
            template_pcd: Template/model point cloud (M, 3)
            camera_intrinsics: Camera intrinsic matrix (3, 3)
            
        Returns:
            Tuple of (pose_matrix, confidence_score)
            pose_matrix: 4x4 transformation matrix
            confidence_score: Algorithm confidence in the result
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get algorithm name."""
        pass
