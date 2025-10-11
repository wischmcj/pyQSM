import pyvista as pv
import open3d as o3d
import numpy as np
from typing import Dict, List, Optional

class FilterablePointCloud(pv.PolyData):
    """
    Extension of pyvista.PolyData that maintains filter indices and can convert to open3d
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_indices: Dict[str, List[int]] = {}
        
    def add_filter(self, name: str, indices: List[int]) -> None:
        """Add a named filter with associated point indices"""
        self.filter_indices[name] = indices
        
    def remove_filter(self, name: str) -> None:
        """Remove a named filter"""
        if name in self.filter_indices:
            del self.filter_indices[name]
            
    def get_filtered_points(self, filter_name: str) -> Optional[np.ndarray]:
        """Get points corresponding to a filter name"""
        if filter_name not in self.filter_indices:
            return None
        return self.points[self.filter_indices[filter_name]]
    
    def to_o3d(self) -> o3d.geometry.PointCloud:
        """Convert to Open3D PointCloud"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        
        # Copy colors if they exist
        if self.n_arrays > 0 and 'RGB' in self.array_names:
            colors = self.get_array('RGB')
            if colors.shape[1] == 3:
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
        # Copy normals if they exist
        if self.n_arrays > 0 and 'Normals' in self.array_names:
            normals = self.get_array('Normals') 
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
        return pcd
        
    def get_filtered_o3d(self, filter_name: str) -> Optional[o3d.geometry.PointCloud]:
        """Get filtered points as Open3D PointCloud"""
        if filter_name not in self.filter_indices:
            return None
            
        filtered_points = self.get_filtered_points(filter_name)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        # Copy filtered colors if they exist
        if self.n_arrays > 0 and 'RGB' in self.array_names:
            colors = self.get_array('RGB')[self.filter_indices[filter_name]]
            if colors.shape[1] == 3:
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
        # Copy filtered normals if they exist
        if self.n_arrays > 0 and 'Normals' in self.array_names:
            normals = self.get_array('Normals')[self.filter_indices[filter_name]]
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
        return pcd
