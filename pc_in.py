# with open("/code/code/Research/lidar/Secrest_27-5.xyb", mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

import numpy as np

import open3d as o3d

skeletor ="/code/code/Research/lidar/converted_pcs/skeletor.pts"
s27 ="/code/code/Research/lidar/converted_pcs/Secrest27_05.pts"
s32 ="/code/code/Research/lidar/converted_pcs/Secrest32_06.pts"

s27d = "s32_downsample_0.04.pcd"
s32d = "s32_downsample_0.04.pcd"


def read_xyz(filename):
    """Reads an XYZ point cloud file.

    Args:
        filename (str): The path to the XYZ file.

    Returns:
        numpy.ndarray: A Nx3 array of point coordinates (x, y, z).
    """

    points = []
    with open(filename, 'rb') as f:
        lines = []
        i=0
        for line in f:
            if i != 0:
                try:
                    line = line.decode('unicode_escape')
                    lines.append([x for x in line.split(' ')])
                    
                    # attr, x, y, z = line.split()
                    # points.append(list(map(float,[x, y, z])))
                except Exception as e:
                    breakpoint()
                    print(e)
                    print(line)
            i+=1

    return lines

# Example usage
# points = read_xyz(s32)
# breakpoint()

# euro6 = "/code/code/Research/downsampledlesscloudEURO6.pcd"

print("Testing IO for point cloud ...")
pcd = o3d.io.read_point_cloud("s27_downsample_0.04.pcd", format='xyz',print_progress=True)

# pcd = o3d.io.read_point_cloud("s32_downsample_0.04.pcd", format='xyz',print_progress=True)


downpcd = pcd.voxel_down_sample(voxel_size=0.04)
# o3d.io.write_point_cloud("s27_downsample_0.01.pcd", downpcd)

breakpoint()

# o3d.visualization.draw_geometries([pcd.to_legacy()])