import open3d as o3d
import numpy as np

from utils.math_utils import (
    get_angles,
    get_center,
    get_percentile,
    get_radius,
    rotation_matrix_from_arr,
    unit_vector,
    poprow,
)
from set_config import log, config

from viz.viz_utils import color_continuous_map, draw


def clean_cloud(pcd, voxels=None, neighbors=20, ratio=2.0, iters=3):
    """Reduces the number of points in the point cloud via
    voxel downsampling. Reducing noise via statistical outlier removal.
    """
    run_voxels = voxels
    run_stat = all([neighbors, ratio, iters])
    voxel_down_pcd = pcd

    if run_voxels:
        log.info("Downsample the point cloud with voxels")
        log.info(f"orig {pcd}")
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxels)
        log.info(f"downed {voxel_down_pcd}")
    if run_stat:
        log.info("Statistical oulier removal")
        for i in range(iters):
            _, ind = voxel_down_pcd.remove_statistical_outlier(    nb_neighbors=int(neighbors), std_ratio=ratio)
            voxel_down_pcd = voxel_down_pcd.select_by_index(ind)
            neighbors = neighbors * 2
            ratio = ratio / 1.5
        final = voxel_down_pcd
    else:
        final = pcd
    if not run_voxels and not run_stat:
        log.warning("No cleaning steps were run")        
    return final

def crop(pts, minx=None, maxx=None, miny=None, maxy=None, minz=None, maxz=None):
    x_vals = pts[:, 0]
    y_vals = pts[:, 1]
    z_vals = pts[:, 2]
    to_remove = []
    all_idxs = [idx for idx, _ in enumerate(pts)]
    for min_val, max_val, pt_vals in [
        (minx, maxx, x_vals),
        (miny, maxy, y_vals),
        (minz, maxz, z_vals),
    ]:
        if min_val:
            to_remove.append(np.where(pt_vals <= min_val))
        if max_val:
            to_remove.append(np.where(pt_vals >= max_val))

    select_idxs = np.setdiff1d(all_idxs, to_remove)
    return select_idxs

def crop_by_percentile(pcd, 
                  start = config['trunk']['lower_pctile'],
                  end = config['trunk']['upper_pctile'],
                  axis = 2,
                  invert = False):
    algo_source_pcd = pcd  
    algo_pcd_pts = np.asarray(algo_source_pcd.points)
    log.info(f"Getting points between the {start} and {end} percentiles")
    not_too_low_idxs, _ = get_percentile(algo_pcd_pts,start,end, axis,invert)
    low_cloud = algo_source_pcd.select_by_index(not_too_low_idxs)
    return low_cloud, not_too_low_idxs

def crop_and_highlight(pcd,lower,upper,axis):
    cropped_pcd,cropped_idxs = crop_by_percentile(pcd,lower,upper,axis)
    print(f'selecting from branch_grp')
    removed = pcd.select_by_index(cropped_idxs,invert=True)
    removed.paint_uniform_color([1,0,0])
    print(f'drawing removed')
    draw([cropped_pcd,removed])
    return cropped_pcd, cropped_idxs

def cluster_plus(pcd,
                    eps=config['trunk']['cluster_eps'],
                    min_points=config['trunk']['cluster_nn'],
                    draw_result = False,
                    color_clusters = True,
                    top=None,
                    from_points=True,
                    return_pcds = True):
    labels = np.array(pcd.cluster_dbscan(eps=.11, min_points=5,print_progress=True))
    color_continuous_map(pcd, labels)
    unique_vals, counts = np.unique(labels, return_counts=True)
    if draw_result: draw(pcd)
    num_clusters = len(unique_vals)
    if not top: top = num_clusters
    print(f"point cloud has {num_clusters} clusters")
    unique_vals, counts = np.unique(labels, return_counts=True)
    largest = unique_vals[np.argmax(counts)]
    max_cluster_idxs = np.where(labels == largest)[0]
    max_cluster = pcd.select_by_index(max_cluster_idxs)
    return max_cluster, max_cluster_idxs

def cluster_and_get_largest(pcd,
                                eps=config['trunk']['cluster_eps'],
                                min_points=config['trunk']['cluster_nn'],
                                draw_clusters = False):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    color_continuous_map(pcd, labels)
    if draw_clusters: draw(pcd)
    unique_vals, counts = np.unique(labels, return_counts=True)
    largest = unique_vals[np.argmax(counts)]
    max_cluster_idxs = np.where(labels == largest)[0]
    max_cluster = pcd.select_by_index(max_cluster_idxs)
    return max_cluster

def orientation_from_norms(norms, samples=10, max_iter=100):
    """Attempts to find the orientation of a cylindrical point cloud
    given the normals of the points. Attempts to find <samples> number
    of vectors that are orthogonal to the normals and then averages
    the third ortogonal vector (the cylinder axis) to estimate orientation.
    """
    sum_of_vectors = [0, 0, 0]
    found = 0
    iter_num = 0
    while found < samples and iter_num < max_iter and len(norms) > 1:
        iter_num += 1
        rand_id = np.random.randint(len(norms) - 1)
        norms, vect = poprow(norms, rand_id)
        dot_products = abs(np.dot(norms, vect))
        most_normal_val = min(dot_products)
        if most_normal_val <= 0.001:
            idx_of_normal = np.where(dot_products == most_normal_val)[0][0]
            most_normal = norms[idx_of_normal]
            approx_axis = np.cross(unit_vector(vect), unit_vector(most_normal))
            sum_of_vectors += approx_axis
            found += 1
    log.info(f"found {found} in {iter_num} iterations")
    axis_guess = np.asarray(sum_of_vectors) / found
    return axis_guess


def filter_by_norm(pcd, angle_thresh=10, rev = False):
    norms = np.asarray(pcd.normals)
    angles = np.apply_along_axis(get_angles, 1, norms)
    angles = np.degrees(angles)
    log.info(f"{angle_thresh=}")
    if rev:
        stem_idxs = np.where((angles < -angle_thresh) | (angles > angle_thresh))[0]
    else:
        stem_idxs = np.where((angles > -angle_thresh) & (angles < angle_thresh))[0]
    stem_cloud = pcd.select_by_index(stem_idxs)
    return stem_cloud


def get_ball_mesh(pcd,radii= [0.005, 0.01, 0.02, 0.04]):
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return rec_mesh


def get_shape(pts, shape="sphere", as_pts=True, rotate="axis", **kwargs):
    if not kwargs.get("center"):
        kwargs["center"] = get_center(pts)
    if not kwargs.get("radius"):

        kwargs["radius"] = get_radius(pts)

    if shape == "sphere":
        shape = o3d.geometry.TriangleMesh.create_sphere(radius=kwargs["radius"])
    elif shape == "cylinder":
        try:
            shape = o3d.geometry.TriangleMesh.create_cylinder(
                radius=kwargs["radius"], height=kwargs["height"]
            )
        except Exception as e:
            breakpoint()
            log.info(f"error getting cylinder {e}")

    # log.info(f'Starting Translation/Rotation')

    if as_pts:
        shape = shape.sample_points_uniformly()
        shape.paint_uniform_color([0, 1.0, 0])

    shape.translate(kwargs["center"])
    arr = kwargs.get("axis")
    if arr is not None:
        vector = unit_vector(arr)
        log.info(f"rotate vector {arr}")
        if rotate == "axis":
            R = shape.get_rotation_matrix_from_axis_angle(kwargs["axis"])
        else:
            R = rotation_matrix_from_arr([0, 0, 1], vector)
        shape.rotate(R, center=kwargs["center"])
    elif rotate == "axis":
        log.info("no axis given for rotation, not rotating")
        return shape

    return shape

def query_via_bnd_box(sub_pcd, pcd):
    from copy import deepcopy
    pcd_pts = arr(pcd.points)
    obb = pcd.get_oriented_bounding_box()
    # obb = sub_pcdpcd.get_minimal_oriented_bounding_box()
    obb.color = (1, 0, 0)
    up_shifted_obb = deepcopy(obb).translate((0,0,1),relative = True)
    up_shifted_obb.scale(.8,center = obb.center)
    draw([sub_pcd,obb,up_shifted_obb])
    new_pt_ids = up_shifted_obb.get_point_indices_within_bounding_box( 
                                o3d.utility.Vector3dVector(pcd_pts) )
    old_pt_ids = obb.get_point_indices_within_bounding_box( 
                                o3d.utility.Vector3dVector(pcd_pts) )
    next_slice =pcd.select_by_index(new_pt_ids)
    draw([next_slice,sub_pcd,obb,up_shifted_obb])
    
    new_neighbors = np.setdiff1d(new_pt_ids, old_pt_ids)
    nn_pts = pcd_pts[new_neighbors]
    total_found = new_pt_ids