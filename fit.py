
# import open3d as o3d
import numpy as np
# import scipy.spatial as sps 
# import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt


import open3d as o3d
import scipy.spatial as sps 
import scipy.cluster as spc 
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc
from open3d.visualization import draw_geometries as draw

from utils import get_radius
from vectors_shapes import get_shape


def spc_cluster_neighbors(points,current_clusters ):
    """
    https://www.comet.com/site/blog/how-to-evaluate-clustering-models-in-python/
    """
    silhouette_score_co = 0.3
    ch_max = 50
    clusters_to_try = [current_clusters-1, 
                        current_clusters, 
                        current_clusters+1]
    results = []
    codes,book = None,None
    for num in clusters_to_try:
        if num>0:
            codes, book = spc.vq.kmeans2(points, num)
            cluster_sizes = np.bincount(book)
            if num ==1:
                results.append(((codes,book,num),[.5,0,0]))
            else:
                sh_score =  silhouette_score(points,book)
                ch_score = calinski_harabasz_score(points,book)
                db_score = davies_bouldin_score(points,book)
                print(f'''num clusters: {num}, sizes: {cluster_sizes}, 
                            sh_score: {sh_score}, ch_score: {ch_score}, db_score: {db_score}''') 
                results.append(((codes,book,num),[sh_score,ch_score,db_score]))
    ret = results[0]
    for res in results:
        # if res[1][1] >= ch_max:
        #     pass
        if res[1][2]<.7 and res[1][0] >= silhouette_score_co:
            ret = res
    return ret[0]

def cluster_neighbors(pts_idxs, points,dist=.3, min_samples=5):
    clustering = DBSCAN(eps=dist, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    idxs = []
    noise = []
    # converting ids for 'points' to ids
    #   in main_pcd stored in pts_idxs
    for k in unique_labels:
        class_member_mask = labels == k
        if k == -1:
            idx_bool = class_member_mask & (core_samples_mask==False)
            pt_idxs = np.where((np.array(idx_bool)==True))
            noise = pts_idxs[pt_idxs]
        else:
            idx_bool = class_member_mask & core_samples_mask
            pt_idxs = np.where((np.array(idx_bool)==True))
            neighbor_idxs = pts_idxs[pt_idxs]
            idxs.append(neighbor_idxs)

    return unique_labels, idxs, noise

def fit_cylinder(pcd=None, pts=None, threshold=0.1):
    if pts is None:
        pts = np.asarray(pcd.points)

    # sphere_pts = get_sphere(trunk_pts)
    radius = get_radius(pts)
    cyl = pyrsc.Cylinder()
    center, axis, fit_radius, inliers = cyl.fit( pts = np.asarray(pts), thresh = threshold)
    print(f'fit_cyl = center: {center}, axis: {axis}, radius: {fit_radius}')
    if len(center)==0:
        breakpoint()
        print(f'no no fit cyl found')


    in_pts = pts[inliers]
    lowest, heighest = min(in_pts[:,2]), max(in_pts[:,2])
    height= heighest-lowest
    center_height = (height/2) + lowest
    test_center = [center[0],center[1],center_height]
    
    cyl_mesh = get_shape(pts, shape='cylinder', as_pts=False, 
                         center=tuple(test_center), radius=fit_radius*1.2, 
                         height=height)    
    in_pcd=None
    if pcd:
        draw([pcd, cyl_mesh])
        
        in_pcd = pcd.select_by_index(inliers)
        pcd.paint_uniform_color([1.0,0,0])
        in_pcd.paint_uniform_color([0,1.0,0])
        draw([pcd, in_pcd])

    return cyl_mesh, in_pcd, inliers