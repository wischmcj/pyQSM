
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
from matplotlib import pyplot as plt

from utils import get_radius, get_center
from vectors_shapes import get_shape

def kmeans(points,min_clusters ):
    """
    https://www.comet.com/site/blog/how-to-evaluate-clustering-models-in-python/
    """
    # silhouette_score_co = 0.3
    # ch_max = 50
    clusters_to_try = [ min_clusters, 
                        min_clusters+1, 
                        min_clusters+2]
    results = []
    codes,book = None,None
    pts_2d = points[:,:2]
    for num in clusters_to_try:
        if num>0:
            codes, book = spc.vq.kmeans2(pts_2d, num)
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
    plt.scatter(pts_2d[:,0],pts_2d[:,1],c=book)
    plt.show()
    breakpoint()
    # ret = results[0]
    # for res in results:
    #     # if res[1][1] >= ch_max:
    #     #     pass
    #     if res[1][2]<.7 and res[1][0] >= silhouette_score_co:
    #         ret = res
    return results

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

def fit_shape(pcd=None, pts=None, 
                 threshold=0.1, draw_pcd = False,
                 lower_bound= None,max_radius=None,
                 shape = 'circle'):
    if pts is None:
        pts = np.asarray(pcd.points)
    if lower_bound:
        for pt in pts:
            if pt[2] < lower_bound:
                pt[2] = lower_bound
    
    lowest, heighest = min(pts[:,2]), max(pts[:,2])


    radius = get_radius(pts)
    fit_pts = np.asarray(pts.copy())
    if shape == 'circle':
        for pt in fit_pts:
            pt[2] = 0
        shape = pyrsc.Circle()
    if shape == 'cylinder':
        shape = pyrsc.Cylinder()

    center, axis, fit_radius, inliers = shape.fit(pts = np.asarray(fit_pts), thresh = threshold)
    print(f'fit_cyl = center: {center}, axis: {axis}, radius: {fit_radius}')
    
    # if max_radius is not None:
    #     if fit_radius> max_radius:
    #         print(f'{shape} had radius {fit_radius} but max_radius is {max_radius}')
    #         return None, None, None, None, None

    if len(center)==0:
        print(f'no no fit {shape} found')
        return None, None, None, None, None
    
    in_pts = pts[inliers]
    lowest, heighest = min(in_pts[:,2]), max(in_pts[:,2])
    height= heighest-lowest
    center_height = (height/2) + lowest
    test_center = [center[0],center[1],center_height]

    # test_center = center
    if height<=0:
        breakpoint()
        return None, None, None, None, None

    # if ((axis[0] == 0 and axis[1] == 0 and axis[2] == 1) or
    #     (axis[0] == 0 and axis[1] == 0 and axis[2] == -1)):
    #     print(f'No Rotation Needed')
    #     cyl_mesh = get_shape(pts, shape='cylinder', as_pts=False,  
    #                         center=tuple(test_center), 
    #                         radius=fit_radius*1.2,  
    #                         height=height)    
    # else:
    #     print(f'Rotation Needed')
    #     cyl_mesh = get_shape(pts, shape='cylinder', as_pts=False,  
    #                         center=tuple(test_center), 
    #                         radius=fit_radius*1.2,  
    #                         height=height,
    #                         axis=axis)   
    # 
    shape_radius = fit_radius*1.05
    cyl_mesh = get_shape(pts, shape='cylinder', as_pts=False,  
                            center=tuple(test_center), 
                            radius=shape_radius,
                            height=height)    
    in_pcd=None
    if pcd is not None:
        in_pcd = pcd.select_by_index(inliers)
        pcd.paint_uniform_color([1.0,0,0])
        in_pcd.paint_uniform_color([0,1.0,0])
        if draw_pcd:
            draw([pcd, cyl_mesh])
            draw([pcd, in_pcd])

    return cyl_mesh, in_pcd, inliers, fit_radius, axis