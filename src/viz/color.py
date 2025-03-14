from numpy import array as arr, mean
import scipy.spatial as sps
import  open3d as o3d

from viz.viz_utils import draw

def homog_colors(pcd):
    colors = arr(pcd.colors)
    pts=arr(pcd.points)
    white_idxs = [idc for idc,color in enumerate(colors) if sum(color)>2.7]
    white_pts = [pts[idc] for idc in white_idxs]

    non_white_tree = pcd.select_by_index(white_idxs, invert=True)
    tree = sps.KDTree(arr(non_white_tree.points))
    white_nbrs = tree.query([white_pts],30)
    avg_neighbor_color = mean(colors[ white_nbrs[1]][0],axis=1)
    for white_idx,avg_color in zip(white_idxs,avg_neighbor_color):  colors[white_idx] = avg_color
    pcd.colors =  o3d.utility.Vector3dVector(colors)
    draw(pcd)
    
def remove_color_pts(pcd,
                        color_lambda = lambda x: sum(x)>2.7,
                        invert=False):
    colors = arr(pcd.colors)
    ids = [idc for idc, color in enumerate(colors) if color_lambda(color)]
    new_pcd = pcd.select_by_index(ids, invert = invert)
    return new_pcd

def get_green_surfaces(pcd,invert=False):
    green_pcd= remove_color_pts(pcd,    lambda rgb: rgb[1]>rgb[0] and rgb[1]>rgb[2] and 0.5<(rgb[0]/rgb[2])<2, invert)
    return green_pcd

def mute_colors(pcd):
    colors = arr(pcd.colors)
    rnd_colors = [[round(a,2) for a in col] for col in colors]
    pcd.colors= o3d.utility.Vector3dVector(rnd_colors)
    draw(pcd)

def bin_colors(zoom_region=[(97.67433, 118.449), (342.1023, 362.86)]):
    # binning colors in pcd, finding most common
    # round(2) reduces 350k to ~47k colors
    region = [(pt,color) for pt,color in zip(low_pts,low_colors) 
                if (pt[0]>zoom_region[0][0] and pt[0]<zoom_region[0][1]       
                    and pt[1]>zoom_region[1][0] 
                    and  pt[1]<zoom_region[1][1])]

    cols = [','.join([f'{round(y,1)}' for y in x[1]]) for x in region]
    d_cols, cnts = np.unique(cols, return_counts=True)
    print(len(d_cols))
    print((max(cnts),min(cnts)))
    # removing points of any color other than the most common
    most_common = d_cols[np.where(cnts)]
    most_common_rgb = [tuple((float(num) for num in col.split(','))) for col in most_common]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) in most_common_rgb ]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) != tuple((1.0,1.0,1.0)) ]
    limited_pcd = o3d.geometry.PointCloud()
    limited_pcd.points = o3d.utility.Vector3dVector(arr([x[0] for x in color_limited]))
    limited_pcd.colors = o3d.utility.Vector3dVector(arr([x[1] for x in color_limited]))
    draw(limited_pcd)