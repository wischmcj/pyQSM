import numpy as np
from numpy import array as arr

rot_90_y = np.array([[0,1,0],[1,0,0],[0,0,-1]]) # rotates dead on
rot_90_z = np.array([[0,-1,0],[1,0,0],[0,0,1]]) 
rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])
rot_n90_x = np.array([[1,0,0],[0,-1,0],[0,0,1]]) 

def get_percentile(pts, low, high, axis=2, invert = False):
    """
    Returns the indices of the points that fall within the
    low and high percentiles of the z values of the points
    """
    if isinstance(axis,list):
        vals = sum([pts[:, axum] for axum in axis])
        if invert:
           vals =  pts[:, axis[0]] -  pts[:, axis[1]]
    else:
        vals = pts[:, axis]
    lower = np.percentile(vals, low)
    upper = np.percentile(vals, high)
    all_idxs = np.where(vals)
    too_low_idxs = np.where(vals <= lower)
    not_too_low_idxs = np.setdiff1d(all_idxs, too_low_idxs)
    if high<100:
        too_high_idxs = np.where(vals >= upper)
        select_idxs = np.setdiff1d(not_too_low_idxs, too_high_idxs)
    else:
        select_idxs = not_too_low_idxs
    vals = vals[select_idxs]
    # Similar but by scalar max
    # zmin = np.min(pts[:,2])
    # min_mask = np.where(pts[:, 2] <= (zmin+.4))[0]
    # pcd_minus_ground = pcd.select_by_index(min_mask, invert=True)
    return select_idxs, vals


def poprow(my_array, pr):
    """Row popping in numpy arrays
    Input: my_array - NumPy array, pr: row index to pop out
    Output: [new_array,popped_row]"""
    i = pr
    pop = my_array[i]
    new_array = np.vstack((my_array[:i], my_array[i + 1 :]))
    return new_array, pop


def find_normal(a, norms, min_allowed_dot=None):
    """
    For vector a, finds an approximately normal 
        vector in a given list of vectors (norms)

    """
    min_dot = min_allowed_dot
    candidate = None
    if min_allowed_dot is None:
        # ensures most normal vector is selected if 
        #  no min is passed
        min_dot = np.dot(a, norms[0])
        candidate = norms[0]
    for norm in norms[1:]:
        dot = np.dot(a, norm)
        if dot < min_dot:
            min_dot = dot
            candidate = norm
    if candidate is None:
        raise ValueError("No near normal vector found")
    return candidate
        

def rotation_matrix_from_arr(a, b: np.array):
    """
    Returns matrix R such that a*R = b.
    """
    if np.linalg.norm(b) == 0:
        return np.eye(3)
    if np.linalg.norm(b) < 0.99 or np.linalg.norm(b) > 1.01:
        raise ValueError("b must be a unit vector")
    # Algorithm from https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    # The skew-symmetric cross product matrix of v
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) * -1
    # Rotation matrix as per Rodregues formula
    R = np.eye(3) - vx + np.dot(-vx, -vx) * ((1 - c) / (s**2))
    return R


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_from_xy(v1):
    v2 = [v1[0], v1[1], 0]
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_angles(tup, radians=False, reference = 'XY'):
    """Gets the angle of a vector with the XY axis"""
    if reference == 'XY':
        a = tup[0]
        b = tup[1]
        c = tup[2]
    if reference == 'XZ':
        a = tup[0]
        b = tup[2]
        c = tup[1]
    if reference == 'ZY':
        a = tup[1]
        b = tup[0]
        c = tup[2]
    denom = np.sqrt(a**2 + b**2)
    if denom != 0:
        radians = np.arctan(c / np.sqrt(a**2 + b**2))
        if radians:
            return radians
        else:
            return np.degrees(radians)
    else:
        return 0


def get_center(points, center_type="centroid"):
    """Attempts to find a representitiver 'center' given a
    set of 3D points.
    """
    if len(points[0]) != 3:
        breakpoint()
        print("not 3 points")
    if center_type == "centroid":
        # Average of each coordinate value
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        centroid = np.average(x), np.average(y), np.average(z)
        return centroid
    elif center_type == "top":
        # X and y values a straight average 
        # Z values average of the values in 90th %ile
        idxs, vals = get_percentile(points,90, 100)
        top_ten_pts = points[idxs]
        x = top_ten_pts[:, 0]
        y = top_ten_pts[:, 1]
        z = top_ten_pts[:, 2]
        centroid = np.average(x), np.average(y), np.average(z)
        return centroid
    elif center_type == "bottom":
        # X and y values a straight average 
        # Z values average of the values below the 10th %ile
        idxs, vals = get_percentile(points,0, 10)
        top_ten_pts = points[idxs]
        x = top_ten_pts[:, 0]
        y = top_ten_pts[:, 1]
        z = top_ten_pts[:, 2]
        centroid = np.average(x), np.average(y), np.average(z)
        return centroid

def get_radius(points, center_type="centroid"):
    """
    Given a set of 3D points, returns the average distance
    from a theoretical center (defined above)
    """
    center = get_center(points, center_type)
    xy_pts = points[:, :2]
    xy_center = center[:2]
    r = np.average([np.sqrt(np.sum((xy_pt - xy_center) ** 2)) for xy_pt in xy_pts])
    return r

def generate_grid(min_bnd,
                  max_bnd,
                    grid_xyz_cnt =  arr((2,3,1)),
                    overlap_ratio = 1/7
                  ):
    """Divides the region defined by the provide range 
        into a grid with a defined # of divisions of the x,y and z dimensions

    Args:
        min_bnd: the 'bottom-left' of the region (x,y,z)
        max_bnd:  the 'top-right' of the region (x,y,z)
        grid_xyz_cnt: the number of divisions desired in the x, y and z dimensions
    """
    col_lwh = max_bnd -min_bnd
    grid_xyz_cnt = arr((2,3,1)) # x,y,z
    grid_lines = [np.linspace(0,num,1) for num in grid_xyz_cnt]
    grid_lwh = col_lwh/grid_xyz_cnt
    ll_mults =  [np.linspace(0,num,num+1) for num in grid_xyz_cnt]
    llv = [minb + dim*mult for dim, mult,minb in zip(grid_lwh,ll_mults,min_bnd)]

    grid = arr([[[llv[0][0],llv[1][0]],[llv[0][1],llv[1][1]]],[[llv[0][1],llv[1][0]],[llv[0][2],llv[1][1]]],    
                [[llv[0][0],llv[1][1]],[llv[0][1],llv[1][2]]],[[llv[0][1],llv[1][1]],[llv[0][2],llv[1][2]]],    
                [[llv[0][0],llv[1][2]],[llv[0][1],llv[1][3]]],[[llv[0][1],llv[1][2]],[llv[0][2],llv[1][3]]]])
    ## We want a bit of overlap since nearby clusters sometime contest for points 
    overlap = grid_lwh*overlap_ratio    
    safe_grid = [[[ll[0]-overlap[0],ll[1]-overlap[1]],[ur[0]+overlap[0], ur[1]+overlap[1]]] for ll, ur in grid]
    return safe_grid