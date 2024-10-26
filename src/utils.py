import numpy as np


def get_percentile(pts,low,high):
    """
        Returns the indices of the points that fall within the
        low and high percentiles of the z values of the points
    """
    z_vals = pts[:,2]
    lower  = np.percentile(z_vals, low)
    upper  = np.percentile(z_vals, high)
    all_idxs =  np.where(z_vals)
    too_low_idxs = np.where(z_vals<=lower)
    too_high_idxs = np.where(z_vals>=upper)
    not_too_low_idxs = np.setdiff1d(all_idxs ,too_low_idxs)
    select_idxs = np.setdiff1d(not_too_low_idxs ,too_high_idxs)
    vals =  z_vals[select_idxs]
    return select_idxs, vals 

def poprow(my_array,pr):
    """ Row popping in numpy arrays
    Input: my_array - NumPy array, pr: row index to pop out
    Output: [new_array,popped_row] """
    i = pr
    pop = my_array[i]
    new_array = np.vstack((my_array[:i],my_array[i+1:]))
    return new_array,pop

def rotation_matrix_from_arr(a, b: np.array):
    """
    Returns matrix R such that a*R = b. 
    """
    if np.linalg.norm(b) == 0:
        return np.eye(3)
    if np.linalg.norm(b) <.99 or np.linalg.norm(b) > 1.01:
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
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)

def angle_from_xy(v1):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v2 = [v1[0],v1[1],0]
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_angles(tup,radians=False):
    """Gets the angle of a vector with the XY axis"""
    a=tup[0]
    b=tup[1]
    c=tup[2]
    denom = np.sqrt(a**2 +b**2)
    if denom !=0:
        radians = np.arctan(c/np.sqrt(a**2 + b**2))
        if radians:
            return radians
        else:
            return np.degrees(radians)
    else:
        return 0

def get_center(points, center_type = 'centroid'):
    """Attempts to find a representitiver 'center' given a 
        set of 3D points. 
    """
    if len(points[0]) !=3:
        breakpoint()
        print('not 3 points')
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    if center_type == 'centroid':
        centroid = np.average(x), np.average(y), np.average(z)
        return centroid
    if center_type == 'middle':
        middle = middle(x), middle(y), middle(z)
        return middle

def get_radius(points, center_type = 'centroid'):
    """
        Given a set of 3D points, returns the average distance
        from a theoretical center (defined above)
    """
    center = get_center(points, center_type)
    xy_pts = points[:,:2]
    xy_center = center[:2]
    r = np.average([np.sqrt(np.sum((xy_pt - xy_center)**2)) for xy_pt in xy_pts])
    return r
