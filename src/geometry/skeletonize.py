import robust_laplacian
from plyfile import PlyData
import numpy as np
import polyscope as ps
from scipy.sparse import csr_matrix, diags, csgraph, vstack, linalg as sla
import open3d as o3d

from set_config import config, log
from viz import draw, pts_to_cloud



config['skeletonize']['moll']
config['skeletonize']['n_neighbors'] 
config['skeletonize']['max_iter']
config['skeletonize']['semantic_weight'] 
config['skeletonize']['init_contraction']
config['skeletonize']['init_attraction']
config['skeletonize']['max_contraction'] 
config['skeletonize']['max_attraction']
config['skeletonize']['termination_ratio']
config['skeletonize']['max_iteration_steps']
config['skeletonize']['graph_k_n']


def least_squares_sparse(pcd_points, 
                         L, 
                         laplacian_weighting, 
                         positional_weighting,
                         trunk_points=None):
    """
    Perform least squares sparse solving for the Laplacian-based contraction.
    """
    # Define Weights
    WL = diags(laplacian_weighting)
    WH = diags(positional_weighting)

    A = vstack([L.dot(WL), WH]).tocsc()
    b = np.vstack([np.zeros((pcd_points.shape[0], 3)), WH.dot(pcd_points)])

    A_new = A.T @ A

    x = sla.spsolve(A_new, A.T @ b[:, 0], permc_spec='COLAMD')
    y = sla.spsolve(A_new, A.T @ b[:, 1], permc_spec='COLAMD')
    z = sla.spsolve(A_new, A.T @ b[:, 2], permc_spec='COLAMD')

    ret = np.vstack([x, y, z]).T

    if (np.isnan(ret)).all():
        log.warning('No points in new matrix ')
        ret = pcd_points
    return ret

def extract_skeleton(pcd, 
                     moll= config[]['moll'],
                     n_neighbors = config['skeletonize']['n_neighbors'],
                     max_iter= config['skeletonize']['max_iter'],
                     debug = False,
                     termination_ratio=config['skeletonize']['termination_ratio'],
                     contraction_factor=config['skeletonize']['init_contraction'],
                     attraction_factor= config['skeletonize']['init_attraction'],
                     max_contraction = config['skeletonize']['max_contraction'],
                     max_attraction = config['skeletonize']['max_attraction']):
    max_iteration_steps = max_iter

    pcd_points = np.asarray(pcd.points)

    print('generating laplacian')
    L, M = robust_laplacian.point_cloud_laplacian(pcd_points, 
                                                  mollify_factor=moll, 
                                                  n_neighbors=n_neighbors)
    # L - weak Laplacian
    # M - Mass (actually Area) matrix (along diagonal)
    # so M-1 * L is the strong Laplacian
    M_list = [M.diagonal()]

    # Init weights
    positional_weights = attraction_factor * np.ones(M.shape[0])
    laplacian_weights = (contraction_factor * 10 ** 3 * np.sqrt(np.mean(M.diagonal())) * np.ones(M.shape[0]))

    iteration = 0
    volume_ratio = 1 # since areas array is len 1

    pcd_points_current = pcd_points
    total_point_shift = np.zeros_like(pcd_points_current)
    # we run this until the volumne of the previously added row
    #  becomes less than or equal to termination_ratio * the first row
    while (volume_ratio  > termination_ratio):
        log.info(f'{volume_ratio=}, {np.mean(laplacian_weights)=}, {np.mean(positional_weights)=}')

        pcd_points_new = least_squares_sparse(pcd_points=pcd_points_current,
                                               L=L,
                                               laplacian_weighting=laplacian_weights,
                                               positional_weighting=positional_weights)

        if (pcd_points_new == pcd_points_current).all():
            print('No more contraction in last iter, ending run.')
            break
        else:
            pcd_point_shift = pcd_points_current-pcd_points_new
            total_point_shift += pcd_point_shift
            pcd_points_current = pcd_points_new

        if debug:
            breakpoint()
            draw([pts_to_cloud(pcd_points_current)])
            

        # Update laplacian weights with amplification factor
        laplacian_weights *= contraction_factor
        # Update positional weights with the ratio of the first Mass matrix and the current one.
        positional_weights = positional_weights * np.sqrt((M_list[0] / M.diagonal()))

        # Clip weights
        laplacian_weights = np.clip(laplacian_weights, 0.1, max_contraction)
        positional_weights = np.clip(positional_weights, 0.1, max_attraction)

        M_list.append(M.diagonal())

        iteration += 1

        L, M = robust_laplacian.point_cloud_laplacian(pcd_points_current,
                                                          mollify_factor=moll, 
                                                          n_neighbors=n_neighbors)

        volume_ratio = np.mean(M_list[-1]) / np.mean(M_list[0])
        print(f"Completed iteration {iteration}")
        if iteration >= max_iteration_steps:
            break

    print(f'Finished after {iteration} iterations')

    contracted = pts_to_cloud(pcd_points_current)

    return contracted, total_point_shift

if __name__ == "__main__":
    trunk = o3d.io.read_point_cloud("skeletor_trunk.pcd")
    contracted, total_point_shift = extract_skeleton(trunk)

# max_cluster = o3d.io.read_point_cloud("skeletor_trunk.pcd")
# points = np.asarray(max_cluster.points)
# # Build point cloud Laplacian
# L, M = robust_laplacian.point_cloud_laplacian(points,1e-10)

# # (or for a mesh)
# # L, M = robust_laplacian.mesh_laplacian(verts, faces)

# # Compute some eigenvectors
# n_eig = 10
# evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

# # Visualize
# ps.init()
# ps_cloud = ps.register_point_cloud("my cloud", points)
# for i in range(n_eig): 
#     ps_cloud.add_scalar_quantity("eigenvector_"+str(i), evecs[:,i], enabled=True)
# breakpoint()
# ps.show()


