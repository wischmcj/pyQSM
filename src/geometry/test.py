from copy import deepcopy
import sys
sys.path.insert(0,'/code/code/pyQSM/src/')

import robust_laplacian
import numpy as np
from pc_skeletor import LBC
from tqdm import tqdm
import open3d as o3d
import rustworkx as rx
import networkx as nx
from scipy import sparse
from scipy.sparse import csr_matrix, diags, csgraph, vstack, linalg as sla
import mistree as mist
import matplotlib.colors as mcolors
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt

from viz.viz_utils import color_continuous_map
from set_config import config, log
from geometry.point_cloud_processing import clean_cloud, cluster_and_get_largest
from qsm_generation import filter_by_norm, get_low_cloud, sphere_step
from viz.viz_utils import draw
from utils.lib_integration import pts_to_cloud
from utils.math_utils import get_angles, get_percentile, poprow, rotation_matrix_from_arr, unit_vector, angle_from_xy




semantic_weight: float = 10
init_contraction: float = 3.
init_attraction: float = 0.6
max_contraction: int = 2048
max_attraction: int = 1024
step_wise_contraction_amplification: float | str = 'auto'
termination_ratio: float = 0.003
max_iteration_steps: int = 20
graph_k_n = 15


try: assert init_attraction != 0  
except AssertionError as err: log.warning(f'Improper Config:{err}')
try: assert init_contraction != 0 
except AssertionError as err: log.warning(f'Improper Config:{err}')
try: assert (init_contraction / (init_attraction + 1e-12)) < 1e6  
except AssertionError as err: log.warning(f'Improper Config:{err}')


def semantic_weighting( pcd_points, trunk_points, L, laplacian_weight_matrix):
        """
            Adds weights to the weak laplacian to allow discrimination
            between high and low magnitude (in terms of gradient) regions.
        """
        WL = laplacian_weight_matrix

        num_amplification_points = np.asarray(trunk_points).shape[0]
        multiplier = semantic_weight

        S = sparse.csc_matrix.copy(L)
        rows = S.nonzero()[0]
        cols = S.nonzero()[1]
        pos = np.vstack([rows, cols]).T
        connection_pos = np.where(pos[:, 1] > num_amplification_points)[0]
        connection_pos_idx = np.unique(pos[connection_pos][:, 0])
        mask = np.ones(pcd_points.shape[0], bool)
        mask[connection_pos_idx] = 0
        num_valid = np.arange(0, pcd_points.shape[0])[mask]
        S[rows, cols] = 1

        # ToDo: Speed up!
        for i in num_valid: S[i, L[i].nonzero()[1]] = multiplier

        WL_L = (WL @ L)
        WL_L = WL_L.multiply(S)
        
        return WL_L



def least_squares_sparse(pcd_points, 
                         L, 
                         laplacian_weighting, 
                         positional_weighting,
                         trunk_points=None):
    """
    Perform least squares sparse solving for the Laplacian-based contraction.
    Optionally allows weighting of the lapacian to discriminate between trunk and branch points
    Args:
        pcd_points: The input point cloud points.
        L: The Laplacian matrix.
        laplacian_weighting: The Laplacian weighting matrix.
        positional_weighting: The positional weighting matrix.

    Returns:
        The contracted point cloud.
    """
    # Define Weights
    WL = diags(laplacian_weighting)
    if trunk_points is not None:
        WL = semantic_weighting(pcd_points, trunk_points, L, WL)
    WH = diags(positional_weighting)

    A = vstack([L.dot(WL), WH]).tocsc()
    b = np.vstack([np.zeros((pcd_points.shape[0], 3)), WH.dot(pcd_points)])

    A_new = A.T @ A

    x = sla.spsolve(A_new, A.T @ b[:, 0], permc_spec='COLAMD')
    y = sla.spsolve(A_new, A.T @ b[:, 1], permc_spec='COLAMD')
    z = sla.spsolve(A_new, A.T @ b[:, 2], permc_spec='COLAMD')

    ret = np.vstack([x, y, z]).T

    if (np.isnan(ret)).all():
        log.warning('Matrix is exactly singular. Stopping Contraction.')
        ret = pcd_points
    return ret

def show_graph( graph: nx.Graph, 
            pos: np.ndarray | bool = True,
            fig_size: tuple = (20, 20)):
    # For more info: https://networkx.org/documentation/stable/reference/drawing.html
    plt.figure(figsize=fig_size)

    pos = [graph.nodes()[node_idx]['pos'] for node_idx in range(graph.number_of_nodes())]
    nx.draw_networkx(G=graph, pos=np.asarray(pos)[:, [0, 2]])
    # pos = [graph.nodes(node_idx)['pos'] for node_idx in graph.node_indicies()]
    # nx.draw_networkx(G=graph, pos=np.asarray(pos)[:, [0, 2]])

    plt.show()

def set_amplification(step_wise_contraction_amplification,
                        num_pcd_points):
    termination_ratio = 0.003
    # Set amplification factor of contraction weights.
    if isinstance(step_wise_contraction_amplification, str):
        if step_wise_contraction_amplification == 'auto':
            # num_pcd_points = num_pts.shape[0]
            print('Num points: ', num_pcd_points)

            if num_pcd_points < 1000:
                contraction_amplification = 1
                termination_ratio = 0.01
            elif num_pcd_points < 1e4:
                contraction_amplification = 2
                termination_ratio = 0.007
            elif num_pcd_points < 1e5:
                contraction_amplification = 5
                termination_ratio = 0.005
            elif num_pcd_points < 0.5 * 1e6:
                contraction_amplification = 5
                termination_ratio = 0.004
            elif num_pcd_points < 1e6:
                contraction_amplification = 5
                termination_ratio = 0.003
            else:
                contraction_amplification = 8
                termination_ratio = 0.0005

            contraction_factor = contraction_amplification
        else:
            raise ValueError('Value: {} Not found!'.format(step_wise_contraction_amplification))
    else:
        contraction_factor = step_wise_contraction_amplification
    return termination_ratio,contraction_factor

def show_graph( graph: nx.Graph, 
            pos: np.ndarray | bool = True,
            fig_size: tuple = (20, 20)):
    # For more info: https://networkx.org/documentation/stable/reference/drawing.html
    plt.figure(figsize=fig_size)

    pos = [graph.nodes()[node_idx]['pos'] for node_idx in range(graph.number_of_nodes())]
    nx.draw_networkx(G=graph, pos=np.asarray(pos)[:, [0, 2]])
    # pos = [graph.nodes(node_idx)['pos'] for node_idx in graph.node_indicies()]
    # nx.draw_networkx(G=graph, pos=np.asarray(pos)[:, [0, 2]])

    plt.show()
def extract_skeleton(pcd, 
                     trunk_points,
                     moll= 1e-5,
                     n_neighbors = 20,
                     max_iter= 20,
                     debug = False,
                     termination_ratio=None,
                     contraction_factor=None):
    max_iteration_steps = max_iter
    print('setting amplification')

    if not termination_ratio or not contraction_factor:
        termination_ratio_calc,contraction_factor_calc = set_amplification(
                                                                step_wise_contraction_amplification,
                                                                len(np.asarray(pcd.points)))
        if not termination_ratio:
            termination_ratio =termination_ratio_calc
        if not contraction_factor:
            contraction_factor= contraction_factor_calc

    pcd_points = np.asarray(pcd.points)

    print('generating laplacian')
    L, M = robust_laplacian.point_cloud_laplacian(pcd_points, 
                                                  mollify_factor=moll, 
                                                  n_neighbors=n_neighbors)
    # L - weak Laplacian
    # M - Mass (actually Area) matrix (along diagonal)
    # so M-1 * L is the strong Laplacian
    #
    M_list = [M.diagonal()]

    # Init weights
    positional_weights = init_attraction * np.ones(M.shape[0])
    laplacian_weights = (init_contraction * 10 ** 3 * np.sqrt(np.mean(M.diagonal())) * np.ones(M.shape[0]))

    iteration = 0
    volume_ratio = 1 # since areas array is len 1
    progress_bar = tqdm(total=max_iteration_steps)

    pcd_points_current = pcd_points
    total_point_shift = np.zeros_like(pcd_points_current)
    # we run this until the volumne of the previously added row
    #  becomes less than or equal to termination_ratio * the first row
    while (volume_ratio  > termination_ratio):
        msg = "Volume ratio: {}. Contraction weights: {}. Attraction weights: {}. Progress {}".format(
                volume_ratio, np.mean(laplacian_weights), np.mean(positional_weights), 'LBC')
        progress_bar.set_description(msg)
        print('Laplacian Weight: {}'.format(laplacian_weights))
        print('Mean Positional Weight: {}'.format(np.mean(positional_weights)))

        pcd_points_new = least_squares_sparse(pcd_points=pcd_points_current,
                                               L=L,
                                               laplacian_weighting=laplacian_weights,
                                               positional_weighting=positional_weights,
                                               trunk_points=trunk_points)

        if (pcd_points_new == pcd_points_current).all():
            print('No longer able to contract. Stopping Contraction.')
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
        # Update positional weights with the ration of the first Mass matrix and the current one.
        positional_weights = positional_weights * np.sqrt((M_list[0] / M.diagonal()))

        # Clip weights
        laplacian_weights = np.clip(laplacian_weights, 0.1, max_contraction)
        positional_weights = np.clip(positional_weights, 0.1, max_attraction)

        M_list.append(M.diagonal())

        iteration += 1
        progress_bar.update(1)

        try:
            L, M = robust_laplacian.point_cloud_laplacian(pcd_points_current,
                                                          mollify_factor=moll, 
                                                          n_neighbors=n_neighbors)
        except RuntimeError as er:
            print(f"Error finding laplacian on iteration {iteration}: {er}")
            break

        volume_ratio = np.mean(M_list[-1]) / np.mean(M_list[0])
        print(f"Completed iteration {iteration}")
        if iteration >= max_iteration_steps:
            break

    print(f'Contraction is Done after {iteration} iters')

    contracted = pts_to_cloud(pcd_points_current)

    return contracted, total_point_shift

def create_graph(skeletal_points):
    from itertools import product
    from math import dist
    pts = skeletal_points
    graph = rx.generators.complete_graph(len(pts))
    graph.add_nodes_from(pts)
    conn_graph = rx.complement(graph)
    graph.add_edges_from([(u,v,None) for u,v in  product(graph.nodes(),graph.nodes()) ])
    tree = rx.minimum_spanning_edges(graph, weight_fn=lambda u,v: dist(u,v))


def extract_skeletal_graph(skeletal_points: np.ndarray):
    np.bool = bool
    test=True
    points = skeletal_points    
    _, edge_x, edge_y, edge_z, edge_index = mist.construct_mst(x=points[:, 0], y=points[:, 1], z=points[:, 2],k_neighbours=graph_k_n)

    # degree, edge_length, branch_length, branch_shape, edge_index, branch_index = mst.get_stats(
            # include_index=True, k_neighbours=graph_k_n)
    mst_graph = nx.Graph(edge_index.T.tolist())
    for idx in range(mst_graph.number_of_nodes()): 
        mst_graph.nodes[idx]['pos'] = skeletal_points[idx].T
    
    edge_diff = [ (x,y,(x1-x2,y1-y2,z1-z2)) for (x,y),(x1,x2),(y1,y2),(z1,z2) in zip(edge_index.T.tolist(),edge_x.T.tolist(),edge_y.T.tolist(),edge_z.T.tolist())]
    rust_graph = rx.PyGraph()
    for idx in range(mst_graph.number_of_nodes()): 
        rust_graph.add_node({'pos':skeletal_points[idx].T})
    rust_graph.add_nodes_from(range(len(points)))
    rust_graph.add_edges_from(edge_diff)
    return mst_graph , rust_graph


def simplify_graph(G):
    """
    The simplifyGraph function simplifies a given graph by removing nodes of degree 2 and fusing their incident edges.
    Source:  https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges

    :param G: A NetworkX graph object to be simplified
    :return: A tuple consisting of the simplified NetworkX graph object, a list of positions of kept nodes, and a list of indices of kept nodes.
    """

    g = G.copy()

    while any(degree == 2 for _, degree in g.degree):

        keept_node_pos = []
        keept_node_idx = []
        g0 = g.copy()  # <- simply changing g itself would cause error `dictionary changed size during iteration`
        for node, degree in g.degree():
            if degree == 2:

                if g.is_directed():  # <-for directed graphs
                    a0, b0 = list(g0.in_edges(node))[0]
                    a1, b1 = list(g0.out_edges(node))[0]

                else:
                    edges = g0.edges(node)
                    edges = list(edges.__iter__())
                    a0, b0 = edges[0]
                    a1, b1 = edges[1]

                e0 = a0 if a0 != node else b0
                e1 = a1 if a1 != node else b1

                g0.remove_node(node)
                g0.add_edge(e0, e1)
            else:
                keept_node_pos.append(g.nodes[node]['pos'])
                keept_node_idx.append(node)
        g = g0

    return g, keept_node_pos, keept_node_idx

def simplify_and_update(graph):
    G_simplified, node_pos, _ = simplify_graph(graph)
    skeleton_cleaned = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(node_pos)))
    skeleton_cleaned.paint_uniform_color([0, 0, 1])
    skeleton_cleaned_points = np.asarray(skeleton_cleaned.points)
    mapping = {}
    for node in G_simplified:
        pcd_idx = np.where(skeleton_cleaned_points == G_simplified.nodes[node]['pos'])[0][0]
        mapping.update({node: pcd_idx})
    return nx.relabel_nodes(G_simplified, mapping), skeleton_cleaned_points

def extract_topology(contracted):
    contracted_zero_artifact = deepcopy(contracted)

    # Artifacts at zero
    pcd_contracted_tree = o3d.geometry.KDTreeFlann(contracted)
    idx_near_zero = np.argmin(np.linalg.norm(np.asarray(contracted_zero_artifact.points), axis=1))
    min_norm =  np.linalg.norm(contracted_zero_artifact.points[idx_near_zero])
    if min_norm<= 0.01:
        [k, idx, _] = pcd_contracted_tree.search_radius_vector_3d(contracted_zero_artifact.points[idx_near_zero], 0.01)
        contracted = contracted_zero_artifact.select_by_index(idx, invert=True)
    contracted_pts = np.asarray(contracted.points)

    # Compute points for farthest point sampling
    fps_points = int(contracted_pts.shape[0] * 0.1)
    fps_points = max(fps_points, 15)

    # Sample with farthest point sampling
    skeleton = contracted.farthest_point_down_sample(num_samples=fps_points)
    skeleton_points = np.asarray(skeleton.points)

    if (np.isnan(contracted_pts)).all():
        print('Element is NaN!')

    skeleton_graph, rx_graph = extract_skeletal_graph(skeletal_points=skeleton_points)
    topology_graph, topology_points = simplify_and_update(graph=skeleton_graph)

    topology = o3d.geometry.LineSet()
    topology.points = o3d.utility.Vector3dVector(topology_points)
    topology.lines = o3d.utility.Vector2iVector(list((topology_graph.edges())))

    return topology, topology_graph, skeleton, skeleton_points, skeleton_graph,rx_graph



if __name__=="__main__":
    from qsm_generation import config
    from point_cloud_processing import get_sub_bounding_sphere
    skeletor = "skeletor.pts"

    ## Running 27
    # pcd = o3d.io.read_point_cloud("data/input/27_vox_pt02_sta_6-4-3.pcd")
    # pcd = clean_cloud(pcd, voxels=     0.08,  neighbors=  None, ratio=      4, iters=      3)
    # pcd =  o3d.io.read_point_cloud(skeletor,"xyz")
    # pcd = clean_cloud(pcd, voxels=config['initial_clean']['voxel_size'],
    #                           neighbors=config['initial_clean']['neighbors'], 
    #                           ratio=config['initial_clean']['ratio'], 
    #                           iters = config['initial_clean']['iters'])
    # o3d.io.write_point_cloud("skeletor_trunk_4-9.pcd",pcd)
   
    # stat_down = get_low_cloud(pcd)
    # stat_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    # stat_down.normalize_normals()
    # stem_cloud = filter_by_norm(stat_down,40)
    # draw([stem_cloud])

    # Try diff minimal contraction configs
    pcd = o3d.io.read_point_cloud("skeletor_super_clean_contracted.pcd")
    stem_cloud = pcd.uniform_down_sample(10)
    moll,iter_num,contract_fact = [1, 1e-2,1e-4,1e-5],[1,2,3,4,5],[1,3,5,8]
    var_cases = [moll,iter_num,contract_fact]
    run_cases(stem_cloud,var_cases,extract_stuctures)


    pcd = o3d.io.read_point_cloud("skeletor_trunk.pcd",'xyz')
    # pcd.voxel_down_sample(voxel_size=0.04)
    pcd = clean_cloud(pcd, voxels=config['initial_clean']['voxel_size'],
                              neighbors=config['initial_clean']['neighbors'], 
                              ratio=config['initial_clean']['ratio'], 
                              iters = config['initial_clean']['iters'])

    stat_down, idxs = get_low_cloud(pcd,3,9)

    max_cluster = cluster_and_get_largest(stat_down)
    max_cluster = o3d.io.read_point_cloud("skeletor_trunk.pcd")
    draw(max_cluster)

    import itertools
    cases = [(5,'ZY'),(10,'ZY'),(15,'ZY'),(20,'ZY'),
                (5,'XZ'),(10,'XZ'),(15,'XZ'),(20,'XZ'),]
    for case in cases:
        angle_thresh, ref = case
        pcd = deepcopy(max_cluster)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=.1, max_nn=10))
        norms = np.asarray(pcd.normals)
        # Modified filter by norm
        ## Coloring by norm angle
        angles = [get_angles(x,reference = ref) for x in norms]
        angles = np.degrees(angles)
        print("Mean:", np.mean(angles))
        print("Median:", np.median(angles))
        print("Standard Deviation:", np.std(angles))
        color_continuous_map(pcd,angles)
        draw(pcd)
        # The angles on the side of each ridge are closer to 0 with reference to the XZ plane
        stem_idxs = np.where((angles > -angle_thresh) & (angles < angle_thresh))[0]
        stem_cloud = pcd.select_by_index(stem_idxs)
        draw(stem_cloud)
        clean_stem = clean_cloud(stem_cloud, voxels=0.08,  neighbors=  7, ratio=      2, iters=      3)
        draw([clean_stem])  

        ## clustering
        labels = np.array(clean_stem.cluster_dbscan(eps=.2,min_points=  15,print_progress=True))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        clean_stem.colors = o3d.utility.Vector3dVector(colors[:, :3])
        unique_vals, counts = np.unique(labels, return_counts=True)
        print(len(unique_vals))
        draw([clean_stem])

        largest = unique_vals[np.argmax(counts)]
        max_cluster_idxs = np.where(labels == largest)[0]
        max_cluster = stat_down.select_by_index(max_cluster_idxs)
        draw(max_cluster)

        breakpoint()  
    import itertools

    # contracted = o3d.io.read_point_cloud("skeletor_super_clean_contracted.pcd")
    # not_so_low_idxs, _ = get_percentile(np.asarray(contracted.points), 0, 90, axis =0)
    # test = contracted.select_by_index(not_so_low_idxs)
    # o3d.visualization.draw_geometries([test])

    # breakpoint()

    # breakpoint()
    # topology, topology_graph, skeleton, skeleton_points, skeleton_graph,rx_graph = extract_topology(contracted)
    # draw([skeleton])
    # breakpoint()