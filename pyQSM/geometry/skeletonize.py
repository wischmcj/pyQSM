
import sys
from collections import defaultdict
from itertools import chain
import scipy.spatial as sps
import pickle

sys.path.insert(0,'/code/pyQSM/src/')
import robust_laplacian
from plyfile import PlyData
import numpy as np
from numpy import asarray as arr
import polyscope as ps
from scipy.sparse import csr_matrix, diags, csgraph, vstack, linalg as sla,eye
import open3d as o3d
from skspatial.objects import Cylinder

from copy import deepcopy, copy
import mistree as mist
from matplotlib import pyplot as plt
import rustworkx as rx
import networkx as nx



# from tree_isolation import recover_original_detail, zoom, extend_seed_clusters,create_one_or_many_pcds
from geometry.point_cloud_processing import clean_cloud,cluster_plus,crop_by_percentile,get_ball_mesh
from geometry.mesh_processing import map_density,get_surface_clusters 
from set_config import config, log
from viz.viz_utils import draw, color_continuous_map 
from utils.lib_integration import pts_to_cloud,get_neighbors_in_tree
from utils.math_utils import get_center,rot_90_x,unit_vector, get_percentile
from utils.io import save,load


def extract_skeletal_graph(skeletal_points: np.ndarray, graph_k_n):
    np.bool = bool
    test=True
    points = skeletal_points    
    _, edge_x, edge_y, edge_z, edge_index = mist.construct_mst(x=points[:, 0], y=points[:, 1], z=points[:, 2],k_neighbours=graph_k_n)

    # degree, edge_length, branch_length, branch_shape, edge_index, branch_index = mst.get_stats(
            # include_index=True, k_neighbours=graph_k_n)
    mst_graph = nx.Graph(edge_index.T.tolist())
    for idx in range(mst_graph.number_of_nodes()): 
        mst_graph.nodes[idx]['pos'] = skeletal_points[idx].T
    
    # add the total shift 
    edge_diff = [ (x,y,(x1-x2,y1-y2,z1-z2)) for (x,y),(x1,x2),(y1,y2),(z1,z2) in zip(edge_index.T.tolist(),edge_x.T.tolist(),edge_y.T.tolist(),edge_z.T.tolist())]
    rust_graph = rx.PyGraph()
    for idx in range(mst_graph.number_of_nodes()): 
        rust_graph.add_node({'pos':skeletal_points[idx].T})
    rust_graph.add_nodes_from(range(len(points)))
    rust_graph.add_edges_from(edge_diff)
    return mst_graph , rust_graph

def simplify_graph(G):
    """
    The simplifyGraph function simplifies a given graph 
    by removing nodes of degree 2 and fusing their incident edges.
    Source:  https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges

    :param G: A NetworkX graph object to be simplified
    :return: A tuple consisting of the simplified NetworkX graph object, a list of positions of kept nodes, and a list of indices of kept nodes.
    """

    g = G.copy()
    keept_node_pos = []
    keept_node_idx = []
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
                    edges = g0.edges(node,data=True)
                    edges = list(edges.__iter__())
                    a0, b0, data0 = edges[0]
                    a1, b1, data1 = edges[1]

                e0 = a0 if a0 != node else b0
                e1 = a1 if a1 != node else b1
                edata = data0.get('data',[]) +  data1.get('data',[]) 
                edata.append(node)

                g0.remove_node(node)
                g0.add_edge(e0, e1, data = edata)
            else:
                keept_node_pos.append(g.nodes[node]['pos'])
                keept_node_idx.append(node)
        g = g0

    return g, keept_node_pos, keept_node_idx

def simplify_and_update(graph):
    """Simplifies graph (see corresponding doc string) and relabels
      graphs nodes to have the id of the corresponding point cloud index."""
    G_simplified, node_pos, _ = simplify_graph(graph)
    skeleton_cleaned = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(node_pos)))
    skeleton_cleaned.paint_uniform_color([0, 0, 1])
    skeleton_cleaned_points = np.asarray(skeleton_cleaned.points)
    mapping = {}
    for node in G_simplified:
        pcd_idx = np.where(skeleton_cleaned_points == G_simplified.nodes[node]['pos'])[0][0]
        mapping.update({node: pcd_idx})
    return nx.relabel_nodes(G_simplified, mapping), skeleton_cleaned_points, mapping

def extract_topology(contracted, graph_k_n = config['skeletonize']['graph_k_n']):
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
    fps_points = int(contracted_pts.shape[0] * 0.1) # reduce to 30%
    fps_points = max(fps_points, 15)   # Dont remove any more than 15 points

    log.info(f'down sampling contracted, starting with {contracted}, taking {fps_points} samples')
    # Sample with farthest point sampling
    skeleton = contracted.farthest_point_down_sample(num_samples=fps_points)
    skeleton_points = np.asarray(skeleton.points)
    log.info(f'Down Sample, the skeleton, has {skeleton_points.shape[0]} points')

    if (np.isnan(contracted_pts)).all():
        log.info('Element is NaN!')

    skeleton_graph, rx_graph = extract_skeletal_graph(skeletal_points=skeleton_points,graph_k_n=graph_k_n)
    topology_graph, topology_points, mapping = simplify_and_update(graph=skeleton_graph)

    topology = o3d.geometry.LineSet()
    topology.points = o3d.utility.Vector3dVector(topology_points)
    topology.lines = o3d.utility.Vector2iVector(list((topology_graph.edges())))

    return topology, topology_graph, skeleton, skeleton_points, skeleton_graph,rx_graph, mapping

def least_squares_sparse(pts, 
                         L, 
                         laplacian_weighting, 
                         positional_weighting,
                         trunk_points=None):
    """
    Perform least squares sparse solving for the Laplacian-based contraction.
    """
    print('start least squares')
    # Define Weights
    # I = eye(pts.shape[0])
    # WL = I * laplacian_weighting
    WL = diags(laplacian_weighting)
    WH = diags(positional_weighting)
    print('start lin alg')

    A = vstack([L.dot(WL), WH]).tocsc()
    b = np.vstack([np.zeros((pts.shape[0], 3)), WH.dot(pts)])

    A_new = A.T @ A
    log.info('defined least squares equation, solving 1...')
    x = sla.spsolve(A_new, A.T @ b[:, 0], permc_spec='COLAMD')
    log.info('solving 2...')
    y = sla.spsolve(A_new, A.T @ b[:, 1], permc_spec='COLAMD')
    log.info('solving 3...')
    z = sla.spsolve(A_new, A.T @ b[:, 2], permc_spec='COLAMD')

    ret = np.vstack([x, y, z]).T

    if (np.isnan(ret)).all():
        log.warning('No points in new matrix ')
        ret = pts
    return ret

def set_amplification(step_wise_contraction_amplification,
                        num_pcd_points,
                        termination_ratio):
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
                # termination_ratio = 0.005
                termination_ratio = 0.003
            elif num_pcd_points < 0.5 * 1e6:
                contraction_amplification = 5
                termination_ratio = 0.004
            else:
                contraction_amplification = 5
                termination_ratio = 0.003
            # else num_pcd_points < 1e6:
            #     contraction_amplification = 5
            #     termination_ratio = 0.003
            # ### was having issues with the below, creating equation
            # ### with no points in solution matrix
            # else:
            #     contraction_amplification = 8
            #     termination_ratio = 0.001

            contraction_factor = contraction_amplification
        else:
            raise ValueError('Value: {} Not found!'.format(step_wise_contraction_amplification))
    else:
        contraction_factor = step_wise_contraction_amplification
    log.info(f'contraction_factor set to {contraction_factor} based on number of points ')
    log.info(f'termination_ratio set to {termination_ratio} based on number of points ')
    return termination_ratio,contraction_factor


def extract_skeleton(pcd, 
                     moll= config['skeletonize']['moll'],
                     n_neighbors = config['skeletonize']['n_neighbors'],
                     max_iter= config['skeletonize']['max_iter'],
                     debug = False,
                     termination_ratio=config['skeletonize']['termination_ratio'],
                     contraction_factor=config['skeletonize']['init_contraction'],
                     attraction_factor= config['skeletonize']['init_attraction'],
                     max_contraction = config['skeletonize']['max_contraction'],
                     max_attraction = config['skeletonize']['max_attraction'],
                     step_wise_contraction_amplification = config['skeletonize']['step_wise_contraction_amplification'],
                     cmag_save_file = '',
                     min_contraction = 0):
    # Hevily inspired by https://github.com/meyerls/pc-skeletor
    obb = pcd.get_oriented_bounding_box()
    allowed_range = (obb.get_min_bound(), obb.get_max_bound())
    
    # termination_ratio,_ = set_amplification(step_wise_contraction_amplification,
    #                                                           len(np.asarray(pcd.points)),termination_ratio)
    # termination_ratio,contraction_factor = set_amplification(step_wise_contraction_amplification,
                                                                # len(np.asarray(pcd.points)),termination_ratio)
    
    max_iteration_steps = max_iter

    pts = np.asarray(pcd.points)

    log.info('generating laplacian')
    L, M = robust_laplacian.point_cloud_laplacian(pts, 
                                                  mollify_factor=moll, 
                                                  n_neighbors=n_neighbors)
    # L - weak Laplacian
    # M - Mass (actually Area) matrix (along diagonal)
    # so M-1 * L is the strong Laplacian
    M_list = [M.diagonal()]

    # Init weights
    positional_weights = attraction_factor * np.ones(M.shape[0]) # WH
    laplacian_weights = (contraction_factor * np.sqrt(np.mean(M.diagonal()))) * np.ones(M.shape[0]) # WL
    # (contraction_factor * np.sqrt(np.mean(M.diagonal())) 
    #                      * np.ones(M.shape[0])) 
    # Init weights, weighted by the mass matrix
    iteration = 0
    volume_ratio = 1 # since areas array is len 1

    pts_current = pts
    shift_by_step = []
    total_point_shift = np.zeros_like(pts_current)

    # we run this until the volume of the previously added row
    #  becomes less than or equal to termination_ratio * the first row
    while (volume_ratio  > termination_ratio):
        log.info(f'{volume_ratio=}, {np.mean(laplacian_weights)=}, {np.mean(positional_weights)=}')
        print('running least_squares_sparse')
        pts_new = least_squares_sparse(pts=pts_current,
                                               L=L,
                                               laplacian_weighting=laplacian_weights,
                                               positional_weighting=positional_weights)

        if (pts_new == pts_current).all():
            log.info('No more contraction in last iter, ending run.')
            break
        else:
            for point in pts_new:
                for i in range(3):
                    if point[i] < allowed_range[0][i]:
                        point[i] = allowed_range[0][i]
                    if point[i] > allowed_range[1][i]:
                        point[i] = allowed_range[1][i]
            # for curr,new in zip(pts_new_raw,pts_current):
            #     diff = curr-new
            #     dist = np.linalg.norm(diff)
            #     if dist >= min_contraction:
            #         pts_new.append(new)
            #     else:
            #         pts_new.append(curr)
            pcd_point_shift = pts_current-pts_new
            total_point_shift += pcd_point_shift
            pts_current = pts_new
            shift_by_step.append(pcd_point_shift)

        print('running debug')
        if debug or iteration ==0:
            try:
                print('saving cmag')
                # c_mag = np.array([np.linalg.norm(x) for x in pcd_point_shift])
                save(f'{cmag_save_file}_shift.pkl',shift_by_step)
                # curr_pts_pcd = pts_to_cloud(pts_current)
                # o3d.write_point_cloud(f'{cmag_save_file}_contracted.pcd',curr_pts_pcd)
            except FileNotFoundError as e:
                print(f'error in cmag, file not found: {e.filename}')
                import pickle
                with open(f'{cmag_save_file}_shift.pkl', 'wb') as f:
                    pickle.dump(shift_by_step, f)
            except Exception as e:
                print(f'error in cmag saving: {e}')
            # breakpoint()

        # Update laplacian weights with amplification factor
        laplacian_weights *= contraction_factor
        # Update positional weights with the ratio of the first Mass matrix and the current one.
        positional_weights = positional_weights * np.sqrt((M_list[0] / M.diagonal()))

        # Clip weights
        laplacian_weights = np.clip(laplacian_weights, 0.1, max_contraction)
        positional_weights = np.clip(positional_weights, 0.1, max_attraction)

        M_list.append(M.diagonal())

        iteration += 1

        L, M = robust_laplacian.point_cloud_laplacian(pts_current,
                                                          mollify_factor=moll, 
                                                          n_neighbors=n_neighbors)

        if debug:
            contracted = pts_to_cloud(pts_current)
            draw([contracted])

        volume_ratio = np.mean(M_list[-1]) / np.mean(M_list[0])
        log.info(f"Completed iteration {iteration}")

        # Checking to see if we've reached the end
        if iteration >= max_iteration_steps:
            try:
                save(f'{cmag_save_file}_tpshift.pkl',shift_by_step)
            except Exception as e:
                print(f'error when saving total shift {e}.')
            break
        if volume_ratio < termination_ratio:
            try:
                save(f'{cmag_save_file}_tpshift.pkl',shift_by_step)
            except Exception as e:
                print(f'error when saving total shift {e}.')

    log.info(f'Finished after {iteration} iterations')

    contracted = pts_to_cloud(pts_current)

    return contracted, total_point_shift, shift_by_step

def skeleton_to_QSM(topology,topology_graph, total_point_shift, test = True):
    """
        Takes topology graph, whose edge data contains the points in the original
          point cloud that map to each edge. This is used to get the average contraction made 
          to get each point to the edge. 
          The average contraction is used as the radius of a cylinder representing the given 
            section of the tree. 

        Returns:
            all_cyl_pcd: PointCloud of all cylinders
            cyls: List of Cylinder objects
            cyl_objects: List of Cylinder objects
            radii: List of radii of cylinders
    """
    edges = topology_graph.edges(data=True)
    edge_to_orig = {tuple((x[0],x[1])):x[2].get('data') for x in edges}
    points = np.asarray(topology.points)
    lines = np.asarray(topology.lines)
    contraction_dist = np.linalg.norm(total_point_shift, axis=1)
    cyls = []
    cyl_objects = []
    radii = []
    for idx, line in enumerate(lines):
        try:
            start = points[line[0]]
            end = points[line[1]]

            orig_verticies = edge_to_orig[tuple(line)] 
            contraction_dists = contraction_dist[orig_verticies]
            
            cyl_radius= np.mean(contraction_dists)
            cyl  = Cylinder.from_points(start,end,cyl_radius)
            cyl_pts = cyl.to_points(n_angles=20).round(3).unique()
            cyl_pcd = o3d.geometry.PointCloud()
            cyl_pcd.points = o3d.utility.Vector3dVector(cyl_pts)
            cyls.append(cyl_pcd)
            cyl_objects.append(cyl)
            radii.append(cyl_radius)
            
            # if test:
            #     breakpoint()
            #     print('test')
            if idx %10 == 0:    
                # draw(cyls)
                print(f'finished iteration {idx}')
        except Exception as e:
            print(f'error in skeleton_to_QSM: {e}')
            breakpoint()
        
        # for idp, point in enumerate(pcd_points): 
        #     if cyl.is_point_within(point):
        #         pcd_pts_contained.append(point)
        #         pcd_points.pop(idp)
        # if idx %10 == 0:
        #     log.info(f'finished iteration {idx}')
        #     draw(cyls)
        #     breakpoint()
        #     print('checkin')
    # contained_pcd = o3d.geometry.PointCloud()
    # contained_pcd.points = o3d.utility.Vector3dVector(pcd_pts_contained)
    all_cyl_pcd= o3d.geometry.PointCloud()
    pts = []
    for cyl in cyls: 
        pts.extend(np.array(cyl.points))
    all_cyl_pcd.points = o3d.utility.Vector3dVector(pts)
    draw(cyls)
    return all_cyl_pcd, cyls, cyl_objects, radii

    # breakpoint()
    # gif_center = get_center(np.asarray(pcds[0].points),center_type = "bottom")
    # animate_contracted_pcd( pcd,trunk.voxel_down_sample(.05),  point_size=3, rot_center = gif_center, steps=360, save = True, file_name = '_proto_qsm_trunk',transient_period = 30)
    # animate_contracted_pcd( pcds[0],topology,  point_size=3, rot_center = gif_center, steps=360, save = False, file_name = 'test',transient_period = 30)
    
    return all_cyl_pcd, cyls, cyl_objects , radii
#                     edge_to_orig,pcd,
#                     contraction_dist):
#     from skspatial.objects import Cylinder
#     start = points[line[0]]
#     end = points[line[1]]
#     vector = end-start
#     uvector = unit_vector(vector)    

#     orig_verticies = edge_to_orig[tuple(line)] 
#     contraction_dists = contraction_dist[orig_verticies]

#     cyl_radius= np.mean(contraction_dists)

#     cyl  = Cylinder.from_points(start,end,cyl_radius)
#     cyl_pts = cyl.to_points(n_angles=30).round(3).unique()


#     start_idx = contracted_to_orig[line[0]]
#     end_idx = contracted_to_orig[line[1]]


#     orig_node_graph = nx.simple_paths(orig_graph,start_idx,end_idx)

#     pts = np.asarray(pcd.points)
#     prec_start = pts[start_idx]
#     prec_end = pts[start_idx]

def remove_leaves():
    trunk = o3d.io.read_point_cloud("skeletor_stem20.pcd")
    # trunk = trunk.uniform_down_sample(4)
    leaves = o3d.io.read_point_cloud("sig_contracted_stem20_default_leaves.pcd")
    btm_half = crop_by_percentile(leaves,0, 13)
    top_half = crop_by_percentile(leaves, 14,100)
    # o3d.io.write_point_cloud("skel_leaves_top60cont_stem20_default.pcd", top_half)
    top_half_pts = np.asarray(top_half[0].points)

    import scipy.spatial as sps 
    full_tree =sps.KDTree(trunk)
    trunk_tree = sps.KDTree(top_half_pts)
    # nbr_idxs = get_neighbors_in_tree(top_half_pts, full_tree,1)
    pairs = trunk_tree.query_ball_tree(full_tree, r=.1)
    neighbors = np.array(list(set(chain.from_iterable(pairs))))
    leaves_in_trunk = trunk.select_by_index(neighbors)
    draw(leaves_in_trunk)

    non_leaves = trunk.select_by_index(neighbors,invert=True)
    leaves_in_trunk.paint_uniform_color([1,0,0])
    draw([non_leaves,leaves_in_trunk])
        # o3d.io.write_point_cloud("skel_leaves_nbrs_.5rad.pcd", leaves_in_trunk)

if __name__ == "__main__":
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Info) as cm:
        ########### Notes ##############
        ###
        ### - density mesh doenst seem to capture the same as 
        ###     'most contracted after first lapalce iter
        ### - first contraction brings points to a local
        ###     centroid - leaves to a point central to the leaf,
        ###     branches to a tagent line
        ### - high order branches near leaves tend to contract
        ###     further (in abs distance) than low order ones:
        ###     possibly due to that region's local dispersion?
        ###
        # ################################

        with open('skel_w_order_complete.pkl','rb') as f:
            skel_completed = dict(pickle.load(f))
        # pts_and_orders = [x for idx, x in enumerate(skel_completed.values()) if idx % 5==0]
        # breakpoint()
        # pts = [[pt for pt,order in pt_and_order_list ] for pt_and_order_list  in pts_and_orders]
        # breakpoint()
        pts =[arr([y[0] for y in x]) for x in skel_completed.values()] 

        labels =[arr([y[1] for y in x]) for x in skel_completed.values()] 

        # pcds = [skel.select_by_index(idxs) for idxs in idx_lists] 
        pcds = create_one_or_many_pcds([pts[0]])
        draw(pcds)
        contracted, total_point_shift, shift_by_step = extract_skeleton(pcds[0], max_iter = 5, debug=True)
        draw([contracted])
        # draw(pcds[])
        topology, topology_graph, skeleton, skeleton_points, skeleton_graph,rx_graph, orig_to_contracted= extract_topology(contracted)
       
        edges = topology_graph.edges(data=True)
        edge_to_orig = {tuple((x[0],x[1])):x[2].get('data') for x in edges}
        points = np.asarray(topology.points)
        lines = np.asarray(topology.lines)
        contraction_dist = np.linalg.norm(total_point_shift, axis=1)
        all_cyl_pcd, cyls, cyl_objects = skeleton_to_QSM(lines,points,edge_to_orig,contraction_dist)
        breakpoint()


        # detailed_pcd = recover_original_detail(pcds, save_result = True, 
        # save_file = 'branch8_orig_detail', file_prefix = 'skeletor_translated.pcd')
        # breakpoint()


        print('reading in idxs')
        with open('skeletor_branches_mod5_complete.pkl','rb') as f: 
            complete_branches = dict(pickle.load(f))
        idx_lists = [skel_idxs for idb, skel_idxs in enumerate(complete_branches.values())]
        idbs = [idb for idb, skel_idxs in enumerate(complete_branches.values())]
        print('reading in skel')
        skel = o3d.io.read_point_cloud('skeletor_translated.pcd')
        print('reading in branches')
        branches = [skel.select_by_index(idxs) for idxs in idx_lists]

        draw(branches)
        breakpoint()
        branches = [skel.select_by_index(idx_lists[0])]
        del skel
        ##Branch 0
        pcd= branches[0]
        ## Great first ratio. Almost all leaf fuzz pts
        print('removing outliers')
        inliers, inds = pcd.remove_statistical_outlier( nb_neighbors=5, std_ratio=4)
        inliers2, inds2 = pcd.remove_statistical_outlier( nb_neighbors=30, std_ratio=4)
        inliers4, inds4 = pcd.remove_statistical_outlier(  nb_neighbors=20, std_ratio=.4)
        # inds4 = inds
        final_inds = set(inds).intersection(set(inds2)).intersection(set(inds4))

        final_pcd = pcd.select_by_index(list(final_inds))
        # draw(final_pcd)

        # pcd = clean_cloud(branches[0])
        final_pcd = final_pcd.uniform_down_sample(4)

        # with open('branch0_2clean_cmag.pkl','rb') as f:
        #     b0_cmag = pickle.load(f)
        
        # color_continuous_map(final_pcd,b0_cmag)
        # draw([final_pcd])
        # high_c_idxs = np.where(b0_cmag>np.percentile(b0_cmag,80))[0]
        # high_c_pcd = final_pcd.select_by_index(high_c_idxs)
        # low_c_pcd = final_pcd.select_by_index(high_c_idxs,invert=True)
        # draw([high_c_pcd])
        # # clean_high_c_pcd = clean_cloud(high_c_pcd, iters=1)
        # clean_high_c_pcd, inds = high_c_pcd.remove_statistical_outlier( nb_neighbors=20, std_ratio=2)
        # draw([clean_high_c_pcd])
        # clean_high_c_pcd.paint_uniform_color([1,0,0])
        # breakpoint()   
        # idxs_in_orig = high_c_idxs[inds]
        # draw([final_pcd,clean_high_c_pcd])
        # left =  final_pcd.select_by_index(high_c_idxs,invert=True)
        # draw(left)   
        # breakpoint()      

        ### Extracting skeleton
        contracted, total_point_shift, shift_by_step = extract_skeleton(final_pcd, max_iter = 20, debug=True)
        draw([contracted])

        topology, topology_graph, skeleton, skeleton_points, skeleton_graph,rx_graph, orig_to_contracted= extract_topology(contracted)
        print('reached end ')
        # o3d.io.write_point_cloud(f"skel_stem20_defaults_topology.pcd", topology)
        # draw([skeleton])
        draw([topology])
        breakpoint()


        print('clustering...')
        # final_pcd = final_pcd.uniform_down_sample(2)
        labels = np.array( clean_high_c_pcd.cluster_dbscan(eps=.09, min_points=5,print_progress=True))  
        max_label = labels.max()
        # visualize the labels
        log.info(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        first = colors[0]
        colors[0] = colors[-1]
        colors[-1]=first
        clean_high_c_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        draw(clean_high_c_pcd)

        # pcd = o3d.io.read_point_cloud('banches_0_stat5_pt25.pcd')
        breakpoint()
        inliers3, ind3 = pcd.remove_statistical_outlier( nb_neighbors=20, std_ratio=.25)
        draw(inliers3)
        inliers3b = inliers3.uniform_down_sample(7)
        draw(inliers3b)
        breakpoint()

       


        ### defining and cluster mesh
        # print('defining normals')
        # normals_radius   = config['stem']["normals_radius"]
        # normals_nn       = config['stem']["normals_nn"]    
        # pcd.estimate_normals(    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn=normals_nn))
        # print('defining mesh via ball')
        # mesh = get_ball_mesh(pcd,radii= [0.001,0.0025,0.005,0.0075,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.1])
        # mesh = get_ball_mesh(pcd,radii= [0.08,0.1,0.2,0.3])
        print('drawing mesh')
        draw(mesh)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, .001)
        # c_mesh = get_surface_clusters(mesh, top_n_clusters=200, min_cluster_area=0, max_cluster_area=200)



        ## Investigating how good we did 

        # Color by over all contraction from pcd -> skeleton
        # o3d.io.write_point_cloud(f"skel_iso_trunk.pcd", in_pcd)
        from viz.viz_utils import color_continuous_map
        magnitude_of_contraction = np.array([np.linalg.norm(x) for x in total_point_shift])
        non_outliers = np.where(magnitude_of_contraction<np.percentile(magnitude_of_contraction,90))[0]
        magnitude_of_contraction = magnitude_of_contraction[non_outliers]
        non_outliers_pcd = pcd.select_by_index(non_outliers)
        color_continuous_map(non_outliers_pcd,magnitude_of_contraction)

        lower80 = np.where(magnitude_of_contraction<np.percentile(magnitude_of_contraction,80))[0]
        lower80_contraction = magnitude_of_contraction[lower80]
        lower80_pcd = pcd.select_by_index(lower80)
        color_continuous_map(lower80_pcd,lower80_contraction)
        draw([lower80_pcd])

        test = pcd.uniform_down_sample(8)
        c_mags = [np.array([np.linalg.norm(x) for x in pcd_point_shift]) for pcd_point_shift in shift_by_step]
        # Color by step by step contraction from pcd -> skeleton
        for c_mag in c_mags:
            color_continuous_map(pcd, c_mag)
            non_outliers = np.where(c_mags[0]<np.percentile(c_mags[0],95))[0]
            clean_c_mags = c_mags[0][non_outliers]
            non_outliers_pcd = pcd.select_by_index(non_outliers)
            color_continuous_map(non_outliers_pcd,clean_c_mags)
            draw(non_outliers_pcd)
            draw([pcd])


        # edges = topology_graph.edges(data=True)
        # edge_to_orig = {tuple((x[0],x[1])):x[2].get('data') for x in edges}
        # points = np.asarray(topology.points)
        # lines = np.asarray(topology.lines)
        # contraction_dist = np.linalg.norm(total_point_shift, axis=1)
        # skeleton_to_QSM(lines,points,edge_to_orig,contraction_dist):
        
        breakpoint()
        ####
        #    there is a bijection trunk to contracted - Same ids
        #    there is a bijection contracted to skeleton_points - same ids
        #    there is a surgection skeleton_points to skeleton, skeleton_graph
        #       ****Ids wise skeleton points = contracted.points = trunk.points
        #    there is a surjection skeleton to topology graph
        #    there is a surjection (albeit, of low order) topology_graph to topologogy
        # ####


        # distances = np.linalg.norm(points[lines[:, 0]] - points[lines[:, 1]], axis=1)
        # all_verticies = list(chain.from_iterable([x[2].get('data',[]) for x in edges]))
        
        # orig_verticies = [x for x in edge_to_orig.values()]
        # most_contracted_idx = np.argmax([len(x) for x in orig_verticies if x is not None])
        # most_contracted_list = orig_verticies[most_contracted_idx]

        # print("Mean:", np.mean(distances))
        # print("Median:", np.median(distances))
        # print("Standard Deviation:", np.std(distances))
        # colors = [[0,0,0]]*len(distances)
        # long_line_idxs = np.where(distances>np.percentile(distances,60))[0]
        # for idx in long_line_idxs: colors[idx] = [1,0,0]
        # topology.colors = o3d.utility.Vector3dVector(colors)

        # contracted_to_orig = {v:k for k,v in orig_to_contracted.items()} # mapping is a bijection
        # # For point with idp in topology.points, 
        # #   point = skeleton_points[contracted_to_orig[idp]]
        # #
        # # For point w/ idp in pcd, and idc in (0,1,2),
        # #  absolute difference of pcd.points[idp][idc]
        # #  and skeleton_points[idp][idc] is total_point_shift[idp][idc]

