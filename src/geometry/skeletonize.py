
import sys
from collections import defaultdict
from itertools import chain

sys.path.insert(0,'/code/code/pyQSM/src/')
import robust_laplacian
from plyfile import PlyData
import numpy as np
import polyscope as ps
from scipy.sparse import csr_matrix, diags, csgraph, vstack, linalg as sla
import open3d as o3d
from skspatial.objects import Cylinder

from copy import deepcopy, copy
import mistree as mist
from matplotlib import pyplot as plt
import rustworkx as rx
import networkx as nx


from geometry.point_cloud_processing import clean_cloud, cluster_and_get_largest,get_low_cloud
from set_config import config, log
from viz.viz_utils import draw
from utils.lib_integration import pts_to_cloud
from utils.math_utils import get_center,rot_90_x,unit_vector


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
    fps_points = int(contracted_pts.shape[0] * 0.1) # reduce by 10%
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
    # Define Weights
    WL = diags(laplacian_weighting)
    WH = diags(positional_weighting)

    A = vstack([L.dot(WL), WH]).tocsc()
    b = np.vstack([np.zeros((pts.shape[0], 3)), WH.dot(pts)])

    A_new = A.T @ A

    x = sla.spsolve(A_new, A.T @ b[:, 0], permc_spec='COLAMD')
    y = sla.spsolve(A_new, A.T @ b[:, 1], permc_spec='COLAMD')
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
                     step_wise_contraction_amplification = config['skeletonize']['step_wise_contraction_amplification']):
    # Hevily inspired by https://github.com/meyerls/pc-skeletor
    obb = pcd.get_oriented_bounding_box()
    allowed_range = (obb.get_min_bound(), obb.get_max_bound())
    
    termination_ratio,contraction_factor = set_amplification(step_wise_contraction_amplification,
                                                                len(np.asarray(pcd.points)),termination_ratio)
    
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
    laplacian_weights = (contraction_factor * 1000 * np.sqrt(np.mean(M.diagonal())) 
                         * np.ones(M.shape[0])) # WL

    iteration = 0
    volume_ratio = 1 # since areas array is len 1

    pts_current = pts
    shift_by_step = []
    total_point_shift = np.zeros_like(pts_current)

    # we run this until the volume of the previously added row
    #  becomes less than or equal to termination_ratio * the first row
    while (volume_ratio  > termination_ratio):
        log.info(f'{volume_ratio=}, {np.mean(laplacian_weights)=}, {np.mean(positional_weights)=}')

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
            pcd_point_shift = pts_current-pts_new
            total_point_shift += pcd_point_shift
            pts_current = pts_new
            shift_by_step.append(pcd_point_shift)

        if debug and iteration ==0:
            from viz.viz_utils import color_continuous_map
            curr_pcd = pts_to_cloud(pts_current)
            c_mag = np.array([np.linalg.norm(x) for x in pcd_point_shift])
            color_continuous_map(pcd,c_mag)
            # o3d.io.write_point_cloud(f"min_clean_high_c_colored.pcd", pcd)
            draw([pcd])
            high_c_idxs = np.where(c_mag>np.percentile(c_mag,60))[0]
            test = pcd.select_by_index(high_c_idxs)
            draw([test])
            testt = clean_cloud(test)
            draw([testt])
            # o3d.io.write_point_cloud(f"min_clean_high_c_top60.pcd", testt)
            # breakpoint()
            # max_cluster = cluster_and_get_largest(test, min_points=50)
            # o3d.io.write_point_cloud(f"min_clean_high_c_colored.pcd", pcd)
            

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
            # breakpoint()

        volume_ratio = np.mean(M_list[-1]) / np.mean(M_list[0])
        log.info(f"Completed iteration {iteration}")
        if iteration >= max_iteration_steps:
            break

    log.info(f'Finished after {iteration} iterations')

    contracted = pts_to_cloud(pts_current)

    return contracted, total_point_shift, shift_by_step

def animate_contracted_pcd(transient_pcd, constant_pcd_in,
                               init_rot: np.ndarray = np.eye(3),
                               steps: int = 360,
                               transient_period: int = 45,
                               point_size: float = 1.0,
                               output = '/code/code/pyQSM/test/',
                               rot_center = [0,0,0],
                               save = False,
                               file_name = 'pcd_compare_animation'):
        """
            Creates a GIF comparing two point cloud. 
        """
        import os 
        import time
        from scipy.spatial.transform import Rotation as R
        import imageio

        output_folder = os.path.join(output, 'tmp')
        print(output_folder)
        # os.mkdir(output_folder)

        # We 

        # Load PCD
        orig = deepcopy(transient_pcd)
        # if not trans_has_color:
        #     orig.rotate(init_rot, center=[0, 0, 0])

        # skel = copy(contracted)
        # skel.paint_uniform_color([0, 0, 1])
        # skel.rotate(init_rot, center=[0, 0, 0])

        constant_pcd = deepcopy(constant_pcd_in)
        # constant_pcd.paint_uniform_color([0, 0, 0])
        constant_pcd.rotate(init_rot, center=[0, 0, 0])

        transient_pcd = deepcopy(orig)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        vis.add_geometry(transient_pcd)
        vis.add_geometry(constant_pcd)

        ctl = vis.get_view_control()
        ctl.set_zoom(0.6)

        # Set smaller point size. Default is 5.0
        vis.get_render_option().point_size = point_size
        vis.get_render_option().line_width = 15
        vis.get_render_option().light_on = False
        vis.update_renderer()

        # Calculate rotation matrix for every step. Must only be calculated once as rotations are added up in the point cloud
        Rot_mat = R.from_euler('y', np.deg2rad(540 / steps)).as_matrix()

        image_path_list = []

        pcd_idx = 0
        for i in range(steps):
            orig.rotate(Rot_mat, center=rot_center)
            # skel.rotate(Rot_mat, center=rot_center)
            constant_pcd.rotate(Rot_mat, center=rot_center)

            if pcd_idx == 0:
                transient_pcd.points = orig.points
                transient_pcd.colors = orig.colors
                transient_pcd.normals = orig.normals
            if pcd_idx == 1:
                # pcd.paint_uniform_color([0, 0, 0])
                transient_pcd.points = constant_pcd.points
                transient_pcd.colors = constant_pcd.colors

            vis.update_geometry(transient_pcd)
            vis.update_geometry(constant_pcd)
            vis.poll_events()
            vis.update_renderer()

            # Draw pcd for 30 frames at a time
            #  remove for 30 between then
            if ((i % transient_period) == 0) and i != 0:
                pcd_idx = (pcd_idx + 1) % 2
                
            current_image_path = f"{output_folder}/img_{i}.jpg"
            if save:
                vis.capture_screen_image(current_image_path)
                image_path_list.append(current_image_path)

        vis.destroy_window()
        images = []
        log.info(f'Creating gif at {output_folder}{file_name}.gif')
        if save:
            for filename in image_path_list:
              images.append(imageio.imread(filename))
            log.info(f'Creating gif at {image_path_list[0]}')
            imageio.mimsave(os.path.join(os.path.dirname(image_path_list[0]), 
                                         '{}.gif'.format(file_name)), 
                                         images, format='GIF')
            
def skeleton_to_QSM(lines,points,edge_to_orig,contraction_dist):
    cyls = []
    cyl_objects = []
    for idx, line in enumerate(lines):
        start = points[line[0]]
        end = points[line[1]]

        orig_verticies = edge_to_orig[tuple(line)] 
        contraction_dists = contraction_dist[orig_verticies]
        
        cyl_radius= np.mean(contraction_dists)
        cyl  = Cylinder.from_points(start,end,cyl_radius*1.1)
        cyl_pts = cyl.to_points(n_angles=20).round(3).unique()
        cyl_pcd = o3d.geometry.PointCloud()
        cyl_pcd.points = o3d.utility.Vector3dVector(cyl_pts)
        cyls.append(cyl_pcd)
        cyl_objects.append(cyl)
        
        if test:
            breakpoint()
            print('test')
        if idx %10 == 0:    
            print(f'finished iteration {idx}')
        
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
    pts = []
    for cyl in cyls: 
        pts.extend(np.array(cyl.points))
    all_cyl_pcd= o3d.geometry.PointCloud()
    all_cyl_pcd.points = o3d.utility.Vector3dVector(pts)
    draw(cyls)
    return all_cyl_pcd, cyls, cyl_objects   
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

    




if __name__ == "__main__":
    # pcd = o3d.io.read_point_cloud("skeletor_vox_0.03_sta_6-4-3.pcd")

    ## Files for Testing w/ minimally cleaned trunk
    # trunk =o3d.io.read_point_cloud("skeletor_min_clean_trunk.pcd")
    # test =o3d.io.read_point_cloud("min_high_c_top60_clean_trunk.pcd")
    # pcd = pcd.uniform_down_sample(4)
    # contracted = o3d.io.read_point_cloud("min_clean_high_c_contracted.pcd")
    # pcd_colored = o3d.io.read_point_cloud("min_clean_high_c_colored.pcd")

    ## Files for Testing w/ first try
    # pcd = o3d.io.read_point_cloud("skeletor_trunk.pcd")
    # pcd_colored = o3d.io.read_point_cloud("high_contraction1.pcd")
    # trunk = o3d.io.read_point_cloud("high_contraction_iter1_top60.pcd")
    # test =o3d.io.read_point_cloud("high_c_pcd_contracted.pcd")

    ## Files for Testing w/ super clean
    # pcd           = o3d.io.read_point_cloud("pcd_super_3to9.pcd")
    # pcd_colored   = o3d.io.read_point_cloud("pcd_super_4to9_high_c_colorted.pcd")
    # trunk = o3d.io.read_point_cloud("high_c_super4to9.pcd")

    trunk = o3d.io.read_point_cloud("skeletor_super_clean.pcd")
    trunk = trunk.uniform_down_sample(4)
    # draw(trunk)
    # draw(pcd)
    # draw(test)
    # draw(pcd_colored)
    breakpoint()

    contracted, total_point_shift, shift_by_step = extract_skeleton(trunk, max_iter = 20, debug=False)
    draw([contracted])
    # o3d.io.write_point_cloud(f"min_clean_high_c_contracted.pcd", contracted)

    topology, topology_graph, skeleton, skeleton_points, skeleton_graph,rx_graph, orig_to_contracted= extract_topology(contracted)
    print('reached end ')
    # o3d.io.write_point_cloud(f"min_clean_high_c_topology.pcd", contracted)
    # draw([skeleton])
    draw([topology])

    breakpoint()
    ## Finding point to rotate around
    base_center = get_center(np.asarray(trunk.points),center_type = "bottom")
    # mn = trunk.get_min_bound()
    # centroid = mn+((mx-mn)/2)
    # base = (base_center[0], base_center[1], mn[2])
    # sp = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    # sp.paint_uniform_color([1,0,0])
    # sp.translate(base)

    for pcd in [trunk,contracted,skeleton,topology]:
        pcd.translate(np.array([-x for x in base_center ]))
        pcd.rotate(rot_90_x)
        pcd.rotate(rot_90_x)
        pcd.rotate(rot_90_x)

    # draw([ttrunk,trunk])
    # draw([contracted,sp])
    # draw([skeleton])
    # draw([topology])
    
    
    # gif_center = get_center(np.asarray(trunk.points),center_type = "centroid")
    # eps=config['trunk']['cluster_eps']
    # min_points=config['trunk']['cluster_nn']
    # labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))
    
    # breakpoint()
    # animate_contracted_pcd(trunk,topology, point_size=4,rot_center = gif_center, steps=360, transient_period= 40, save = True, file_name = '_super_whole_and_topo')
    
    # animate_contracted_pcd(pcd_colored,  topology, point_size=3,  
    #                        rot_center = gif_center,  steps=360, save = True, file_name ='_min_w_topo') 
    # animate_contracted_pcd(trunk.voxel_down_sample(.05),  topology,  point_size=3,  rot_center = gif_center,   steps=360, save = True, file_name = '_min_w_topo' 
    
    ####
    #    there is a bijection trunk to contracted - Same ids
    #    there is a bijection contracted to skeleton_points - same ids
    #    there is a surgection skeleton_points to skeleton, skeleton_graph
    #       ****Ids wise skeleton points = contracted.points = trunk.points
    #    there is a surjection skeleton to topology graph
    #    there is a surjection (albeit, of low order) topology_graph to topologogy
    ####


    points = np.asarray(topology.points)
    lines = np.asarray(topology.lines)
    distances = np.linalg.norm(points[lines[:, 0]] - points[lines[:, 1]], axis=1)
    edges = topology_graph.edges(data=True)
    edge_to_orig = {tuple((x[0],x[1])):x[2].get('data') for x in edges}
    all_verticies = list(chain.from_iterable([x[2].get('data',[]) for x in edges]))
    
    orig_verticies = [x for x in edge_to_orig.values()]
    most_contracted_idx = np.argmax([len(x) for x in orig_verticies if x is not None])
    most_contracted_list = orig_verticies[most_contracted_idx]

    print("Mean:", np.mean(distances))
    print("Median:", np.median(distances))
    print("Standard Deviation:", np.std(distances))
    colors = [[0,0,0]]*len(distances)
    long_line_idxs = np.where(distances>np.percentile(distances,60))[0]
    for idx in long_line_idxs: colors[idx] = [1,0,0]
    topology.colors = o3d.utility.Vector3dVector(colors)

    contraction_dist = np.linalg.norm(total_point_shift, axis=1)
    contracted_to_orig = {v:k for k,v in orig_to_contracted.items()} # mapping is a bijection
    # For point with idp in topology.points, 
    #   point = skeleton_points[contracted_to_orig[idp]]
    #
    # For point w/ idp in pcd, and idc in (0,1,2),
    #  absolute difference of pcd.points[idp][idc]
    #  and skeleton_points[idp][idc] is total_point_shift[idp][idc]
    test = False
    cyls = []
    cyl_objects = []
    pcd_pts_contained = []
    pcd_points = np.array(pcd_colored.points)
    for idx, line in enumerate(lines):
        from skspatial.objects import Cylinder
        start = points[line[0]]
        end = points[line[1]]
        vector = end-start
        uvector = unit_vector(vector)    

        orig_verticies = edge_to_orig[tuple(line)] 
        contraction_dists = contraction_dist[orig_verticies]
        
        cyl_radius= np.mean(contraction_dists)
        cyl  = Cylinder.from_points(start,end,cyl_radius*1.1)
        cyl_pts = cyl.to_points(n_angles=20).round(3).unique()
        cyl_pcd = o3d.geometry.PointCloud()
        cyl_pcd.points = o3d.utility.Vector3dVector(cyl_pts)
        cyls.append(cyl_pcd)
        cyl_objects.append(cyl)
        
        if test:
            breakpoint()
            print('test')
        if idx %10 == 0:    
            print(f'finished iteration {idx}')
        

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
    pts = []
    for cyl in cyls: 
        pts.extend(np.array(cyl.points))
    all_cyl_pcd= o3d.geometry.PointCloud()
    all_cyl_pcd.points = o3d.utility.Vector3dVector(pts)
    draw(cyls)
    breakpoint()
    gif_center = get_center(np.asarray(trunk.points),center_type = "bottom")
    animate_contracted_pcd( all_cyl_pcd,trunk.voxel_down_sample(.05),  point_size=3, rot_center = gif_center, steps=360, save = True, file_name = '_proto_qsm_trunk',transient_period = 30)
    animate_contracted_pcd( all_cyl_pcd,topology,  point_size=3, rot_center = gif_center, steps=360, save = True, file_name = '_proto_qsm_topo',transient_period = 30)
    
    breakpoint()


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


