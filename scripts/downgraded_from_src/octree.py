

def build_octree(pcd,
                   ot_depth=4,
                   ot_expand_factor = 0.01):
    print('octree division')
    octree = o3d.geometry.Octree(max_depth=ot_depth)
    octree.convert_from_point_cloud(pcd, size_expand=ot_expand_factor)
    # o3d.visualization.draw_geometries([octree])
    return octree


def node_to_pcd(parent_pcd,node):
    nb_inds = node.indices
    nbhood = parent_pcd.select_by_index(nb_inds)
    nb_labels = np.array( nbhood.cluster_dbscan(eps=.5, min_points=20, print_progress=True))
    # nb_colors = color_and_draw_clusters(nbhood, nb_labels)
    return nbhood, nb_labels


def get_leaves(node, leaves):
    if not isinstance(node, o3d.geometry.OctreeLeafNode): 
        for c_node in [x for x in node.children if x]:
            get_leaves(c_node, leaves)
    else:
        leaves.append(node)
        return leaves

def get_containing_tree(octree, 
                  find_node):
    ret_tree = []
    def is_ancestor(node, ancestor_tree, find_idx):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            for idc, child in enumerate(node.children):
                if child is not None:
                    if find_idx in node.indices:
                        ancestor_tree.append(idc)
                        print(f'found find idx {node=} {idc=}')  
                        return is_ancestor(child, ancestor_tree, find_idx)
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
                return True
    find_idx = find_node.indices[0]
    is_ancestor(octree.root_node, ret_tree, find_idx)
    return ret_tree

    # def traverse_get_parent(node, node_info):
#     if found_node: return True
#     early_stop = found_node or False
#     if isinstance(node, o3d.geometry.OctreeInternalPointNode):
#         for child in node_info.children:
#             if child is not None:
#                 if find_idx in node.indicies:
#                     curr_tree.append((node,node_info))
#                     log.info(f'found find idx in node {node} {node_info}')   
#                 else:
#                     early_stop = True
#     elif isinstance(node, o3d.geometry.OctreeLeafNode):
#         if node_info.origin == find_origin:
#             log.info(f'found find node, ending with {curr_tree}')
#             found_node = True
#             early_stop = True
#     return early_stop


# def find_local_octree_nodes(oct, point):
#     pt_node = oct.locate_leaf_node(point)
#     pt_node
#     traversed = []
#     traversed_info = []
#     def find(node, node_info, depth = 4):
#     octree.locate_leaf_node(pcd.points[0])

def get_ancestors(octree, 
                  find_node):
    ret_tree = []
    def is_ancestor(node, ancestor_tree, find_idx):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            for idc, child in enumerate(node.children):
                if child is not None:
                    if find_idx in node.indices:
                        ancestor_tree.append(idc)
                        print(f'found find idx {node=} {idc=}')  
                        return is_ancestor(child, ancestor_tree, find_idx)
                elif child is None:
                    print(f'child {idc} is none')
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
                return True
    find_idx = find_node.indices[0]
    is_ancestor(octree.root_node, ret_tree, find_idx)
    return ret_tree