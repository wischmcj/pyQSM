import open3d as o3d
import numpy as np

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

def node_to_pcd(parent_pcd,node):
    nb_inds = node.indices
    nbhood = parent_pcd.select_by_index(nb_inds)
    nb_labels = np.array( nbhood.cluster_dbscan(eps=.5, min_points=20, print_progress=True))
    # nb_colors = color_and_draw_clusters(nbhood, nb_labels)
    return nbhood, nb_labels

def get_leaves(node, leaves):
    if isinstance(node, o3d.geometry.OctreeLeafNode): 
        for c_node in [x for x in node.children if x]:
            get_leaves(c_node, leaves)
    else:
        leaves.append(node)
        return leaves
    
agg = []
def agg_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            n = len([x for x in node.children if x])
            # for child in node.children:
            #     if child is not None:
            #         n += 1

            num_pts = len(node.indices)
            spaces_by_depth = '    ' * node_info.depth
            print(
                f"""{spaces_by_depth}{node_info.child_index}: Internal node at depth {node_info.depth} 
                    has {n} children and {num_pts} points ({node_info.origin})""")

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        agg.append((node,node_info))
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        # log.info(f'Reached a unrecognized node type: {type(node)}')
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop