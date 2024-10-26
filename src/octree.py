import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np

class AwareInternalPointNode(o3d.geometry.OctreeInternalPointNode):
    parent = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def get_update_function(self,*args, **kwargs):
        return self.update(*args,**kwargs)

    def update(self, idx =None, parent = None):
        if idx:
            f = super().get_update_function(idx)
            f()
        if parent:
            self.set_parent = parent

class AwarePointColorLeafNode(o3d.geometry.OctreePointColorLeafNode):
    parent = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent
    
    def get_update_function(self,*args, **kwargs):
        return self.update(*args,**kwargs)

    def update(self, idx =None, color = None, parent = None):
        if idx and color:
            f = super().get_update_function(idx, color)
            f()
        if parent:
            self.set_parent = parent


def draw_leaves(nodes, source_pts):
    x,y,z,c= [],[],[],[]
    for node, node_info in nodes:
        node_pts = source_pts[node.indices]
        x.append(node_info.origin[0])
        y.append(node_info.origin[1])
        z.append(node_info.origin[2])
        c.append('r')
        x.extend(node_pts[:,0])
        y.extend(node_pts[:,1])
        z.extend(node_pts[:,2])
        c.extend(['b']*len(node_pts))
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x,y,z,'c')
    plt.show()

def color_node_pts(nodes, source_pcd , color = [1,0,0]):
    all_idxs=[]
    for idn, (node,_) in enumerate(nodes):
        print(f"painted {idn} of {len(nodes)} nodes ...")
        new_ids = node.indices
        all_idxs.extend(new_ids)
        for idx in new_ids: 
            f = node.get_update_function(idx, color)
            f(node)
        print(f'found {len(new_ids)} new points')
    all_idxs = np.array(list(set(all_idxs)))
    node_pcd = source_pcd.select_by_index(all_idxs)
    node_pcd.paint_uniform_color(color)
    return node_pcd

def nodes_from_point_idxs(octree,algo_pcd_pts, idxs):
    nodes = [octree.locate_leaf_node(algo_pcd_pts[idx]) for idx in idxs]
    unique_origins =  set([tuple(node[1].origin) for node in nodes])
    unique_nodes= []
    for tup in unique_origins:
        for node in nodes:
            if tuple(node[1].origin) == tup:
                unique_nodes.append(node)
                break
    return unique_nodes

def nodes_to_pcd(nodes,source_pcd):
    all_idxs=[]
    for idn, (node,_) in enumerate(nodes):
        print(f"painted {idn} of {len(nodes)} nodes ...")
        new_ids = node.indices
        all_idxs.extend(new_ids)
    all_idxs = np.array(list(set(all_idxs)))
    node_pts = np.asarray(source_pcd.points)[all_idxs]
    node_pcd = source_pcd.select_by_index(all_idxs)
    return node_pts, node_pcd
    

def cloud_to_octree(pcd, max_depth = 9):
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),center=pcd.get_center())
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(1000, 3)))
    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    return octree


# def awaken_node(node,node_info):
#     parent = node
#     if isinstance(node, o3d.geometry.OctreeInternalNode):
#         if isinstance(node, o3d.geometry.OctreeInternalPointNode):
#             for child in node.children:
#                 if child is not None:
#                     f = node.get_update_function(parent=parent)
#                     f(child)
#     elif isinstance(node, o3d.geometry.OctreeLeafNode):
#         if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
#             print("{}{}: Reached leaf node at depth {} has {} points with origin {}".
#                   format('    ' * node_info.depth, node_info.child_index,
#                          node_info.depth, len(node.indices), node_info.origin))
#     return True

# def createAwarePointColorLeafNode(*args,**kwargs):
#     return AwarePointColorLeafNode(*args, **kwargs) 

# def create_a_were_tree(pcd, octree = None):
#     were_tree=None

#     f_init= AwareInternalPointNode.__init__
#     f_update= AwareInternalPointNode.update
#     fi_init= AwarePointColorLeafNode
#     fi_update= AwarePointColorLeafNode.update
    
#     if octree:
#         were_tree = o3d.geometry.Octree.__init__(octree.max_depth,
#                                                 octree.origin,
#                                                 octree.size,)
#     if pcd:
#         were_tree = o3d.geometry.Octree(max_depth=9)
#         breakpoint()
#         for point in np.asarray(pcd.points): were_tree.insert_point(point,f_init,f_update)
    
#     were_tree.traverse(awaken_node)
#     return were_tree
    

# if __name__=="__main__":
#     ex_node=  octree.locate_leaf_node(pcd_pts[0])
#     pcd.paint_uniform_color([0,1,0])
#     node_pcd = color_node_pts([ex_node], pcd, [1,0,0])
#     draw_leaves([ex_node], pcd_pts)
#     draw([octree])
#     draw([pcd, node_pcd])

#     nodes = [octree.locate_leaf_node(pcd_pts[idx]) for idx in range(100)]
#     draw_leaves(nodes, pcd_pts)
#     pcd.paint_uniform_color([0,1,0])

#     breakpoint()
#     node_pcd = color_node_pts(nodes, pcd, [1,0,0])
#     draw([octree])
#     draw([pcd, node_pcd])
