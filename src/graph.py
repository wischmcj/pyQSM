
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph

        
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

def graph2data(graph):
    graph = graph.tocoo()
    weights = graph.data
    index1 = graph.row
    index2 = graph.col
    return index1, index2, weights

def data2graph(index1, index2, weights, num_nodes):

    graph = csr_matrix((weights, (index1, index2)), shape=(num_nodes, num_nodes))
    return graph

def graph_scale_cut(graph, scale_cut_length, num_nodes):
    index1, index2, distances = gr.graph2data(graph)
    condition = np.where((distances >= scale_cut_length))[0]
    num_removed_edges_fraction = float(len(index1) - len(condition))/float(len(index1))
    index1, index2, distances = index1[condition], index2[condition], distances[condition]
    graph_cut = gr.data2graph(index1, index2, distances, num_nodes)
    return graph_cut, index1, index2, num_removed_edges_fraction


def k_nearest_neighbour_scale_cut(x, y, scale_cut_length, k_neighbours, z=None):
    if z is None:
        vertices = np.array([x, y]).T
    else:
        vertices = np.array([x, y, z]).T
    knn = kneighbors_graph(vertices, n_neighbors=k_neighbours, mode='distance')
    knn, index1, index2, num_removed_edges_fraction = graph_scale_cut(knn, scale_cut_length, len(x))
    if z is None:
        return x, y, knn, num_removed_edges_fraction
    else:
        return x, y, z, knn, num_removed_edges_fraction
    
def convert_tomo_knn_length2angle(k_nearest_neighbour_graph, number_of_nodes):
    index1, index2, distances = graph2data(k_nearest_neighbour_graph)
    distances_angles = 2. * np.arcsin(distances / 2.)
    k_nearest_neighbour_graph_angle = data2graph(index1, index2, distances_angles, number_of_nodes)
    return k_nearest_neighbour_graph_angle


def construct_mst(x, y, z=None, k_neighbours=20, two_dimensions=False, scale_cut_length=0., is_tomo=False):
    if two_dimensions == True:
        vertices = np.array([x, y]).T
    else:
        vertices = np.array([x, y, z]).T
    if scale_cut_length != 0.:
        if two_dimensions == True:
            x, y, k_nearest_neighbour_graph, num_removed_edges = \
                k_nearest_neighbour_scale_cut(x, y, scale_cut_length, k_neighbours)
        else:
            x, y, z, k_nearest_neighbour_graph, num_removed_edges = \
                k_nearest_neighbour_scale_cut(x, y, scale_cut_length, k_neighbours, z=z)
    else:
        k_nearest_neighbour_graph = kneighbors_graph(vertices, n_neighbors=k_neighbours, mode='distance')
    if is_tomo == True:
        k_nearest_neighbour_graph = convert_tomo_knn_length2angle(k_nearest_neighbour_graph, len(x))
    tree = minimum_spanning_tree(k_nearest_neighbour_graph, overwrite=True)
    tree = tree.tocoo()
    edge_length = tree.data
    index1 = tree.row
    index2 = tree.col
    x1 = x[index1]
    x2 = x[index2]
    edge_x = np.array([x1, x2])
    y1 = y[index1]
    y2 = y[index2]
    edge_y = np.array([y1, y2])
    if two_dimensions == False:
        z1 = z[index1]
        z2 = z[index2]
        edge_z = np.array([z1, z2])
    edge_index = np.array([index1, index2])
    if scale_cut_length == 0.:
        if two_dimensions == True:
            return edge_length, edge_x, edge_y, edge_index
        else:
            return edge_length, edge_x, edge_y, edge_z, edge_index
    else:
        if two_dimensions == True:
            return edge_length, edge_x, edge_y, edge_index, num_removed_edges
        else:
            return edge_length, edge_x, edge_y, edge_z, edge_index, num_removed_edges
