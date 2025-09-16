from matplotlib import pyplot as plt
import numpy as np
from set_config import log


def plot_dist_dist(pcd,distances=None):
    log.info(f'Computing KNN distance')
    if distances is None:
        distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f'{avg_dist=}')
    log.info(f'plotting KNN distance distrobution')
    bins = np.histogram_bin_edges(distances, bins=300)
    cum_sum=[ len([x for x in distances if lb>x]) for lb in bins]
    ax = plt.subplot()
    ax.scatter(bins,cum_sum)
    plt.axvline(x=avg_dist, color='r', linestyle='--')
    plt.axvline(x=avg_dist*2, color='r', linestyle='--')
    plt.show()