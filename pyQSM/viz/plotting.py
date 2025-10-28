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


def plot_3d(xyz, labels=None,
            cutoff=.1,
            elev=40,
            azim=110,
            roll=0,
            save_file=None):
    x, y, z = xyz
    rands = np.random.sample(len(x))
    x = x[rands<cutoff]
    y = y[rands<cutoff]
    z = z[rands<cutoff]
    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(x, y, z, marker=".")
    if labels is not None:
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.set_zlabel(labels[2])
    axis.view_init(elev=elev, azim=azim, roll=roll)
    plt.show()
    breakpoint()


def histogram(feature, feat_name='Feature'):
    title = f'{feat_name} Histogram'
    plt.hist(feature, bins=100, color='dodgerblue', edgecolor='black', alpha=0.75)
    plt.title(title)
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()