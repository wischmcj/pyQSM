# Methods for Isolating Epiphytes from LiDAR Point Clouds

## Introduction

This document describes the computational methods developed to identify and isolate epiphytic vegetation from terrestrial LiDAR (light detection and ranging) point cloud data. The methodology involves point-based segmentation of three-dimensional point clouds to classify points as belonging to epiphytes, leaves, or woody material (branches and trunk). This segmentation then enables the quantification of the 2-D area covered by epiphytes - in proportion to leaves and woody material - which can in turn be used to demostrate the value of a more specific canopy cover indicies, namely Epiphyte Area Index.

In order to esitmate the proportion of canopy projected area attributable to epiphytes, point-based segmentation is performed to label each point withing the point cloud as lying on the surface of of an epiphyte, wood (i.e. a tree stem or branch) or a leaf. Following the segmentation of the point cloud, the area covered by each subset of points is calculated based off of both 1. the 2-D area covered by clustered sets of points (to provide a lowerbound for effective area) and 2. more percisely using a ray-casting based approach (to provide an upper bound of effective area). 

Prior to feature collection to support segmentation, point cloud data underwent preprocessing to facilitate the identification of each surface's geometric characteristics. Points were voxelized (in three-dimensional space) with a voxel size of .05m. This process involves merging all points that fall within the same 0.05m voxel - instead representing these points with a single point.. This step ensures a lowerbound on point cloud density, obsfucating small-scale surface variation while preserving macro-scale differences in point density and relative position. This serves to reduce computational complexity while still allowing for the identification of the geometric features differentiating tree stems/branches and epiphytes.

Operating on the 'cleaned' clouds, the neighborhood of each point is then analyzed with the intent of gathering insights about the surface on which the point lies. Given the prominence of epiphytes compared to other canopy featues, segemetation of epiphytes is greatly facilitated by measures of surface topology that indicate the inclusion of a given point within physically outstaning objects. At a high level, the intention is to assign a ranking to each point in the point cloud that represents how pronounced the 'curvature' of the suface represented by the cloud is in the neighborhood of each point. 

Many such measures are utilized in the field of object detection and these are primarily derrived from analyis of the 'gradient' of a surface. The measurement ultimately chosen for use in our segmentation is known as the surface's mean-curvature vector (https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf, page 88). In particular, we calculate the magnitude of this vector as a proxy for the prominence of the object which the surface represents. It is important to note that the self-same calculation used in this paper is in common use in algorithms for 'laplacian based contraction' - a method commonly used in computation geometry to 'smooth' surfaces by reducing localized suface roughness while preserving larger-scale surface characteristics. 

In practice, we calculate the mean curvature of the point cloud's surface at each point through the use of a descritized Laplace-Beltrami operator (https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf, page 88). Given the mean curvature at each point, points in the 75th precentile by magnitude are selected as candidate points for labeling as epiphytes. Now, the points representing leaves likewise form apparent surfaces with high curvature. s our intent is to identify pendulous epiphytes (i.e. spanish moss), we further segment these points of high curvature based on the direction of the point's associated mean-curvature vector. In practice, this allows us to form two distinct groups of points 1. points that lie on surfaces below adjacent portions of the tree canopy - which are considered to be p. epiphytes and 2. points that lie on surfaces above or at an similar height to adjacent points of the tree canopy - which are considered to be leaves. 

Following classification, the aggregate projected area of the points in each segementation group are analyzed seperately. Two approaches the the estimation of projected area - one representing an lowerbound for the metric and the other an upperbound. To obtain said lowerbound a cluster based projection was performed; points in each category were organized into spatially coherent groups via a DBSCAN clustering algorithm. The resultant clusters (also reffered to as 'clumps') have a maximum vertical extent (height) of 0.5 m, so that large areas of leaf/eiphyte cover are considered to consist of multiple clumps. To obtain an upper-bound for projected area a ray casting based technique is used.  In this approach, a triangular mesh representing the 3d volume represented by each group of points is generated using an intrisic Delaunay triangulation The projection process iteratively cast rays across the mesh, adds the area of interception regions to the final sum, removes said regions from considerations and and repeats this process until all surfaces of the mesh have been accounted for

The primary distinction between these two approaches lies in how each accounts for three-dimensional overlap of canopy components. The manner in which we account for this overlap is analaous to widely accepted methods for measuring leaf area index (citation needed). While it is typical, in the case of LAI measurements, for this overlap to be estimated using predictive factors (ie. ceptometer readings), we find that the measurments provdided here align well with ceptometer estimated LAI measurements reported in the existing literature ( Vitar et al. ). 

The Epiphyte Area Index (EAI) was introduced as a novel metric to quantify epiphyte contribution to canopy structure:

\[
\text{EAI} = \frac{\text{Projected Pendulous Epiphyte Area}}{\text{Projected Canopy Area (PCA)}}
\]

This index parallels the classical Leaf Area Index (LAI) and enables direct comparison of epiphyte abundance relative to total canopy extent. A modified LAI metric (LAI\_proposed) was also calculated excluding epiphyte area:

\[
\text{LAI}_{\text{proposed}} = \frac{\text{Wood Area} + \text{Leaf Area}}{\text{Projected Canopy Area (PCA)}}
\]

This allows for separation of epiphyte contributions from traditional foliage metrics.

## Technical Parameters

Key parameters used in the analysis:

- Voxel spacing: 0.1 m
- Alpha shape curvature (PCA): 0.1
- Alpha shape curvature (clumps < 1 m diameter): 0.25
- Maximum clump height: 0.5 m
- Clustering method: DBSCAN (for corticular epiphytes)
- Mesh generation: Delaunay triangulation
- Ray casting: Parallel rays from nadir


The point cloud was then projected onto a two-dimensional horizontal plane to enable calculation of projected canopy area. Projected Canopy Area (PCA) was computed as an alpha shape (Edelsbrunner et al., 1983) with a curvature parameter (α) of 0.1 around the projected points. For spatially clustered structures with diameters less than 1 m (including epiphyte clumps), an alpha shape curvature of 0.25 was employed to better capture the geometric complexity of these smaller features

## References

Edelsbrunner, H., Kirkpatrick, D., & Seidel, R. (1983). On the shape of a set of points in the plane. *IEEE Transactions on Information Theory*, 29(4), 551-559.

Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining*, 226-231.

