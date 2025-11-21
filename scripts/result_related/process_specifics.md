# Tree Isolation Process for Epiphyte Area Extraction

## Overview

This document describes the methodology used to isolate individual trees from the original lidar scan (referred to RowBox "skio" or "collective" throughout the codebase) and subsequently extract epiphyte area measurements. The epiphyte area extraction process was essential for quantifying the surface area covered by non-woody vegetation (epiphytes) on each tree.

## Data Source

The input data consisted of a complete lidar scan covering a forest area. For processing efficiency, this scan was voxelized and cleaned before the tree isolation process. The original high-resolution point cloud was stored separately for detail recovery in later stages of the pipeline.

## Tree Isolation Methodology

The tree isolation process was performed in four sequential steps:

### 1. Trunk Identification

**Objective**: Identify points belonging to tree trunks (stems) using geometric properties.

**Method**: 
- The point cloud was processed to estimate surface normals for each point
- Points with horizontal normal vectors (indicating roughly vertical, cylindrical surfaces) were identified
- This filtering was performed using a configurable angle cutoff parameter that selected points with normals within a specified angular range from vertical
- The resulting "stem cloud" was further cleaned through statistical outlier removal and optional voxel downsampling

**Key Parameters**:
- `normals_radius`: Search radius for normal estimation
- `normals_nn`: Maximum number of neighbors for normal estimation
- `angle_cutoff`: Angular threshold (in degrees) for identifying horizontal normals
- `voxel_size`: Downsampling resolution for processing efficiency

### 2. Seed Cluster Identification

**Objective**: Identify distinct trunk base clusters that would serve as seed points for each individual tree.

**Method**:
- The stem cloud was cropped to isolate the bottom portion (typically bottom 10-18% by height percentile)
- This low cross-section was then clustered using DBSCAN with density-based parameters
- Each resulting cluster in this bottom region represents a distinct trunk base and seeds an individual tree
- These seed clusters were saved as reference points for the extension process

**Key Parameters**:
- `low_percentiles`: Height range defining the bottom cross-section (typically 16-18 percentile)
- `cluster_eps`: Maximum distance for DBSCAN clustering
- `min_points`: Minimum points required to form a cluster

### 3. Cluster Extension via KNN

**Objective**: Extend each seed cluster upward to classify all points belonging to that tree.

**Method**:
- Using a voxelized, cleaned version of the original scan (for computational efficiency)
- For each seed cluster, iteratively identified nearest neighbors using KDTrees
- Neighbors within a maximum distance threshold were added to the cluster
- This process continued over multiple cycles until no new neighbors were found within the search radius
- Points already assigned to other clusters were excluded to prevent overlap
- This resulted in extended clusters representing individual trees at reduced resolution

**Key Parameters**:
- `k`: Number of nearest neighbors queried per iteration
- `max_distance`: Maximum distance threshold for neighbor inclusion
- `cycles`: Number of extension iterations performed

### 4. Original Detail Recovery

**Objective**: Recover high-resolution point data for each isolated tree from the original, non-voxelized scan.

**Method**:
- For each voxelized tree cluster:
  1. Generated an oriented bounding box around the cluster extent
  2. Identified all original detail tiles/files that intersect with this bounding box
  3. For each intersecting file, performed a KNN search to find neighbors of voxelized tree points within the original high-resolution data
  4. Extracted original points, colors, and intensity values for all neighbors
  5. Combined results from all intersecting files to form a complete, high-resolution tree point cloud

This recovery step essentially reverses the initial voxelization to restore the full detail of the original lidar scan while maintaining the tree-level segmentation achieved during cluster extension.

## Epiphyte Area Calculation

Once individual trees were isolated with full resolution detail, epiphyte area was calculated through:

1. **Surface Reconstruction**: Converting the point cloud to a mesh representation using alpha shape or ball pivoting algorithms
2. **Vertical Shifts**: Computing vector displacements between matched points in sequential scan intervals to identify dynamic vs. static components
3. **Epiphyte Identification**: Using percentile-based filtering on vertical shift magnitudes to separate epiphytes (highly dynamic) from leaves (moderately dynamic) and woody material (relatively static)
4. **Area Projection**: Computing the projected surface area of the epiphyte component
5. **Normalization**: Expressing epiphyte area as Epiphyte Area Index (EAI), representing epiphyte surface area per unit ground area

## References

- Functions implementing this pipeline are located in:
  - `pyQSM/tree_isolation.py`: Tree isolation and cluster extension
  - `pyQSM/qsm_generation.py`: Trunk identification
  - `pyQSM/geometry/reconstruction.py`: Original detail recovery
  - `pyQ pytest canopy_metrics.py`: Epiphyte identification and area calculation



