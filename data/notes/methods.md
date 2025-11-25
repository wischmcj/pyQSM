# Methods

## Meta

### Overview

**Goal**: Determine the portion of projected canopy area that can be attributed to epiphytes. Introduce a summary metric - Epiphyte Area Index (EAI) - to supplement the classic Leaf Area Index (LAI).

**Methodology**: Perform point-based segmentation of point-cloud data - labeling each point as part of the terrain (ignored) or as part of an epiphyte, leaf, or 'wood' (a catch-all for remaining points). Isolate points labeled as p. epiphytes from other point-cloud components.

## Glossary

- **P. Epiphyte**: "Pendulous" Epiphyte. A stand-in term to refer to epiphytes that hang down below tree branches.
- **C. Epiphyte**: Corticular Epiphyte. All non-pendulous epiphytes.
- **Clump**: Spatially clustered points with a maximum height of 0.5 meters.
- **Ray Casting**: A technique for simulating exposure of objects from a given direction (i.e., 'as seen from above'). Given a representation of a 3D object, one generates several thousand parallel vectors ('rays') and calculates where each vector intercepts the object.

## Techniques

### Projected Canopy Area

Points are voxelized in 3D with a spacing of 0.1, then projected to 2D. PCA is then calculated as an alpha shape of curvature 0.1 around these points.

### 'Clump' Projected Area

For sets of clumps containing clumps with diameter less than 1 meter, an alpha shape curvature of 0.25 is used. This includes all leaf and epiphyte area calculations.

### Identifying P. Epiphytes

Identified as points with highly-varied local surfaces. Such points are identified as those located at local maxima with regard to:
1. The density of the point cloud
2. The magnitude of 'curvature' in the point-cloud's surface.

### Identifying C. Epiphytes

**Not intended for reporting, needs refining**

Identified via color analysis of RGB and intensity (as a proxy for albedo) values. DBSCAN clustering used to identify groups of similar points from among the non-leaf, non-p. epiphyte points. The points chosen as the c. epiphytes have a *neighborhood* of points with distinctly green hue.

## Formatted Data (Deprecated)

**Metrics Without Overlap**: Initial projected areas calculated by projection to 2D without prior clustering.

| Column | Description |
|--------|-------------|
| Projected Canopy Area | Projected area of the entire isolated tree. |
| Non-Leaf, Non-P. Epiphytes | Sum of the projected area of the tree with p. epiphytes and leaves removed. |
| Leaves and P. Epiphytes | Sum of the projected area of p. epiphytes and projected area of leaves. |
| Non-Leaf, Non-Epiphyte | Similar to 'Non-Leaf, Non-P. Epiphytes' but with corticular epiphytes removed as well. Includes epiphytes such as mosses and ferns, which were identified based on RGB color analysis. |
| Leaves Only | Projected area of all points labeled as leaves. |
| P. Epiphytes Only | Projected area of all points labeled as p. epiphytes. |

## Raycasting Projection

**Metrics With Overlap**: Metrics calculated using raycasting techniques. A mesh representing the points is produced via Delaunay triangulation, progressively casting rays, removing/summing the area of interception regions and repeating until all mesh components have been removed. Primarily differs from the deprecated method in that (with the exception of PCA) the areas of overlapping objects are counted separately.

| Column | Description |
|--------|-------------|
| Projected Canopy Area (PCA) | Projected area of the entire isolated tree, calculated as an alpha shape with curvature 0.1. Overlap not double counted. |
| 'Wood' (Non-Leaf, Non-P. Epiphytes) | Sum of the projected area of each branch, with epiphytes and leaves removed. |
| Leaf Area | Projected area of points identified as leaves. |
| P. Epiphytes Only | Projected area of points identified as p. epiphytes. |
| Leaf Area Index - Std. (Total / PCA) | LAI as it is classically defined in modern literature. Sum of 'Wood', 'Leaf' and 'P. Epiphyte' Area over PCA. |
| LAI Proposed (Non-Epi / PCA) | LAI with epiphyte area excluded. Sum of 'Wood' and 'Leaf' Area over PCA. |
| Epiphyte Area Index (EAI) | P. Epiphyte Area over PCA. |

## Cluster Projection

**Metrics With Overlap**: Metrics calculated by clustering points, projecting clusters in 2D, then summing over the projected areas of those clusters. Primarily differs from the deprecated method in that (with the exception of PCA) the areas of overlapping objects are counted separately.

| Column | Description |
|--------|-------------|
| Projected Canopy Area (PCA) | Projected area of the entire isolated tree, calculated as an alpha shape with curvature 0.1. Overlap not double counted. |
| 'Wood' (Non-Leaf, Non-P. Epiphytes) | PCA is used here as clustering is not applicable to the largely contiguous trunk and branches. |
| Leaf Area | Sum of the projected area of each clump of leaves. |
| P. Epiphytes Only | Sum of the projected area of each clump of epiphytes. |
| Leaf Area Index - Std. (Total / PCA) | LAI as it is classically defined in modern literature. Sum of 'Wood', 'Leaf' and 'P. Epiphyte' Area over PCA. |
| LAI Proposed (Non-Epi / PCA) | LAI with epiphyte area excluded. Sum of 'Wood' and 'Leaf' Area over PCA. |
| Epiphyte Area Index (EAI) | P. Epiphyte Area over PCA. |

## ML Isolation Refinement

Ray-casting based projection data for refined, isolated trees. The procedural isolation of trees was improved via the application of a pre-weighted neural network. In short, we examined points in the original scan that were not assigned to one of our study trees. This led to the addition of several net-new trees, the re-attribution of some branches from one tree to another, and the addition of some previously unassigned points to existing trees.

| Column | Description |
|--------|-------------|
| Updated Data | The metrics for isolated trees after the ML refinement. |
| Reported 10/23 | The metrics for isolated trees prior to ML refinement. |
| Diff | The difference in the above two metrics (for trees that are not net-new). |
