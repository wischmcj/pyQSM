# pyQSM: Python Quantitative Structural Modeling for TLS LiDAR Data

pyQSM is a Python library for processing Terrestrial Laser Scanning (TLS) LiDAR point clouds and generating Quantitative Structural Models (QSM) of trees. The library provides comprehensive tools for tree isolation, structural analysis, and 3D reconstruction from point cloud data.

## Overview

pyQSM focuses on two primary use cases:
1. **Tree Isolation**: Separating individual trees from surrounding man-made objects and other vegetation
2. **Tree Segmentation**: Isolating and analyzing different parts of trees (trunk, branches, leaves) for structural modeling

## Key Features

### 🌳 Tree Processing Pipeline
- **Point Cloud Preprocessing**: Cleaning, filtering, and statistical outlier removal
- **Tree Isolation**: Advanced algorithms to separate individual trees from complex environments
- **Structural Segmentation**: Automatic identification of trunk, branches, and foliage components
- **Skeletonization**: Extract tree skeletal structure using robust Laplacian-based methods
- **QSM Generation**: Create quantitative structural models with cylindrical approximations

### 🔧 Core Algorithms
- **Skeleton Extraction**: Robust Laplacian-based point cloud skeletonization
- **Clustering**: DBSCAN and K-means clustering for point cloud segmentation
- **Geometric Fitting**: RANSAC-based cylinder and sphere fitting
- **Tree Topology**: Graph-based representation of tree structure
- **Surface Reconstruction**: Mesh generation and processing

### 📊 Analysis & Visualization
- **Canopy Metrics**: Comprehensive tree structure analysis
- **3D Visualization**: Interactive point cloud and mesh visualization using Open3D
- **Color Analysis**: HSV-based foliage classification and analysis
- **Ray Casting**: Advanced geometric analysis and projection methods
- **UI Interface**: GUI components for interactive data exploration


## Installation

### Prerequisites
- Python 3.8+
- CUDA support (optional, for GPU acceleration)

### Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `open3d` - 3D data processing
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `matplotlib` - Plotting and visualization
- `networkx` - Graph processing
- `scikit-learn` - Machine learning algorithms
- `polyscope` - 3D visualization
- `robust_laplacian` - Laplacian mesh processing

### Configuration
The library uses TOML configuration files for algorithm parameters:
- `src/pyqsm_config.toml` - Main configuration file
- Environment variables: `PY_QSM_CONFIG`, `PY_QSM_LOG_CONFIG`

## Usage

### Basic Workflow

```python
import open3d as o3d
from src.qsm_generation import find_low_order_branches
from src.tree_isolation import extend_seed_clusters
from src.geometry.skeletonize import extract_skeleton

# Load point cloud data
pcd = o3d.io.read_point_cloud("tree_scan.pcd")

# Process tree structure
find_low_order_branches(file="tree_scan.pcd", extract_skeleton=True)

# Extract skeleton
skeleton = extract_skeleton(pcd)

# Generate QSM
# ... (detailed workflow in scripts/)
```

### Configuration

Algorithm parameters can be customized in `pyqsm_config.toml`:

```toml
[skeletonize]
moll = 1e-6
n_neighbors = 20
max_iter = 20
init_contraction = 7
init_attraction = 1

[dbscan]
epsilon = 0.1
min_neighbors = 10

[sphere]
min_radius = 0.01
max_radius = 1.5
```

### Processing Scripts

The `scripts/` directory contains ready-to-use processing workflows:

- `tree_isolation_script.py` - Complete tree isolation pipeline
- `tree_iso_from_feature_branch.py` - Feature-based tree isolation
- `visualize_o3d_ml.py` - Machine learning-based visualization

## Key Modules

### Tree Isolation (`tree_isolation.py`)
- **Seed Clustering**: Extend seed clusters using k-NN search
- **Trunk Base Identification**: Automated trunk base detection
- **Building Removal**: Filter out man-made structures

### QSM Generation (`qsm_generation.py`)
- **Sphere Stepping**: Progressive tree structure discovery
- **Cylinder Fitting**: RANSAC-based cylindrical approximation
- **Branch Ordering**: Hierarchical tree structure analysis

### Skeletonization (`geometry/skeletonize.py`)
- **Robust Laplacian**: Point cloud skeleton extraction
- **Topology Extraction**: Graph-based tree topology
- **QSM Conversion**: Convert skeleton to quantitative model

### Visualization (`viz/`)
- **Interactive 3D Rendering**: Real-time point cloud visualization
- **Color Mapping**: Advanced color analysis for foliage classification
- **UI Components**: GUI elements for data exploration

## Data Formats

### Supported Input Formats
- `.pcd` - Point Cloud Data format (primary)
- `.ply` - Polygon File Format
- `.pts` - Point cloud text format

### Output Formats
- `.pcd` - Processed point clouds
- `.pkl` - Serialized Python objects (clusters, trees, etc.)
- `.mesh` - 3D mesh formats
- `.json` - Configuration and metadata

## Research Applications

pyQSM is designed for:
- **Forest Inventory**: Automated tree measurement and analysis
- **Ecological Studies**: Canopy structure and biomass estimation
- **Urban Planning**: Tree health monitoring in urban environments
- **Agricultural Research**: Orchard and plantation analysis

## License

This project is licensed under the Mozilla Public License Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation expectations
- Pull request process

## Citation

If you use pyQSM in your research, please cite:
```
[Publication details to be added when available]
```

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `data/notes/` directory
- Review example workflows in the `scripts/` directory

---

**Note**: This library is under active development. Features and APIs may change between versions. Please check the changelog for breaking changes and updates.
