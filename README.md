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
- **Surface geometry.reconstruction**: Mesh generation and processing

### 📊 Analysis & Visualization
- **Canopy Metrics**: Comprehensive tree structure analysis
- **3D Visualization**: Interactive point cloud and mesh visualization using Open3D
- **Color Analysis**: HSV-based foliage classification and analysis
- **Ray Casting**: Advanced geometric analysis and projection methods
- **UI Interface**: GUI components for interactive data exploration

## Architecture

```
pyQSM/
├── pyQSM/                                # Main source code
│   ├── geometry/                       # Geometric processing modules
│   │   ├── skeletonize.py              # Skeleton/Wireframe extraction algorithms
│   │   ├── point_cloud_processing.py   # Point cloud utilities
│   │   ├── zoom.py          # Bounded filtering utilities
│   │   └── mesh_processing.py          # Triangle Mesh manipulation
│   ├── math/                          # Utility functions
│   │   ├── fit.py                      # RANSAC, DBSCAN, etc. clustering
│   │   └── math_utils.py               # Basic math (i.e. finding center, percentiles)
│   ├── utils/                          # Utility functions
│   │   ├── io.py                       # Input/output operations
│   │   ├── logging_utils.py            # Logging helper functions
│   │   ├── plotting.py                 # Matplotlib scatter plots, histograms, etc.
│   │   └── lib_integration.py          # External library integration
│   ├── viz/                            # Visualization modules
│   │   ├── viz_utils.py                # Open3D visualization utilities
│   │   ├── color.py                    # Color analysis and mapping
│   │   └── tables.py                   # for outputing tabular data
│   ├── tree_isolation.py               # Tree isolation algorithms
│   ├── qsm_generation.py               # QSM generation pipeline
│   ├── canopy_metrics.py               # Tree analysis metrics
│   ├── geometry.reconstruction.py               # 3D geometry.reconstruction tools
│   └── ray_casting.py                  # Ray casting operations
├── scripts/                            # Processing scripts and workflows
├── data/                               # Data storage and examples
└── requirements.txt                    # Dependencies
```

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

## Research Applications

pyQSM is designed for:
- **Tree Canopy Segmentation**: Automated identification of individual trees and epipytes there-in

- **Environmental Simulation**: Triangle meshes can be used along side ray-casting to simulate different angles of sunlight, cloud cover and rain angle 

## License

This project is licensed under the Mozilla Public License Version 2.0. See [LICENSE](LICENSE) for details.

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
