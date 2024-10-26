##sphere returning stem and first
## couple of orders of branches.
##  stem relegated to trunk
# pcd clean
neighbors = 5
ratio = 6
iters = 3
voxel_size = None
# stem
angle_cutoff = 20
# stem_voxel_size = 0.04 ##none used. if used, trunk is shorter, less branches found
# trunk
num_lowest = 500
trunk_neighbors = 10
trunk_ratio = 0.25
# sphere
min_sphere_radius = 0.1
max_radius = 0.5
radius_multiplier = 2
dist = 0.07

###Variant
##pcd clean
# moving to 1 iteration lead to alot more of the tree being map
#   alot of noise, 30 branches, apparently longer stem?

# stem_voxel_size = 0.04 ##none used. if used, trunk is shorter, less branches found

# Changing the angle cutoff to 10 leads to a super short stem

# lowering the min radius lead to far fewer branches found
# increasing the max radius to .8 didnt do much


# sphere radius
# lowering the radius multiplier to 1.25 leads to a bit more sensitivity towards assigning lower branch orders

# clustering
#  doubling dist leads to low branch order sensitivity, less branches found
