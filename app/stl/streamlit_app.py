 
from collections import namedtuple
# import altair as alt
import open3d as o3d
# import pandas as pd
import streamlit as st
import numpy as np
import pyvista as pv
from stpyvista import stpyvista

import logging

log = logging.getLogger(__name__)

"""
# Welcome to Steamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.>
forums](https://discuss.streamlit.io).
(
In the meantime, below is an example of what you can do with just a few lines o>
"""
def get_angles(tup, radians=False):
    """Gets the angle of a vector with the XY axis"""
    a = tup[0]
    b = tup[1]
    c = tup[2]
    denom = np.sqrt(a**2 + b**2)
    if denom != 0:
        radians = np.arctan(c / np.sqrt(a**2 + b**2))
        if radians:
            return radians
        else:
            return np.degrees(radians)
    else:
        return 0

def filter_by_norm(pcd, angle_thresh=10):
    norms = np.asarray(pcd.normals)
    angles = np.apply_along_axis(get_angles, 1, norms)
    angles = np.degrees(angles)
    stem_idxs = np.where((angles > -angle_thresh) & (angles < angle_thresh))[0]
    stem_cloud = pcd.select_by_index(stem_idxs)
    return stem_cloud

def pcd_to_stem(pcd):
   pts  = np.asarray(pcd.points)
   log.debug(f'len pts {len(pts)}')
   zmin = 0
   try:
    zmin = np.min(pts[:,2])
   except Exception as e:
    log.error('exception getting min z')
    log.error(e)
    log.error(pcd)
   min_mask = np.where(pts[:, 2] <= (zmin+.4))[0]
   pcd_minus_ground = pcd.select_by_index(min_mask, invert=True)
   pcd_minus_ground.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
   pcd_minus_ground.normalize_normals()
   stem_cloud = filter_by_norm(pcd_minus_ground,40)
   return stem_cloud

def plotter_add_pcd(pcd, plotter, **kwargs):
    np_pcd = np.asarray(pcd.points)

    x, y, z = np_pcd.T
    plotter.add_mesh(np_pcd,
        scalars = z,
        **kwargs
    )
    log.error('added pcd to plotter')

if __name__ == '__main__':
    pv.start_xvfb()
    log.debug('started')
    pcd = o3d.io.read_point_cloud("27_vox_pt02_sta_6-4-3.pcd")
    log.error(f'{pcd}')
    stem_cloud = pcd_to_stem(pcd)
    log.debug('starting draw code')
    #initialize a plotter object
    plotter = pv.Plotter(window_size=[400, 400])
    plotter_add_pcd(stem_cloud, plotter,
            # cmap="prism",
            show_edges=True,
            edge_color="#001100",
            ambient=0.2)
    ## Some final touches
    plotter.background_color = "white"
    plotter.view_isometric()
    ## Pass a plotter to stpyvista
    log.error('attempting stpyvista')
    stpyvista(plotter)
# with st.echo(code_location='below'):
#    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
#    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

#    Point = namedtuple('Point', 'x y')
#    data = []

#    points_per_turn = total_points / num_turns

#    for curr_point_num in range(total_points):
#       curr_turn, i = divmod(curr_point_num, points_per_turn)
#       angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#       radius = curr_point_num / total_points
#       x = radius * math.cos(angle)
#       y = radius * math.sin(angle)
#       data.append(Point(x, y))

#    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#       .mark_circle(color='#0068c9', opacity=0.5)
#       .encode(x='x:Q', y='y:Q'))
