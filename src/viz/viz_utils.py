from copy import deepcopy
import open3d as o3d
import numpy as np
import os
import scipy.spatial as sps
from open3d.visualization import draw_geometries
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
s27d = "s32_downsample_0.04.pcd"


def iter_draw(idxs_list, pcd):
    pcds = []
    for idxs in idxs_list:
        if len(idxs) > 0:
            pcds.append(pcd.select_by_index(idxs))
    print(f"iterdraw: drawing {len(pcds)} pcds")
    pcd.paint_uniform_color([0, 0, 0.2])
    colors = [
        mcolors.to_rgb(plt.cm.Spectral(each)) for each in np.linspace(0, 1, len(pcds))
    ]

    for idx, sub_pcd in enumerate(pcds):
        sub_pcd.paint_uniform_color(colors[idx])

    # o3d.visualization.draw_geometries([stem_cloud]+pcds)
    o3d.visualization.draw_geometries(pcds)
    return pcd


def draw(pcds, raw=True, side_by_side=False, **kwargs):
    if (not(isinstance(pcds, list))
        and not(isinstance(pcds, np.ndarray))):
        pcds_to_draw = [pcds]
    else:
        pcds_to_draw = pcds
    if side_by_side:
        trans = 0
        pcds_to_draw = []
        for pcd in pcds:
            to_draw = deepcopy(pcd)
            to_draw.translate([trans,0,0])
            pcds_to_draw.append(to_draw)
            min_bound = to_draw.get_axis_aligned_bounding_box().get_min_bound()
            max_bound = to_draw.get_axis_aligned_bounding_box().get_max_bound()
            bounds = max_bound - min_bound
            trans+=bounds[0]
    #below config used for main dev
    # tree, Secrest27
    draw_geometries(
            pcds_to_draw,
            # mesh_show_wireframe=True,
            # zoom=0.7,
            # front=[0, 2, 0],
            # lookat=[3, -3, 4],
            # up=[0, -1, 1],
            **kwargs,
        )

def cdraw(pcd, 
          render_option_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json(render_option_path)
    vis.run()
    vis.destroy_window()

def color_continuous_map(pcd, cvar):
    density_colors = plt.get_cmap('plasma')((cvar - cvar.min()) / (cvar.max() - cvar.min()))
    density_colors = density_colors[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(density_colors)
    return pcd


def rotating_compare_gif(transient_pcd, constant_pcd_in,
                               init_rot: np.ndarray = np.eye(3),
                               steps: int = 360,
                               transient_period: int = 45,
                               point_size: float = 1.0,
                               output = '/code/code/pyQSM/test/',
                               rot_center = [0,0,0],
                               save = False,
                               file_name = 'pcd_compare_animation'):
        """
            Creates a GIF comparing two point clouds. 
        """
        import os 
        import time
        from scipy.spatial.transform import Rotation as R
        import imageio

        output_folder = os.path.join(output, 'tmp')
        print(output_folder)
        # os.mkdir(output_folder)

        # We 

        # Load PCD
        orig = deepcopy(transient_pcd)
        # if not trans_has_color:
        #     orig.rotate(init_rot, center=[0, 0, 0])

        # skel = copy(contracted)
        # skel.paint_uniform_color([0, 0, 1])
        # skel.rotate(init_rot, center=[0, 0, 0])

        constant_pcd = deepcopy(constant_pcd_in)
        # constant_pcd.paint_uniform_color([0, 0, 0])
        constant_pcd.rotate(init_rot, center=[0, 0, 0])

        transient_pcd = deepcopy(orig)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        vis.add_geometry(transient_pcd)
        vis.add_geometry(constant_pcd)

        ctl = vis.get_view_control()
        ctl.set_zoom(0.6)

        # Set smaller point size. Default is 5.0
        vis.get_render_option().point_size = point_size
        vis.get_render_option().line_width = 15
        vis.get_render_option().light_on = False
        vis.update_renderer()

        # Calculate rotation matrix for every step. Must only be calculated once as rotations are added up in the point cloud
        Rot_mat = R.from_euler('y', np.deg2rad(540 / steps)).as_matrix()

        image_path_list = []

        pcd_idx = 0
        for i in range(steps):
            orig.rotate(Rot_mat, center=rot_center)
            # skel.rotate(Rot_mat, center=rot_center)
            constant_pcd.rotate(Rot_mat, center=rot_center)

            if pcd_idx == 0:
                transient_pcd.points = orig.points
                transient_pcd.colors = orig.colors
                transient_pcd.normals = orig.normals
            if pcd_idx == 1:
                # pcd.paint_uniform_color([0, 0, 0])
                transient_pcd.points = constant_pcd.points
                transient_pcd.colors = constant_pcd.colors

            vis.update_geometry(transient_pcd)
            vis.update_geometry(constant_pcd)
            vis.poll_events()
            vis.update_renderer()

            # Draw pcd for 30 frames at a time
            #  remove for 30 between then
            if ((i % transient_period) == 0) and i != 0:
                pcd_idx = (pcd_idx + 1) % 2
                
            current_image_path = f"{output_folder}/img_{i}.jpg"
            if save:
                vis.capture_screen_image(current_image_path)
                image_path_list.append(current_image_path)

        vis.destroy_window()
        images = []
        log.info(f'Creating gif at {output_folder}{file_name}.gif')
        if save:
            for filename in image_path_list:
              images.append(imageio.imread(filename))
            log.info(f'Creating gif at {image_path_list[0]}')
            imageio.mimsave(os.path.join(os.path.dirname(image_path_list[0]), 
                                         '{}.gif'.format(file_name)), 
                                         images, format='GIF')
            

# def draw_w_traj(pcd_or_data_path, 
#                render_option_path= '',
#                camera_trajectory_path):
#     draw_w_traj.index = -1
#     draw_w_traj.trajectory =9
#     o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)

#     draw_w_traj.vis = o3d.visualization.Visualizer()
#     image_path = os.path.join(data_path, 'image')

#     if not os.path.exists(image_path):

#         os.makedirs(image_path)

#     depth_path = os.path.join(test_data_path, 'depth')

#     if not os.path.exists(depth_path):

#         os.makedirs(depth_path)


#     def move_forward(vis):

#         # This function is called within the o3d.visualization.Visualizer::run() loop
#         # The run loop calls the function, then re-render
#         # So the sequence in this function is to:
#         # 1. Capture frame
#         # 2. index++, check ending criteria
#         # 3. Set camera
#         # 4. (Re-render)
#         ctr = vis.get_view_control()
#         glb = draw_w_traj
#         if glb.index >= 0:
#             print("Capture image {:05d}".format(glb.index))
#             depth = vis.capture_depth_float_buffer(False)
#             image = vis.capture_screen_float_buffer(False)
#             plt.imsave(os.path.join(depth_path, '{:05d}.png'.format(glb.index)),
#                        np.asarray(depth),
#                        dpi=1)
#             plt.imsave(os.path.join(image_path, '{:05d}.png'.format(glb.index)),
#                        np.asarray(image),
#                        dpi=1)
#             # vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
#             # vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
#         glb.index = glb.index + 1
#         if glb.index < len(glb.trajectory.parameters):
#             ctr.convert_from_pinhole_camera_parameters(
#                 glb.trajectory.parameters[glb.index], allow_arbitrary=True)
#         else:
#             draw_w_traj.vis.\
#                 register_animation_callback(None)
#         return False


#     vis = draw_w_traj+.vis

#     vis.create_window()

#     vis.add_geometry(pcd)

#     vis.get_render_option().load_from_json(render_option_path)

#     vis.register_animation_callback(move_forward)

#     vis.run()

#     vis.destroy_window()