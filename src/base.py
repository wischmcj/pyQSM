from collections import defaultdict
import logging
import open3d as o3d
import rustworkx as rx


class Qsm():

class Qsm():


class Tree(object):
    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()  
    pcdeton: o3d.geometry.PointCloud = o3d.geometry.PointCloud()               
    graph: rx.Graph = rx.Graph()                 
    qsm: list[o3d.geometry.PointCloud] = o3d.geometry.PointCloud()
    qsm_graph: rx.Graph = rx.Graph()
    branches: defaultdict(list) = []

    def __init__(self, pcd, verbose: bool = False, debug: bool = False):
        self.verbose: bool = verbose
        self.debug: bool = debug

        if self.verbose:
            logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        self.pcd = pcd

    def identify_densities(self):
        '''
        Extract pcdeton from point cloud

        :return:
        '''
        pass

    def process(self):
        pcd = self.pcd
        find_lower_order_branches(pcd)

    def animate_draw(self,
                     init_rot: np.ndarray = np.eye(3),
                     steps: int = 360,
                     point_size: float = 1.0,
                     output: [str, None] = None):
        """
            Creates an animation of a point cloud. The point cloud is simply rotated by 360 Degree in multpile steps.

            :param init_rot: Inital rotation to align pcd for visualization
            :param steps: animation rotates 36o degree and is divided into #steps .
            :param point_size: point size of point cloud points.
            :param output_folder: folder where the rendered images are saved to. If None, no images will be saved.

            :return:
        """
        output_folder = os.path.join(output, './tmp_{}'.format(time.time_ns()))
        os.mkdir(output_folder)

        pcd = copy(self.pcd)
        pcd.paint_uniform_color([0, 0, 1])
        pcd.rotate(init_rot, center=[0, 0, 0])

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        vis.add_geometry(pcd)

        ctl = vis.get_view_control()
        ctl.set_zoom(0.6)

        vis.get_render_option().point_size = point_size
        vis.get_render_option().line_width = 15
        vis.get_render_option().light_on = False
        vis.update_renderer()
                            
        Rot_mat = R.from_euler('y', np.deg2rad(360 / steps)).as_matrix()

        image_path_list = []

        pcd_idx = 0

        for i in range(steps):
            pcd.rotate(Rot_mat, center=[0, 0, 0])
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            if ((i % 30) == 0) and i != 0:
                pcd_idx = (pcd_idx + 1) % 2

        vis.destroy_window()
