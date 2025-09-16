import numpy as np
from copy import deepcopy
import open3d as o3d
from open3d.io import read_point_cloud, write_point_cloud
from decimal import Decimal
from itertools import islice
import fileinput
from math import floor

import pickle
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import numpy as np
from numpy import asarray as arr, mean
import scipy.spatial as sps
import  open3d as o3d
from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd

# from utils.io import load
# from viz.viz_utils import color_continuous_map, draw
# from set_config import log, config


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors   
import cv2
from math import floor
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from scipy import stats
def save(file, to_write):
    # be_root()
    if '.pkl' not in file: file=f'{file}.pkl'
    fqp = f'{file}'

    with open(fqp,'wb') as f:
        pickle.dump(to_write,f)

def load(file):
    # be_root()
    if '.pkl' not in file: file=f'{file}.pkl'
    fqp = f'{file}'

    with open(fqp,'rb') as f:
        ret = pickle.load(f)
    return ret

def color_distribution(in_colors,cutoff=.1,elev=40, azim=110, roll=0, 
                space='none',min_s=.2,sat_correction=2,sc_func =lambda sc: sc):
    data = []
    rands = np.random.sample(len(in_colors))
    corrected_rgb_full = in_colors
    in_colors = np.asarray(in_colors)[rands<cutoff]
    in_colors = arr([tuple((int(128+v*128) for v in rgb)) for rgb in in_colors])
    for sids,series in enumerate(data):
        data[sids] = np.asarray(series)[rands<cutoff]

    # data = []
    # hsv = np.asarray(rgb_to_hsv(in_colors))
    # hc,sc,vc = zip(*hsv)
    # sc = arr(sc)
    # vc = arr(vc)
    # # low_saturation_idxs = np.where(sc<min_s)[0]
    # # sc[sc<min_s] = sc[sc<min_s]*sat_correction
    # ret_sc = sc_func(sc)
    # # vc =   sc_func(vc)
    # # sc = sc*.6
    # corrected_rgb_full = np.asarray(hsv_to_rgb([x for x in zip(hc,ret_sc,vc)]))
    
#    15     lower_blue = np.array([110,50,50])
#    16     upper_blue = np.array([130,255,255])
#    17 
#    18     # Threshold the HSV image to get only blue colors
#    19     mask = cv2.inRange(hsv, lower_blue, upper_blue)

#    15     lower_blue = np.array([110,50,50])
#    16     upper_blue = np.array([130,255,255])
#    17 
#    18     # Threshold the HSV image to get only blue colors
#    19     mask = cv2.inRange(hsv, lower_blue, upper_blue)

    ## RGB
    if space=='rgb':
        pixel_colors = in_colors
        r, g, b = zip(*in_colors)
        # r, g, b = cv2.split(pixel_colors)
        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        axis.scatter(r, g, b, facecolors=in_colors, marker=".")
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
        axis.view_init(elev=elev, azim=azim, roll=roll)
        plt.show()
        breakpoint()

    # HSV 
    if space=='hsv':
        hsv = rgb_to_hsv(in_colors)
        # osc = arr(osc)[rands<cutoff]
        hc,sc,vc = zip(*hsv)
        sc = arr(sc)
        vc = arr(vc)
        # low_saturation_idxs = np.where(sc<min_s)[0]
        # sc[sc<min_s] = sc[sc<min_s]*sat_correction
        ret_sc = sc_func(sc)
        # vc =   sc_func(vc)
        # sc = sc*.6
        corrected_rgb_full = arr(hsv_to_rgb([x for x in zip(hc,ret_sc,vc)]))
        
        # hsv = hsv[hsv[:,1]>min_s]
        hc,sc,vc = zip(*hsv)
        import math
        # sc[sc<.5] = sc[sc<.5]*1.5
        rows = 1+ math.ceil(len(data)/2)
        series = [vc]
        series.extend(data)
        # breakpoint()
        fig = plt.figure(figsize=(12, 9))
        for row in range(rows):
            # row=row+1
            axis = fig.add_subplot(rows, 2, int((row+1)), projection="3d")
            z = series[row]
            # breakpoint()
            axis.scatter(hc, sc, z, facecolors=in_colors, marker=".")
            axis.set_xlabel("Hue")
            axis.set_ylabel("Saturation")
            axis.set_zlabel("Value")
            axis.view_init(elev=elev, azim=azim, roll=roll)

            axis = fig.add_subplot(rows, 2, int(row+2), projection="3d")
            axis.scatter(hc, sc, z, facecolors=corrected_rgb, marker=".")
            axis.set_xlabel("Hue")
            axis.set_ylabel("Saturation")
            axis.set_zlabel("Value")
            axis.view_init(elev=elev, azim=azim, roll=roll)
        plt.show()
    return corrected_rgb_full #,hsv

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
    tcoords = o3d.t.geometry.TriangleMesh.create_coordinate_frame()
    tcoords.translate(pcds_to_draw[0].get_center())
    o3d.visualization.draw_geometries(
            pcds_to_draw,
            # mesh_show_wireframe=True,
            # zoom=0.7,
            # front=[0, 2, 0],
            # lookat=[3, -3, 4],
            # up=[0, -1, 1],
            **kwargs,
        )
    
line_reached =0
def read_file_sections(file,
                       start_line):
    batch = 10000000
    # max_pcd_pts = 3000000
    # test = True
    # factor=10
    # x_min, x_max= 85.13300323-factor, 139.057+factor
    # y_min, y_max= 338.26-factor, 379.921+factor
    debug  = True
    with open(file) as bigfile:
        i=0
        file_num=0
        lines = []
        pts = []
        colors = []
        pcd_pts =  []
        pcd_cols = []
        other_vals = []
        # 87 948 217
        # for lineno, line in enumerate(bigfile):
        for lineno, line in enumerate(islice(bigfile, start_line, None)):
            # if test:
            #     breakpoint()
            #     print('at breakpoint')
            # line  = ' '.join(line.split(' ')[:3]) + '\n'
            if lineno%(batch/5)==0:
                print(f'on iter {lineno}')
            if lineno>0:
                line = line.replace('\n','').split(' ')
                lines.append(line)
            if lineno%batch ==0:
                if lineno>0:
                    import math
                    lines = arr(lines)
                    pcd_cols = lines[:,4:7].astype(int)
                    pcd_pts = lines[:,0:3].astype(float)
                    inten = lines[:,3].astype(int)

                    inten_new_max = np.percentile(inten,100)
                    inten_limited =  np.clip(inten, None,inten_new_max)
                    inten_min = np.min(inten_limited)
                    inten_posi =  inten_limited - inten_min if inten_min<0 else inten_limited + inten_min
                    norm_inten = inten_posi/np.median(inten_posi)


                    norm_pcd_cols = pcd_cols/255
                    new_pcd_cols = (norm_pcd_cols.T*norm_inten).T
                    # new_pcd_cols = new_pcd_cols
                    # pcd.colors =  o3d.utility.Vector3dVector(new_pcd_cols)
                    # o3d.visualization.draw(pcd)


                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pcd_pts)
                    pcd.colors =  o3d.utility.Vector3dVector(new_pcd_cols)
                    pcd = pcd.uniform_down_sample(2)
                    # o3d.visualization.draw(pcd)
                    # new_pcd_cols = new_pcd_cols*
                    # breakpoint()
                    # for idv,vals in enumerate([other_vals[:,0],other_vals[:,1],other_vals[:,2],
                    #                pcd_cols[:,0],pcd_cols[:,1],pcd_cols[:,2] ]):
                    #     oth_desc= stats.describe(np.asarray(vals))
                    #     print(f'{idv}: {oth_desc}')


                    # high_inten_ids = np.where(inten > np.percentile(inten,70))
                    # test = pcd.select_by_index(high_inten_ids)
                    # draw(test)
                    # pcd_cols = arr([tuple((x/255 for x in rgb)) for rgb in pcd_cols])
                    # try:
                    #     o3d.visualization.draw(pcd)
                    # except Exception as e:
                    #     print(f'error in draw {e}' )
                    # breakpoint()
                    
                    write_point_cloud(f'/media/penguaman/code/code/ActualCode/pyQSM/data/skeletor/inputs/trim/skeletor_full_{file_num}.pcd',pcd)
                    print(f'Wrote file {file_num} on iter {lineno}')
                    # save(f'data/skeletor/inputs/skeletor_full_other_{file_num}.pkl',other_vals)
                    # save(f'data/skeletor/inputs/skeletor_full_color_{file_num}.pkl',pcd_cols)
                    lines = []
                    del pcd
                    pcd_pts =  []
                    pcd_cols = []
                    other_vals = []
                    file_num+=1
    

        lines = arr(lines)
        pcd_cols = lines[:,4:7].astype(int)
        pcd_pts = lines[:,0:3].astype(float)
        inten = lines[:,3].astype(int)

        inten_new_max = np.percentile(inten,100)
        inten_limited =  np.clip(inten, None,inten_new_max)
        inten_min = np.min(inten_limited)
        inten_posi =  inten_limited - inten_min if inten_min<0 else inten_limited + inten_min
        norm_inten = inten_posi/np.median(inten_posi)


        norm_pcd_cols = pcd_cols/255
        new_pcd_cols = (norm_pcd_cols.T*norm_inten).T
        # new_pcd_cols = new_pcd_cols
        # pcd.colors =  o3d.utility.Vector3dVector(new_pcd_cols)
        # o3d.visualization.draw(pcd)


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)
        pcd.colors =  o3d.utility.Vector3dVector(new_pcd_cols)
        pcd = pcd.uniform_down_sample(2)
        write_point_cloud(f'/media/penguaman/code/code/ActualCode/pyQSM/data/skeletor/inputs/trim/skeletor_full_final.pcd',pcd)
        print(f'Wrote file {file_num} on iter {lineno}')
        # save(f'data/skeletor/inputs/skeletor_full_other_{file_num}.pkl',other_vals)
        breakpoint()

    
        draw(pcd)
        breakpoint()
        #     if worked:
        #         colors.append(tuple(map(lambda x:round(int(x)/255,5),line[-3:])))
        #         if lineno % lines_per_file == 0:
        #             line_reached = lineno
        #             line_curr = lineno+start_line
        #             print(f'Write reached {line_curr=} ')
        #             try:
        #                 pcd = o3d.geometry.PointCloud()
        #                 pcd.points = o3d.utility.Vector3dVector(pts)
        #                 pcd.colors =  o3d.utility.Vector3dVector(colors)
        #                 # o3d.visualization.draw(pcd)
        #                 pcd = pcd.voxel_down_sample(.05)
        #                 write_point_cloud('3vox_down_skio_raffai_{}.pcd'.format(line_curr),pcd)
        #                 max_bnd= pcd.get_max_bound()
        #                 min_bnd= pcd.get_min_bound()
        #                 print(f'bounded by {min_bnd=},{max_bnd=}')
        #                 pcd_pts.extend(list(np.asarray(pcd.points)))
        #                 pcd_cols.extend(list(np.asarray(pcd.colors)))
        #                 pts = []
        #                 colors = []
        #                 if len(pcd_pts)> max_pcd_pts:
        #                     print(f'Writing points collected so far to pcd')
        #                     pcd = o3d.geometry.PointCloud()
        #                     pcd.points = o3d.utility.Vector3dVector(pcd_pts)
        #                     pcd.colors =  o3d.utility.Vector3dVector(pcd_cols)
        #                     write_point_cloud('3compiled_vox_down_skio_raffai_{}.pcd'.format(line_curr),pcd)
        #                     # o3d.visualization.draw(pcd)
        #                     max_bnd= pcd.get_max_bound()
        #                     min_bnd= pcd.get_min_bound()
        #                     print(f'bounded by {min_bnd=},{max_bnd=}')
        #                     # breakpoint()
        #                     pcd_pts = []
        #                     pcd_cols = []
        #                 # if smallfile:
        #                 #     smallfile.close()
        #                 # small_filename = 'small_file_{}.txt'.format(lineno + lines_per_file)
        #                 # smallfile = open(small_filename, "w")
        #                 i+=1
        #                 print(f'{i}th file')
        #                 if pcd_pts is not None:
        #                     num_pts = len(pcd_pts) 
        #                     print(f'{num_pts} points so far')
        #             except Exception as e:
        #                 # breakpoint()
        #                 print(f'caught err in code {e=}')
        #     # smallfile.write(line)
        # if len(pcd_pts)>0:
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(pcd_pts)
        #     pcd.colors =  o3d.utility.Vector3dVector(pcd_cols)
        #     write_point_cloud('vox_down_skio_raffai_{}.pcd'.format(lineno + lines_per_file),pcd)
        #     o3d.visualization.draw(pcd)
        #     pcd_pts = None
        #     pcd_cols = None

def read_pcd():
# pcd =  read_point_cloud("/mnt/c/Users/wisch/Downloads/SKIO-RaffaiEtAl.pts",'xyz',print_progress=True)
    # pcd =  read_point_cloud("vox_down_skio_raffai_20000000.pcd")
    # print('read')
    # pcd = pcd.voxel_down_sample(.1)
    # print('reduced')
    # write_point_cloud('vox_down_skio_raffai.pcd',pcd)
    # print('written')
    # print(pcd)
    # pcd =  read_point_cloud("vox_down_skio_raffai_30000000.pcd")
    pcd =  read_point_cloud("compiled_vox_down_skio_raffai_80000000.pcd")
    o3d.visualization.draw(pcd)

def pts_to_pcd_rbg(file):
    i = 0
    with fileinput.input(file,backup='.bkp',inplace=True) as f:
        try:
            for line in f:
                _,_,_,r,g,b = line.split(' ')
                print('{} {} {}'.format(r,g,b),end='\n')
        except Exception as e:
            breakpoint()
            pass

    # with open(file) as f:
    #     for line in read_lines
    # breakpoint()


if __name__ == '__main__':
    #########################
    #   scans start on ground, move vertically
    #   vox_down_skio_raffai_30000000 has trunks of a couple of trees
    # compiled_vox_down_skio_raffai_560000000 is a good tree iso file, trees rather already seperated 
    ###########################
    # read_pcd()
    # skeletor = "/code/code/Research/lidar/converted_pcs/skeletor_xyzrgb.pts"
    # pts_to_pcd_rbg(skeletor)

    # file = '/home/penguaman/code/Research/lidar/SKIO-RaffaiEtAl.pts'
    file = '/media/penguaman/code/code/ActualCode/pyQSM/data/skeletor/inputs/skeletor.pts'
    file = '/media/penguaman/code/code/ActualCode/pyQSM/data/skeletor/SkeletorTrim.xyz'
    read_file_sections(file, 0)
    # read_pcd()
    # read_file_sections(0)
    # for i in range(10):
    #     try:
    #         read_file_sections(line_reached)
    #     except Exception as e:
    #         print(e)
    # pcd =  read_point_cloud("compiled_vox_down_skio_raffai_60000000.pcd")
    # o3d.visualization.draw(pcd)
    # breakpoint()