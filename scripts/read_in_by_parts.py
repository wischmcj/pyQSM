import numpy as np
import copy
import open3d as o3d
from open3d.io import read_point_cloud, write_point_cloud
from decimal import Decimal
from itertools import islice
import fileinput

line_reached =0
def read_file_sections(start_line):
    lines_per_file = 20000000
    smallfile = None
    max_pcd_pts = 3000000
    # test = True
    factor=10
    x_min, x_max= 85.13300323-factor, 139.057+factor
    y_min, y_max= 338.26-factor, 379.921+factor
    with open('/home/penguaman/code/Research/lidar/SKIO-RaffaiEtAl.pts') as bigfile:
        i=0
        pts = []
        colors = []
        pcd_pts =  []
        pcd_cols = []
        # for lineno, line in enumerate(bigfile):
        for lineno, line in enumerate(islice(bigfile, start_line, None)):
            # if test:
            #     breakpoint()
            #     print('at breakpoint')
            # line  = ' '.join(line.split(' ')[:3]) + '\n'
            line = line.replace('\n','').split(' ')
            try:
                pt = tuple(map(float,line[:3]))
            #     worked = False
                if (pt[0]>x_min and
                    pt[0]<x_max and
                    pt[1]>y_min and
                    pt[1]<y_max      
                    ):
                    try:
                        pcd_pts.append(pt)
                        color = tuple(map(float,line[3:6]))
                        pcd_cols.append(color)
                        worked=True
                    except:
                        print('err ')
            except Exception as e:
                print(f'err {e}')

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)
        pcd.colors =  o3d.utility.Vector3dVector(pcd_cols)
        # o3d.visualization.draw(pcd)
        # pcd = pcd.voxel_down_sample(.05)
        write_point_cloud('test_region.pcd',pcd)
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
                ,_,_,_,r,g,b = line.split(' ')
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
    read_file_sections(0)
    skeletor = "/code/code/Research/lidar/converted_pcs/skeletor_xyzrgb.pts"
    pts_to_pcd_rbg(skeletor)
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