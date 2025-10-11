import os 
import subprocess
import sys
import pickle 
import open3d as o3d
import numpy as np
from set_config import log
from numpy import array as arr
from itertools import product
from prettytable import PrettyTable

# from open3d.io import read_point_cloud

use_super_user = config['io']['super_user']
data_root = config['io']['data_root']

# def read_pcd(file, root_dir = data_root):
#     pcd = read_point_cloud
#     return pcd

def save_line_set(line_set, base_file = 'skel_stem20_topology',root_dir=data_root):
    base_file.replace('.pkl','')
    save(f'{base_file}_lines.pkl', arr(line_set.lines),root_dir)
    save(f'{base_file}_points.pkl', arr(line_set.points),root_dir)
    
def load_line_set(base_file = 'skel_stem20_topology',root_dir=data_root):
    base_file.replace('.pkl','')
    lines = load(f'{base_file}_lines.pkl',root_dir)
    points = load(f'{base_file}_points.pkl',root_dir)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

def be_root():
    # if os.geteuid() == 0:
    return 1
    # else:
        # subprocess.call(['sudo', 'python3'] + sys.argv)  # modified

def update(file, to_write, root_dir = data_root):
    # be_root()
    curr = load(file,root_dir)
    curr.extend(to_write)
    save(file,curr,root_dir)

def save(file, to_write, root_dir = data_root):
    # be_root()
    if '.pkl' not in file: file=f'{file}.pkl'
    fqp = f'{root_dir}{file}'

    with open(fqp,'wb') as f:
        pickle.dump(to_write,f)

def load(file, root_dir = data_root):
    # be_root()
    if '.pkl' not in file: file=f'{file}.pkl'
    fqp = f'{root_dir}{file}'

    with open(fqp,'rb') as f:
        ret = pickle.load(f)
    return ret

def create_table(
                results:list[dict] | list[tuple[str,dict[str,dict]]] | dict[str,dict[str,dict]]
                 ,cols=None,sub_cols=None,ids=None):
    if cols is None:
        cols = [x for x in results[0].keys()]
    if sub_cols is None:
        try:
            sub_cols = results[0][cols[0]].keys()
        except:
            log.info(f'no sub_cols found for {cols[0]}')
            sub_cols = results[0].keys()
    all_cols =list([x for x in product(cols, sub_cols)])

    col_names = [f'{col}_{sub_col}' for col, sub_col in all_cols]
    fin_cols = ['ID'] + col_names
    myTable = PrettyTable(fin_cols)

    if ids is not None:
        if isinstance(results[0],tuple):
            ids = [x[0] for x in results]
            rows = [x[1] for x in results]
            results = dict(zip(ids,rows))
        elif isinstance(results,dict):
            ids = [x for x in results.keys()]
        elif isinstance(results,list):
            results = dict([(idx,x) for idx,x in enumerate(results)])

    for row_id, row_data in results.items():
        row = [row_data[col] for col in col_names]
        if sub_cols is not None:
            row = [row[col] for col in sub_cols]
        myTable.add_row([row_id] + row)
    print(myTable)
    return myTable

def read_in_pcd_in_parts(file_prefix = 'data/input/SKIO/part_skio_raffai',
                    return_pcd_list =False,
                    return_single_pcd = False,
                    return_list = None,
                    return_lambda = None):
    """loops over a list of files, reading in each as a point cloud to return as a list.
        Used to read in the skio scan which is broken into grid blocks for processing.
        . reads in the gridded sections 

    Args:
        file_prefix (str, optional): _description_. Defaults to 'data/input/SKIO/part_skio_raffai'.
        return_pcd_list (bool, optional): _description_. Defaults to False.
        return_single_pcd (bool, optional): _description_. Defaults to False.
        return_list (_type_, optional): _description_. Defaults to None.
        return_lambda (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if return_lambda: return_list = []
    else:
        if not return_single_pcd and not return_pcd_list:
            return None, None, None 
    extents = {}
    pcds = []
    pts = []
    colors = []
    contains_region= []
    base = 20000000

    for i in range(40):
        num = base*(i+1) 
        file = f'{file_prefix}_{num}.pcd'
        pcd = read_point_cloud(file)
        if return_lambda:
            return_list.append(return_lambda(pcd))
        if return_pcd_list:
            pcds.append(pcd)
        elif return_single_pcd:
            pts.extend(list(arr(pcd.points)))   
            colors.extend(list(arr(pcd.colors)))
    if return_single_pcd:
        collective = o3d.geometry.PointCloud()
        collective.points = o3d.utility.Vector3dVector(pts)
        collective.colors = o3d.utility.Vector3dVector(colors)   
        pcds.append(pcd)     
    return return_list, contains_region, pcds
    