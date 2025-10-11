#!/usr/bin/env python
import argparse
import logging
import os
from os.path import exists, join
import open3d as o3d

import numpy as np
# import open3d.ml.torch as ml3d
# import tensorflow as tf
# from open3d.ml.datasets import (
#     KITTI,
#     S3DIS,
#     ParisLille3D,
#     Semantic3D,
#     SemanticKITTI,
#     Toronto3D,
# )
from open3d.io import read_point_cloud as read_pcd
from open3d.ml.vis import LabelLUT, Visualizer
import urllib3
from gui.visualizer import Visualizer
# from gui.min_vizualizer import MinVisualizer
# from util import ensure_demo_data
from tree_isolation import pcds_from_extend_seed_file


def print_usage_and_exit():
    print(
        "Usage: ml-test.py [kitti|semantickitti|paris|toronto|semantic3d|s3dis|custom] path/to/dataset"
    )
    exit(0)


# ------ for custom data -------
kitti_labels = {
    0: 'unlabeled',
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'truck',
    5: 'other-vehicle',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'traffic-sign'
}


# def parse_args():
#     parser = argparse.ArgumentParser(description='Visualize Datasets')
#     parser.add_argument('dataset_name')
#     parser.add_argument('dataset_path')
#     parser.add_argument('--model', default='RandLANet')

#     args = parser.parse_args()

#     return args
from numpy import asarray as arr


def pred_custom_data(pc_names, pcs, pipeline_r, pipeline_k):
    vis_points = []
    for i, data in enumerate(pcs):
        name = pc_names[i]

        results_r = pipeline_r.run_inference(data)
        pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
        # WARNING, THIS IS A HACK
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label_r[0] = 0

        results_k = pipeline_k.run_inference(data)
        pred_label_k = (results_k['predict_labels'] + 1).astype(np.int32)
        # WARNING, THIS IS A HACK
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label_k[0] = 0

        label = data['label']
        pts = data['point']

        vis_d = {
            "name": name,
            "points": pts,
            "labels": label,
            "pred": pred_label_k,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_randlanet",
            "points": pts,
            "labels": pred_label_r,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_kpconv",
            "points": pts,
            "labels": pred_label_k,
        }
        vis_points.append(vis_d)

    return vis_points

def get_demo_data(pc_names=['000700','000750'], path='data/ml/SemanticKITTI'):

    pc_data = []
    for i, name in enumerate(pc_names):
        pc_path = join(path, 'points', name + '.npy')
        label_path = join(path, 'labels', name + '.npy')
        point = np.load(pc_path)[:, 0:3]
        label = np.squeeze(np.load(label_path))

        data = {
            'point': point,
            'feat': None,
            'label': label,
        }
        pc_data.append(data)

    return pc_data

def get_custom_data(file_name=None,
                    path='',
                        pcds = [],
                        ext = '.pkl',
                        combine=True,
                        idxs = []):

    pc_data = []
    colors = None
    if pcds == []:
        pts,labels, orders = pcds_from_extend_seed_file(f'{path}/{file_name}{ext}',return_pcds=False,pcd_idxs=idxs)
    else:
        labels,pts, orders = zip(*[(idx,arr(pcd.points),None) for idx,pcd in enumerate(pcds)])
        colors = zip(*[(idx,arr(pcd.colors),None) for idx,pcd in enumerate(pcds)])

    for pt_list, label, order in zip(pts,labels, orders):
        # point = pt_list
        # label = label
        data = {
            'name': str(label),
            'points': arr(pt_list).astype(np.float32),
            # 'point_attr1': order,
            'label': label,
            
        }
        if colors is not None:
            data['colors'] = arr(colors)
        pc_data.append(data)
    return pc_data

def pcd_to_dataset(pcd_list):
    dataset = []
    for pcd in pcd_list:
        dataset.append(get_custom_data(pcd))
    return dataset


    vis_points = []
    for i, data in enumerate(pcs):
        name = pc_names[i]

        results_r = pipeline_r.run_inference(data)
        pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
        # WARNING, THIS IS A HACK
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label_r[0] = 0

        results_k = pipeline_k.run_inference(data)
        pred_label_k = (results_k['predict_labels'] + 1).astype(np.int32)
        # WARNING, THIS IS A HACK
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label_k[0] = 0

        label = data['label']
        pts = data['point']

        vis_d = {
            "name": name,
            "points": pts,
            "labels": label,
            "pred": pred_label_k,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_randlanet",
            "points": pts,
            "labels": pred_label_r,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_kpconv",
            "points": pts,
            "labels": pred_label_k,
        }
        vis_points.append(vis_d)

    return vis_points
label_names = {0:'zero', 1: 'one', 2: 'two', 3: 'three'}

label_lut = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten'
}

def draw_ui(file=None,pcd_list=None,path='',idxs=[]):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    
    if file is not None:
        data = get_custom_data(file,path=path,idxs=idxs)
    elif pcd_list is not None:
        data = get_custom_data(pcds=pcd_list)
    else:
        raise ValueError("Either file or pcd_list must be provided")
    

    # which = args.dataset_name.lower()
    # path = args.dataset_path
    # data= get_custom_data(file,path=path,idxs=idxs)
    v = Visualizer()
    # lut = LabelLUT()
    # for val in sorted([X['label'] for X in data]):
    #     lut.add_label(label_lut[val], val)
    # v.set_lut("label", lut)
    breakpoint()
    v.visualize(data)   

def get_kitti_data(pc_names, path):

    pc_data = []
    for i, name in enumerate(pc_names):
        pc_path = join(path, 'points', name + '.npy')
        label_path = join(path, 'labels', name + '.npy')
        point = np.load(pc_path)[:, 0:3]
        label = np.squeeze(np.load(label_path))
        breakpoint()
        data = {
            'point': point,
            'feat': None,
            'label': label,
        }
        pc_data.append(data)

    return pc_data


from open3d.ml.torch.models import KPFCNN, RandLANet
from open3d.ml.torch.pipelines import SemanticSegmentation

def draw_kitti():
    num_scenes =10
    pc_names = ["{:05d}".format(i) for i in range(num_scenes)]
    pc_names = ["000700","000750"]

    test = get_kitti_data(pc_names, "data/ml/SemanticKITTI")
    v = Visualizer()
    
    lut = LabelLUT()
    for val in sorted(kitti_labels.keys()):
        lut.add_label(kitti_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    ckpt_path = "data/ml/dataset/checkpoints/vis_weights_{}.pth".format(
        'RandLANet')


    ckpt_path = "data/ml/dataset/checkpoints/vis_weights_{}.pth".format(
        'RandLANet')
    model = RandLANet(ckpt_path=ckpt_path)
    pipeline_r = SemanticSegmentation(model)
    pipeline_r.load_ckpt(model.cfg.ckpt_path)

    ckpt_path = "data/ml/dataset/checkpoints/vis_weights_{}.pth".format('KPFCNN')
    model = KPFCNN(ckpt_path=ckpt_path, in_radius=10)
    pipeline_k = SemanticSegmentation(model)
    pipeline_k.load_ckpt(model.cfg.ckpt_path)

    pcs_with_pred = pred_custom_data(pc_names, test, pipeline_r, pipeline_k)
    

    v.visualize(pcs_with_pred)


if __name__ == "__main__":
    import glob
    # breakpoint()
    collective_ml_path = '../TreeLearn/remote_data/collective/pipeline/'
    pt_wise = f'{collective_ml_path}/pointwise_results'
    pt_wise = f'{collective_ml_path}/pointwise_results'
    detail_files = glob('*',root_dir=collective_ml_path)
    
    pcds = [read_pcd('data/skio/ext_detail/full_ext_seed107_rf1_orig_detail.pcd')]
    # draw_kitti()
    skio_seeds = 'cluster_roots_w_order_in_process'
    skio_path ='data/skio/exts'
    idxs = [0,1,2,3]
    skeletor_clusters = 'skel_w_order_complete'
    skeletor_path ='data/skeletor/inputs'
    draw_ui(file = skeletor_clusters, path=skio_path,idxs=idxs)

 