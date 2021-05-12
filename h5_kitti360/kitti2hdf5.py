import argparse
from typing import Dict, Union
import os
import numpy as np
import h5py
from glob import glob
from h5datacreator import *

from .io import PlyData
from .constant import *

def create_semanticsmap(config:Dict[str, Union[str]]):
    search_path = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_3d_semantics', config[CONFIG_SEQUENCE_DIR], 'static', '*.ply')
    poly_paths = glob(search_path)
    
    points:np.ndarray = np.empty((0, 3), dtype=np.float32)
    semantic1d:np.ndarray = np.empty((0), dtype=np.uint8)

    for ply_path in poly_paths:
        ply:PlyData = PlyData(ply_path)
        ply_data:np.ndarray = ply.data[PLY_ELEMENT_VERTEX]
        ply_points:np.ndarray = np.stack([ply_data['x'], ply_data['y'], ply_data['z']], axis=1)
        ply_semantic1d:np.ndarray = np.uint8(ply_data['semantic'])

        points = np.append(points, ply_points, axis=0)
        semantic1d = np.append(semantic1d, ply_semantic1d, axis=0)

    print(points.shape)
    print(semantic1d.shape)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-root-dir', type=str, metavar='PATH', required=True, help='Root directory of KITTI-360 Dataset.')
    parser.add_argument('-s', '--sequence', type=int, choices=[0,2,3,4,5,6,7,9,10], required=True, help='Sequence of KITTI-360 Dataset.')
    args = parser.parse_args()
    config:Dict[str, Union[str, int]] = {}
    config[CONFIG_DATASET_ROOT_DIR] = args.dataset_root_dir
    config[CONFIG_SEQUENCE] = args.sequence
    config[CONFIG_SEQUENCE_DIR] = '2013_05_28_drive_{0:04d}_sync'.format(config[CONFIG_SEQUENCE])
    print(config)
    create_semanticsmap(config)

if __name__=='__main__':
    main()
