import argparse
from typing import Dict, Union
import os
import numpy as np
import h5py
from glob import glob
from pointsmap import *
from h5datacreator import *

from .io import PlyData
from .constant import *

def create_semanticsmap(config:Dict[str, Union[str]], dst_h5:H5Dataset):
    search_path = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_3d_semantics', config[CONFIG_SEQUENCE_DIR], 'static', '*.ply')
    poly_paths = glob(search_path)

    dynamic_labels:np.ndarray = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33], dtype=np.uint8)
    
    points = Points(quiet=True)

    for ply_path in poly_paths:
        ply:PlyData = PlyData(ply_path)
        ply_data:np.ndarray = ply.data[PLY_ELEMENT_VERTEX]
        ply_points:np.ndarray = np.stack([ply_data['x'], ply_data['y'], ply_data['z']], axis=1)
        ply_semantic1d:np.ndarray = np.uint8(ply_data['semantic'])
        
        dynamic_mask:np.ndarray = np.isin(ply_semantic1d, dynamic_labels, invert=True)

        tmp_points = Points(quiet=True)
        tmp_points.set_semanticpoints(ply_points[dynamic_mask], ply_semantic1d[dynamic_mask])
        tmp_points.downsampling(VOXELGRIDFILTER_LEAFSIZE)
        ply_points, ply_semantic1d = tmp_points.get_semanticpoints()

        points.add_semanticpoints(ply_points, ply_semantic1d)

    # points.downsampling(VOXELGRIDFILTER_LEAFSIZE)

    points_np, semantic1d_np = points.get_semanticpoints()

    map_group = dst_h5.get_common_group('map')
    set_semantic3d(map_group, 'map', points_np, semantic1d_np, FRAMEID_WORLD, LABEL_TAG, map_id='Seq{0:04d}'.format(config[CONFIG_SEQUENCE]))

def create_labelconfig(dst_h5:H5Dataset):
    label_group = dst_h5.get_label_group(LABEL_TAG)
    set_label_config(label_group,  0, 'unlabeled'           ,   0,   0,   0)
    set_label_config(label_group,  1, 'ego vehicle'         ,   0,   0,   0)
    set_label_config(label_group,  2, 'rectification border',   0,   0,   0)
    set_label_config(label_group,  3, 'out of roi'          ,   0,   0,   0)
    set_label_config(label_group,  4, 'static'              ,   0,   0,   0)
    set_label_config(label_group,  5, 'dynamic'             , 111,  74,   0)
    set_label_config(label_group,  6, 'ground'              ,  81,   0,  81)
    set_label_config(label_group,  7, 'road'                , 128,  64, 128)
    set_label_config(label_group,  8, 'sidewalk'            , 244,  35, 232)
    set_label_config(label_group,  9, 'parking'             , 250, 170, 160)
    set_label_config(label_group, 10, 'rail track'          , 230, 150, 140)
    set_label_config(label_group, 11, 'building'            ,  70,  70,  70)
    set_label_config(label_group, 12, 'wall'                , 102, 102, 156)
    set_label_config(label_group, 13, 'fence'               , 190, 153, 153)
    set_label_config(label_group, 14, 'guard rail'          , 180, 165, 180)
    set_label_config(label_group, 15, 'bridge'              , 150, 100, 100)
    set_label_config(label_group, 16, 'tunnel'              , 150, 120,  90)
    set_label_config(label_group, 17, 'pole'                , 153, 153, 153)
    set_label_config(label_group, 18, 'polegroup'           , 153, 153, 153)
    set_label_config(label_group, 19, 'traffic light'       , 250, 170,  30)
    set_label_config(label_group, 20, 'traffic sign'        , 220, 220,   0)
    set_label_config(label_group, 21, 'vegetation'          , 107, 142,  35)
    set_label_config(label_group, 22, 'terrain'             , 152, 251, 152)
    set_label_config(label_group, 23, 'sky'                 ,  70, 130, 180)
    set_label_config(label_group, 24, 'person'              , 220,  20,  60)
    set_label_config(label_group, 25, 'rider'               , 255,   0,   0)
    set_label_config(label_group, 26, 'car'                 ,   0,   0, 142)
    set_label_config(label_group, 27, 'truck'               ,   0,   0,  70)
    set_label_config(label_group, 28, 'bus'                 ,   0,  60, 100)
    set_label_config(label_group, 29, 'caravan'             ,   0,   0,  90)
    set_label_config(label_group, 30, 'trailer'             ,   0,   0, 110)
    set_label_config(label_group, 31, 'train'               ,   0,  80, 100)
    set_label_config(label_group, 32, 'motorcycle'          ,   0,   0, 230)
    set_label_config(label_group, 33, 'bicycle'             , 119,  11,  32)
    set_label_config(label_group, 34, 'garage'              ,  64, 128, 128)
    set_label_config(label_group, 35, 'gate'                , 190, 153, 153)
    set_label_config(label_group, 36, 'stop'                , 150, 120,  90)
    set_label_config(label_group, 37, 'smallpole'           , 153, 153, 153)
    set_label_config(label_group, 38, 'lamp'                ,   0,  64,  64)
    set_label_config(label_group, 39, 'trash bin'           ,   0, 128, 192)
    set_label_config(label_group, 40, 'vending machine'     , 128,  64,   0)
    set_label_config(label_group, 41, 'box'                 ,  64,  64, 128)
    set_label_config(label_group, 42, 'unknown construction', 102,   0,   0)
    set_label_config(label_group, 43, 'unknown vehicle'     ,  51,   0,  51)
    set_label_config(label_group, 44, 'unknown object'      ,  32,  32,  32)
    set_label_config(label_group, -1, 'license plate'       ,   0,   0, 142)

def create_sequential_data(config:Dict[str, Union[str]], dst_h5:H5Dataset):
    data_group:h5py.Group = dst_h5.get_next_data_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-root-dir', type=str, metavar='PATH', required=True, help='Root directory of KITTI-360 Dataset.')
    parser.add_argument('-o', '--output-dir', type=str, metavar='PATH', required=True, help='Output Directory.')
    parser.add_argument('-s', '--sequence', type=int, choices=[0,2,3,4,5,6,7,9,10], required=True, help='Sequence of KITTI-360 Dataset.')
    args = parser.parse_args()
    config:Dict[str, Union[str, int]] = {}
    config[CONFIG_DATASET_ROOT_DIR] = args.dataset_root_dir
    config[CONFIG_SEQUENCE] = args.sequence
    config[CONFIG_SEQUENCE_DIR] = '2013_05_28_drive_{0:04d}_sync'.format(config[CONFIG_SEQUENCE])
    if os.path.isdir(args.output_dir) is False:
        raise NotADirectoryError('"{0}" is not a directory.'.format(args.output_dir))
    config[CONFIG_HDF5_PATH] = os.path.join(args.output_dir, 'kitti360_seq{0:02d}.hdf5'.format(config[CONFIG_SEQUENCE]))
    print(config)
    h5file:H5Dataset = H5Dataset(config[CONFIG_HDF5_PATH])
    create_sequential_data(config, h5file)
    create_semanticsmap(config, h5file)
    create_labelconfig(h5file)
    h5file.close()

if __name__=='__main__':
    main()
