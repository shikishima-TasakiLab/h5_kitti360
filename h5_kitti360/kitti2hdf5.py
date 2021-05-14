import argparse
from typing import Dict, Union
import os
from datetime import datetime
import cv2
import numpy as np
import h5py
from glob import glob
from pointsmap import *
from h5datacreator import *

from .io import PlyData, SickData, VelodyneData
from .constant import *

def create_semanticsmap(config:Dict[str, Union[str]], dst_h5:H5Dataset):
    search_path = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_3d_semantics', config[CONFIG_SEQUENCE_DIR], 'static', '*.ply')
    poly_paths = glob(search_path)

    dynamic_labels:np.ndarray = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33], dtype=np.uint8)
    
    points = Points(quiet=True)

    for itr, ply_path in enumerate(poly_paths):
        print('SemanticMap {0:010d}'.format(itr))
        print('Ply      :', ply_path)
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
    print('Done')

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

def create_intrinsic(config:Dict[str, Union[str]], dst_h5:H5Dataset):
    intrinsic_group:h5py.Group = dst_h5.get_common_group('intrinsic')
    intrinsic_dict:Dict[str, Dict[str, Union[int, float]]] = {DIR_IMAGE00: {H5_ATTR_FRAMEID: FRAMEID_CAM0}, DIR_IMAGE01: {H5_ATTR_FRAMEID: FRAMEID_CAM1}}

    with open(os.path.join(config[CONFIG_DATASET_ROOT_DIR], DIR_CALIBRATION, 'perspective.txt'), mode='r') as f:
        line:str = f.readline()
        while line:
            values:List[str] = line.split()
            if values[0] == 'S_rect_00:':
                intrinsic_dict[DIR_IMAGE00]['width'] = int(float(values[1]))
                intrinsic_dict[DIR_IMAGE00]['height'] = int(float(values[2]))
            elif values[0] == 'P_rect_00:':
                intrinsic_dict[DIR_IMAGE00]['Fx'] = float(values[1])
                intrinsic_dict[DIR_IMAGE00]['Cx'] = float(values[3])
                intrinsic_dict[DIR_IMAGE00]['Fy'] = float(values[6])
                intrinsic_dict[DIR_IMAGE00]['Cy'] = float(values[7])
            elif values[0] == 'S_rect_01:':
                intrinsic_dict[DIR_IMAGE01]['width'] = int(float(values[1]))
                intrinsic_dict[DIR_IMAGE01]['height'] = int(float(values[2]))
            elif values[0] == 'P_rect_01:':
                intrinsic_dict[DIR_IMAGE01]['Fx'] = float(values[1])
                intrinsic_dict[DIR_IMAGE01]['Cx'] = float(values[3])
                intrinsic_dict[DIR_IMAGE01]['Fy'] = float(values[6])
                intrinsic_dict[DIR_IMAGE01]['Cy'] = float(values[7])
            line = f.readline()
    
    for key, item in intrinsic_dict.items():
        set_intrinsic(intrinsic_group, key, item['Fx'], item['Fy'], item['Cx'], item['Cy'], item['height'], item['width'], item[H5_ATTR_FRAMEID])

def create_static_transforms(config:Dict[str, Union[str]], dst_h5:H5Dataset):
    transforms_group:h5py.Group = dst_h5.get_common_group('tf_static')

    calibration_dir = os.path.join(config[CONFIG_DATASET_ROOT_DIR], DIR_CALIBRATION)
    
    with open(os.path.join(calibration_dir, 'calib_cam_to_pose.txt'), mode='r') as f:
        line:str = f.readline()
        while line:
            values:List[str] = line.split()
            matrix:np.ndarray = np.identity(4, dtype=np.float32)
            if values[0] == 'image_00:':
                matrix[0:3, :] = np.reshape(np.float32(values[1:13]), (3, 4))
                translation, quaternion = matrix2quaternion(matrix)
                set_pose(transforms_group, 'pose_to_cam0', translation, quaternion, FRAMEID_POSE, FRAMEID_CAM0)
            elif values[0] == 'image_01:':
                matrix[0:3, :] = np.reshape(np.float32(values[1:13]), (3, 4))
                translation, quaternion = matrix2quaternion(matrix)
                set_pose(transforms_group, 'pose_to_cam1', translation, quaternion, FRAMEID_POSE, FRAMEID_CAM1)
            elif values[0] == 'image_02:':
                matrix[0:3, :] = np.reshape(np.float32(values[1:13]), (3, 4))
                translation, quaternion = matrix2quaternion(matrix)
                set_pose(transforms_group, 'pose_to_cam2', translation, quaternion, FRAMEID_POSE, FRAMEID_CAM2)
            elif values[0] == 'image_03:':
                matrix[0:3, :] = np.reshape(np.float32(values[1:13]), (3, 4))
                translation, quaternion = matrix2quaternion(matrix)
                set_pose(transforms_group, 'pose_to_cam3', translation, quaternion, FRAMEID_POSE, FRAMEID_CAM3)
            line:str = f.readline()
    
    with open(os.path.join(calibration_dir, 'calib_cam_to_velo.txt'), mode='r') as f:
        values:List[str] = f.readline().split()
        matrix:np.ndarray = np.identity(4, dtype=np.float32)
        matrix[0:3, :] = np.reshape(np.float32(values[0:12]), (3, 4))
        translation, quaternion = matrix2quaternion(matrix)
        translation, quaternion = invertTransform(translation=translation, quaternion=quaternion)
        set_pose(transforms_group, 'cam0_to_velo', translation, quaternion, FRAMEID_CAM0, FRAMEID_VELODYNE)
    
    with open(os.path.join(calibration_dir, 'calib_sick_to_velo.txt'), mode='r') as f:
        values:List[str] = f.readline().split()
        matrix:np.ndarray = np.identity(4, dtype=np.float32)
        matrix[0:3, :] = np.reshape(np.float32(values[0:12]), (3, 4))
        translation, quaternion = matrix2quaternion(matrix)
        set_pose(transforms_group, 'velo_to_sick', translation, quaternion, FRAMEID_VELODYNE, FRAMEID_SICK)

def convert_timestamp(timestamp:str) -> Tuple[int, int]:
    sec_str, nsec_str = timestamp.split('.')
    return int(datetime.strptime(sec_str, '%Y-%m-%d %H:%M:%S').timestamp()), int(nsec_str)

def create_sequential_data(config:Dict[str, Union[str]], dst_h5:H5Dataset):
    data2dRaw_seq_dir:str = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_2d_raw', config[CONFIG_SEQUENCE_DIR])
    data3dRaw_seq_dir:str = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_3d_raw', config[CONFIG_SEQUENCE_DIR])
    data2dSemantic_seq_dir:str = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_2d_semantics', 'train', config[CONFIG_SEQUENCE_DIR])
    dataPoses_seq_dir:str = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_poses', config[CONFIG_SEQUENCE_DIR])

    image00data_paths:List[str] = sorted(glob(os.path.join(data2dRaw_seq_dir, DIR_IMAGE00, DIR_DATA_RECT, '*.png')))
    image01data_paths:List[str] = sorted(glob(os.path.join(data2dRaw_seq_dir, DIR_IMAGE01, DIR_DATA_RECT, '*.png')))
    velodyne_data_paths:List[str] = sorted(glob(os.path.join(data3dRaw_seq_dir, DIR_VELODYNE_POINTS, DIR_DATA, '*.bin')))
    sick_data_paths:List[str] = sorted(glob(os.path.join(data3dRaw_seq_dir, DIR_SICK_POINTS, DIR_DATA, '*.bin')))

    image00_timestamps:List[str]
    with open(os.path.join(data2dRaw_seq_dir, DIR_IMAGE00, 'timestamps.txt'), mode='r') as f:
        image00_timestamps = f.readlines()
    image01_timestamps:List[str]
    with open(os.path.join(data2dRaw_seq_dir, DIR_IMAGE01, 'timestamps.txt'), mode='r') as f:
        image01_timestamps = f.readlines()
    velodyne_timestamps:List[str]
    with open(os.path.join(data3dRaw_seq_dir, DIR_VELODYNE_POINTS, 'timestamps.txt'), mode='r') as f:
        velodyne_timestamps = f.readlines()
    sick_timestamps:List[str]
    with open(os.path.join(data3dRaw_seq_dir, DIR_SICK_POINTS, 'timestamps.txt'), mode='r') as f:
        sick_timestamps = f.readlines()
    oxts_timestamps:List[str]
    with open(os.path.join(dataPoses_seq_dir, DIR_OXTS, 'timestamps.txt'), mode='r') as f:
        oxts_timestamps = f.readlines()
    
    raw_data_dict:Dict[str, Dict[str, Tuple[str, int, int]]] = {}
    for image00_data_path, image01_data_path, velodyne_data_path, sick_data_path, \
        image00_timestamp, image01_timestamp, velodyne_timestamp, sick_timestamp, oxts_timestamp \
        in zip(image00data_paths, image01data_paths, velodyne_data_paths, sick_data_paths, \
            image00_timestamps, image01_timestamps, velodyne_timestamps, sick_timestamps, oxts_timestamps):

        key = str(int(os.path.splitext(os.path.basename(image00_data_path))[0]))

        raw_dataset:Dict[str, Tuple[str, int, int]] = {}

        image00_sec, image00_nsec = convert_timestamp(image00_timestamp)
        raw_dataset[DIR_IMAGE00] = (image00_data_path, image00_sec, image00_nsec)

        image01_sec, image01_nsec = convert_timestamp(image01_timestamp)
        raw_dataset[DIR_IMAGE01] = (image01_data_path, image01_sec, image01_nsec)

        velodyne_sec, velodyne_nsec = convert_timestamp(velodyne_timestamp)
        raw_dataset[DIR_VELODYNE_POINTS] = (velodyne_data_path, velodyne_sec, velodyne_nsec)

        sick_sec, sick_nsec = convert_timestamp(sick_timestamp)
        raw_dataset[DIR_SICK_POINTS] = (sick_data_path, sick_sec, sick_nsec)

        oxts_sec, oxts_nsec = convert_timestamp(oxts_timestamp)
        raw_dataset[DIR_OXTS] = ('', oxts_sec, oxts_nsec)

        raw_data_dict[key] = raw_dataset

    semanticData_paths:List[str] = sorted(glob(os.path.join(data2dSemantic_seq_dir, DIR_SEMANTIC, '*.png')))
    semantic_data_dict:Dict[str, Dict[str, Tuple[str, int, int]]] = {}
    for semanticData_path in semanticData_paths:
        key = str(int(os.path.splitext(os.path.basename(semanticData_path))[0]))
        if key not in raw_data_dict.keys(): continue

        semantic_dataset:Dict[str, Tuple[str, int, int]] = raw_data_dict[key].copy()
        semantic_dataset[DIR_SEMANTIC] = (semanticData_path, 0, 0)

        semantic_data_dict[key] = semantic_dataset

    del raw_data_dict

    pose_data_dict:Dict[str, Dict[str, Union[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray]]]] = {}
    with open(os.path.join(dataPoses_seq_dir, 'poses.txt'), mode='r') as f:
        line:str = f.readline()
        while line:
            values:List[str] = line.split()
            key = values[0]
            if key not in semantic_data_dict.keys():
                line:str = f.readline()
                continue
            pose_dataset:Dict[str, Union[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray]]] = semantic_data_dict[key].copy()
            matrix:np.ndarray = np.identity(4, dtype=np.float32)
            matrix[0:3, :] = np.reshape(np.float32(values[1:13]), (3, 4))
            # translation, quaternion = matrix2quaternion(matrix)
            pose_dataset['world_to_pose'] = matrix2quaternion(matrix)

            pose_data_dict[key] = pose_dataset

            line:str = f.readline()

    del semantic_data_dict

    success:bool = True
    for key, item in pose_data_dict.items():
        if success is True:
            data_group:h5py.Group = dst_h5.get_next_data_group()
        else:
            data_group:h5py.Group = dst_h5.get_current_data_group()

        image00_data_path, image00_sec, image00_nsec = item[DIR_IMAGE00]
        image00_data:np.ndarray = cv2.imread(image00_data_path, cv2.IMREAD_ANYCOLOR)
        if image00_data is None:
            success = False
            continue

        semantic_data_path, _, _ = item[DIR_SEMANTIC]
        semantic_data:np.ndarray = cv2.imread(semantic_data_path, cv2.IMREAD_UNCHANGED)
        if semantic_data is None:
            success = False
            continue

        image01_data_path, image01_sec, image01_nsec = item[DIR_IMAGE01]
        image01_data:np.ndarray = cv2.imread(image01_data_path, cv2.IMREAD_ANYCOLOR)
        if image01_data is None:
            success = False
            continue

        velodyne_data_path, velodyne_sec, velodyne_nsec = item[DIR_VELODYNE_POINTS]
        velodyne_data_instance = VelodyneData(velodyne_data_path)
        velodyne_data_raw:np.ndarray = velodyne_data_instance.data

        sick_data_path, sick_sec, sick_nsec = item[DIR_SICK_POINTS]
        sick_data_instance = SickData(sick_data_path)
        sick_data_raw:np.ndarray = sick_data_instance.data

        oxts_data_path, oxts_sec, oxts_nsec = item[DIR_OXTS]

        world2pose_tr, world2pose_q = item['world_to_pose']

        set_bgr8(data_group, DIR_IMAGE00, image00_data, FRAMEID_CAM0, image00_sec, image00_nsec)
        set_bgr8(data_group, DIR_IMAGE01, image01_data, FRAMEID_CAM1, image01_sec, image01_nsec)
        set_semantic2d(data_group, DIR_SEMANTIC, semantic_data, FRAMEID_CAM0, LABEL_TAG, image00_sec, image00_nsec)
        set_points(data_group, DIR_VELODYNE_POINTS, np.stack([velodyne_data_raw['x'], velodyne_data_raw['y'], velodyne_data_raw['z']], axis=1), FRAMEID_VELODYNE, velodyne_sec, velodyne_nsec)
        set_points(data_group, DIR_SICK_POINTS, np.stack([np.zeros_like(sick_data_raw['y']), sick_data_raw['y'], sick_data_raw['z']], axis=1), FRAMEID_SICK, sick_sec, sick_nsec)
        set_pose(data_group, 'world_to_pose', world2pose_tr, world2pose_q, FRAMEID_WORLD, FRAMEID_POSE, oxts_sec, oxts_nsec)

        success = True
        print('SequentialData {0:010d}'.format(dst_h5.get_current_data_index()))
        print('image00  :', image00_data_path)
        print('image01  :', image01_data_path)
        print('semantic :', semantic_data_path)
        print('velodyne :', velodyne_data_path)
        print('sick     :', sick_data_path)

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

    # data_group:h5py.Group = h5file.get_next_data_group()
    create_sequential_data(config, h5file)
    
    create_intrinsic(config, h5file)

    create_static_transforms(config, h5file)

    create_semanticsmap(config, h5file)
    create_labelconfig(h5file)
    
    h5file.close()

if __name__=='__main__':
    main()
