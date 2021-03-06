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

from .io import OxtsData, PlyData, SickData, VelodyneData
from .constant import *

def create_semanticsmap(config:Dict[str, str], dst_h5:H5Dataset):
    search_path = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_3d_semantics', DIR_TRAIN, config[CONFIG_SEQUENCE_DIR], 'static', '*.ply')
    poly_paths = glob(search_path)

    dynamic_labels:np.ndarray = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33], dtype=np.uint8)

    points = Points(quiet=True)

    # print(f'{"poly_paths":19s}:', len(poly_paths), flush=True)

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
    del points

    vgm = VoxelGridMap(quiet=True)
    vgm.set_semanticmap(points_np, semantic1d_np)
    vgm_np = vgm.get_voxel_semantic3d()
    vgm_size = vgm.get_voxel_size()
    vgm_min = vgm.get_voxels_min()
    vgm_max = vgm.get_voxels_max()
    vgm_center = vgm.get_voxels_center()
    vgm_origin = vgm.get_voxels_origin()

    map_group = dst_h5.get_common_group('map')
    set_voxel_semantic3d(map_group, 'map', vgm_np, FRAMEID_WORLD, vgm_size, vgm_min, vgm_max, vgm_center, vgm_origin, LABEL_TAG, map_id='Seq{0:04d}'.format(config[CONFIG_SEQUENCE]))
    print('SemanticMap Done', flush=True)

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

def create_intrinsic(config:Dict[str, str], dst_h5:H5Dataset):
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

def create_static_transforms(config:Dict[str, str], dst_h5:H5Dataset):
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
                set_pose(transforms_group, 'oxts_pose_to_cam0', translation, quaternion, FRAMEID_OXTS_POSE, FRAMEID_OXTS_CAM0)
            elif values[0] == 'image_01:':
                matrix[0:3, :] = np.reshape(np.float32(values[1:13]), (3, 4))
                translation, quaternion = matrix2quaternion(matrix)
                set_pose(transforms_group, 'pose_to_cam1', translation, quaternion, FRAMEID_POSE, FRAMEID_CAM1)
                set_pose(transforms_group, 'oxts_pose_to_cam1', translation, quaternion, FRAMEID_OXTS_POSE, FRAMEID_OXTS_CAM1)
            elif values[0] == 'image_02:':
                matrix[0:3, :] = np.reshape(np.float32(values[1:13]), (3, 4))
                translation, quaternion = matrix2quaternion(matrix)
                set_pose(transforms_group, 'pose_to_cam2', translation, quaternion, FRAMEID_POSE, FRAMEID_CAM2)
                set_pose(transforms_group, 'oxts_pose_to_cam2', translation, quaternion, FRAMEID_OXTS_POSE, FRAMEID_OXTS_CAM2)
            elif values[0] == 'image_03:':
                matrix[0:3, :] = np.reshape(np.float32(values[1:13]), (3, 4))
                translation, quaternion = matrix2quaternion(matrix)
                set_pose(transforms_group, 'pose_to_cam3', translation, quaternion, FRAMEID_POSE, FRAMEID_CAM3)
                set_pose(transforms_group, 'oxts_pose_to_cam3', translation, quaternion, FRAMEID_OXTS_POSE, FRAMEID_OXTS_CAM3)
            line:str = f.readline()

    with open(os.path.join(calibration_dir, 'calib_cam_to_velo.txt'), mode='r') as f:
        values:List[str] = f.readline().split()
        matrix:np.ndarray = np.identity(4, dtype=np.float32)
        matrix[0:3, :] = np.reshape(np.float32(values[0:12]), (3, 4))
        translation, quaternion = matrix2quaternion(matrix)
        translation, quaternion = invertTransform(translation=translation, quaternion=quaternion)
        set_pose(transforms_group, 'cam0_to_velo', translation, quaternion, FRAMEID_CAM0, FRAMEID_VELODYNE)
        set_pose(transforms_group, 'oxts_cam0_to_velo', translation, quaternion, FRAMEID_OXTS_CAM0, FRAMEID_OXTS_VELODYNE)

    with open(os.path.join(calibration_dir, 'calib_sick_to_velo.txt'), mode='r') as f:
        values:List[str] = f.readline().split()
        matrix:np.ndarray = np.identity(4, dtype=np.float32)
        matrix[0:3, :] = np.reshape(np.float32(values[0:12]), (3, 4))
        translation, quaternion = matrix2quaternion(matrix)
        set_pose(transforms_group, 'velo_to_sick', translation, quaternion, FRAMEID_VELODYNE, FRAMEID_SICK)
        set_pose(transforms_group, 'oxts_velo_to_sick', translation, quaternion, FRAMEID_OXTS_VELODYNE, FRAMEID_OXTS_SICK)

def convert_timestamp(timestamp:str) -> Tuple[int, int]:
    time_str_list = timestamp.split('.')
    if len(time_str_list) != 2:
        return None
    else:
        return int(datetime.strptime(time_str_list[0], '%Y-%m-%d %H:%M:%S').timestamp()), int(time_str_list[1])

def create_sequential_data(config:Dict[str, str], dst_h5:H5Dataset):
    data2dRaw_seq_dir:str = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_2d_raw', config[CONFIG_SEQUENCE_DIR])
    data3dRaw_seq_dir:str = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_3d_raw', config[CONFIG_SEQUENCE_DIR])
    data2dSemantic_seq_dir:str = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_2d_semantics', DIR_TRAIN, config[CONFIG_SEQUENCE_DIR])
    dataPoses_seq_dir:str = os.path.join(config[CONFIG_DATASET_ROOT_DIR], 'data_poses', config[CONFIG_SEQUENCE_DIR])

    # print(f'{"data2dRaw dir":19s}:', data2dRaw_seq_dir, flush=True)
    # print(f'{"data3dRaw dir":19s}:', data3dRaw_seq_dir, flush=True)
    # print(f'{"data2dSemantic dir":19s}:', data2dSemantic_seq_dir, flush=True)
    # print(f'{"dataPoses dir":19s}:', dataPoses_seq_dir, flush=True)

    image00data_paths:List[str] = sorted(glob(os.path.join(data2dRaw_seq_dir, DIR_IMAGE00, DIR_DATA_RECT, '*.png')))
    image00_timestamps:List[str]
    with open(os.path.join(data2dRaw_seq_dir, DIR_IMAGE00, 'timestamps.txt'), mode='r') as f:
        image00_timestamps = f.readlines()

    image00_data_dict:Dict[str, Dict[str, Tuple[str, int, int]]] = {}
    for image00_data_path, image00_timestamp in zip(image00data_paths, image00_timestamps):
        key = str(int(os.path.splitext(os.path.basename(image00_data_path))[0]))

        image00_dataset:Dict[str, Tuple[str, int, int]] = {}

        image00_sec, image00_nsec = convert_timestamp(image00_timestamp)
        image00_dataset[DIR_IMAGE00] = (image00_data_path, image00_sec, image00_nsec)

        image00_data_dict[key] = image00_dataset

    # print(f'{"2d/rect/image00":19s}:', len(image00_data_dict), flush=True)

    image01data_paths:List[str] = sorted(glob(os.path.join(data2dRaw_seq_dir, DIR_IMAGE01, DIR_DATA_RECT, '*.png')))
    image01_timestamps:List[str]
    with open(os.path.join(data2dRaw_seq_dir, DIR_IMAGE01, 'timestamps.txt'), mode='r') as f:
        image01_timestamps = f.readlines()

    image01_data_dict:Dict[str, Dict[str, Tuple[str, int, int]]] = {}
    for image01_data_path, image01_timestamp in zip(image01data_paths, image01_timestamps):
        key = str(int(os.path.splitext(os.path.basename(image01_data_path))[0]))
        if key not in image00_data_dict.keys(): continue

        image01_dataset:Dict[str, Tuple[str, int, int]] = image00_data_dict[key].copy()

        image01_sec, image01_nsec = convert_timestamp(image01_timestamp)
        image01_dataset[DIR_IMAGE01] = (image01_data_path, image01_sec, image01_nsec)

        image01_data_dict[key] = image01_dataset


    del image00_data_dict

    # print(f'{"2d/rect/image01":19s}:', len(image01_data_dict), flush=True)

    velodyne_data_paths:List[str] = sorted(glob(os.path.join(data3dRaw_seq_dir, DIR_VELODYNE_POINTS, DIR_DATA, '*.bin')))
    velodyne_timestamps:List[str]
    with open(os.path.join(data3dRaw_seq_dir, DIR_VELODYNE_POINTS, 'timestamps.txt'), mode='r') as f:
        velodyne_timestamps = f.readlines()

    velodyne_data_dict:Dict[str, Dict[str, Tuple[str, int, int]]] = {}
    for velodyne_data_path, velodyne_timestamp in zip(velodyne_data_paths, velodyne_timestamps):
        key = str(int(os.path.splitext(os.path.basename(velodyne_data_path))[0]))
        if key not in image01_data_dict.keys(): continue
        velodyne_time_tuple = convert_timestamp(velodyne_timestamp)
        if velodyne_time_tuple is None: continue

        velodyne_dataset:Dict[str, Tuple[str, int, int]] = image01_data_dict[key].copy()
        velodyne_dataset[DIR_VELODYNE_POINTS] = (velodyne_data_path, velodyne_time_tuple[0], velodyne_time_tuple[1])
        velodyne_data_dict[key] = velodyne_dataset


    del image01_data_dict

    # print(f'{"3d/velodyne":19s}:', len(velodyne_data_dict), flush=True)

    sick_data_paths:List[str] = sorted(glob(os.path.join(data3dRaw_seq_dir, DIR_SICK_POINTS, DIR_DATA, '*.bin')))
    sick_timestamps:List[str]
    with open(os.path.join(data3dRaw_seq_dir, DIR_SICK_POINTS, 'timestamps.txt'), mode='r') as f:
        sick_timestamps = f.readlines()

    sick_data_dict:Dict[str, Dict[str, Tuple[str, int, int]]] = {}
    for sick_data_path, sick_timestamp in zip(sick_data_paths, sick_timestamps):
        key = str(int(os.path.splitext(os.path.basename(sick_data_path))[0]))
        if key not in velodyne_data_dict.keys(): continue
        sick_time_tuple:Tuple[int, int] = convert_timestamp(sick_timestamp)
        if sick_time_tuple is None: continue

        sick_dataset:Dict[str, Tuple[str, int, int]] = velodyne_data_dict[key].copy()
        sick_dataset[DIR_SICK_POINTS] = (sick_data_path, sick_time_tuple[0], sick_time_tuple[1])

        sick_data_dict[key] = sick_dataset


    del velodyne_data_dict

    # print(f'{"3d/sick":19s}:', len(sick_data_dict), flush=True)

    oxts_data_paths:List[str] = sorted(glob(os.path.join(dataPoses_seq_dir, DIR_OXTS, DIR_DATA, '*.txt')))
    oxts_timestamps:List[str]
    with open(os.path.join(dataPoses_seq_dir, DIR_OXTS, 'timestamps.txt'), mode='r') as f:
        oxts_timestamps = f.readlines()

    oxts_data_dict:Dict[str, Dict[str, Tuple[str, int, int]]] = {}
    for oxts_data_path, oxts_timestamp in zip(oxts_data_paths, oxts_timestamps):
        key = str(int(os.path.splitext(os.path.basename(oxts_data_path))[0]))
        if key not in sick_data_dict.keys(): continue
        oxts_time_tuple:Tuple[int, int] = convert_timestamp(oxts_timestamp)
        if oxts_time_tuple is None: continue

        oxts_dataset:Dict[str, Tuple[str, int, int]] = sick_data_dict[key].copy()
        oxts_dataset[DIR_OXTS] = (oxts_data_path, oxts_time_tuple[0], oxts_time_tuple[1])
        oxts_data_dict[key] = oxts_dataset


    del sick_data_dict

    # print(f'{"pose/oxts":19s}:', len(oxts_data_dict), flush=True)

    semanticData_paths:List[str] = sorted(glob(os.path.join(data2dSemantic_seq_dir, DIR_IMAGE00, DIR_SEMANTIC, '*.png')))
    semantic_data_dict:Dict[str, Dict[str, Tuple[str, int, int]]] = {}
    for semanticData_path in semanticData_paths:
        key = str(int(os.path.splitext(os.path.basename(semanticData_path))[0]))
        if key not in oxts_data_dict.keys(): continue

        semantic_dataset:Dict[str, Tuple[str, int, int]] = oxts_data_dict[key].copy()
        semantic_dataset[DIR_SEMANTIC] = (semanticData_path, 0, 0)

        semantic_data_dict[key] = semantic_dataset


    del oxts_data_dict

    # print(f'{"semantic":19s}:', len(semantic_data_dict), flush=True)

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

    # print(f'{"pose/gt":19s}:', len(pose_data_dict), flush=True)

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
        oxts_data_instance = OxtsData(oxts_data_path)
        oxts_tr:np.ndarray = oxts_data_instance.data[SUBTYPE_TRANSLATION]
        oxts_q:np.ndarray = oxts_data_instance.data[SUBTYPE_ROTATION]

        world2pose_tr, world2pose_q = item['world_to_pose']

        set_bgr8(data_group, DIR_IMAGE00, image00_data, FRAMEID_CAM0, image00_sec, image00_nsec)
        set_bgr8(data_group, DIR_IMAGE01, image01_data, FRAMEID_CAM1, image01_sec, image01_nsec)
        set_semantic2d(data_group, DIR_SEMANTIC, semantic_data, FRAMEID_CAM0, LABEL_TAG, image00_sec, image00_nsec)
        set_points(data_group, DIR_VELODYNE_POINTS, np.stack([velodyne_data_raw['x'], velodyne_data_raw['y'], velodyne_data_raw['z']], axis=1), FRAMEID_VELODYNE, velodyne_sec, velodyne_nsec)
        set_points(data_group, DIR_SICK_POINTS, np.stack([np.zeros_like(sick_data_raw['y']), sick_data_raw['y'], sick_data_raw['z']], axis=1), FRAMEID_SICK, sick_sec, sick_nsec)
        set_pose(data_group, 'world_to_pose', world2pose_tr, world2pose_q, FRAMEID_WORLD, FRAMEID_POSE, oxts_sec, oxts_nsec)
        set_pose(data_group, 'oxts', oxts_tr, oxts_q, FRAMEID_WORLD, FRAMEID_OXTS_POSE, oxts_sec, oxts_nsec)

        success = True
        print(f'SequentialData {dst_h5.get_current_data_index():010d}', flush=True)
        print('image00  :', image00_data_path, flush=True)
        print('image01  :', image01_data_path, flush=True)
        print('semantic :', semantic_data_path, flush=True)
        print('velodyne :', velodyne_data_path, flush=True)
        print('sick     :', sick_data_path, flush=True)
        print('oxts     :', oxts_data_path, flush=True)

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
    print(config, flush=True)

    h5file:H5Dataset = H5Dataset(config[CONFIG_HDF5_PATH])

    create_sequential_data(config, h5file)

    create_intrinsic(config, h5file)

    create_static_transforms(config, h5file)

    create_semanticsmap(config, h5file)

    create_labelconfig(h5file)

    h5file.close()

    print('Saved "{0}"'.format(config[CONFIG_HDF5_PATH]), flush=True)

if __name__=='__main__':
    main()
