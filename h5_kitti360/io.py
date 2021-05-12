
from typing import Dict, List
import numpy as np

PLY_PROPERTIES_DTYPE = {
    'char': 'i1',
    'uchar': 'u1',
    'short': 'i2',
    'ushort': 'u2',
    'int': 'i4',
    'uint': 'u4',
    'float': 'f4',
    'double': 'f8'
}

PLY_PROPERTIES_BYTES = {
    'char': 1,
    'uchar': 1,
    'short': 2,
    'ushort': 2,
    'int': 4,
    'uint': 4,
    'float': 4,
    'double': 8
}

class PlyData():
    ply_format:str = ''
    ply_version:str = ''
    elements:Dict[str, int] = {}
    properties_dtype:Dict[str, Dict[str, str]] = {}
    properties_bytes:Dict[str, int] = {}
    data:Dict[str, np.ndarray] = {}

    def __init__(self, ply_path:str):
        self.load(ply_path)

    def load(self, ply_path:str):
        with open(ply_path, mode='rb') as ply_file:
            current_element:str = ''
            while True:
                line:List[bytes] = ply_file.readline().split()
                if line[0] == b'end_header':
                    break
                elif line[0] == b'element':
                    current_element = line[1].decode()
                    self.elements[current_element] = int(line[2].decode())
                    self.properties_dtype[current_element] = {}
                    self.properties_bytes[current_element] = 0
                elif line[0] == b'property':
                    tmp_type:str = line[1].decode()
                    self.properties_dtype[current_element][line[2].decode()] = PLY_PROPERTIES_DTYPE[tmp_type]
                    self.properties_bytes[current_element] += PLY_PROPERTIES_BYTES[tmp_type]
                elif line[0] == b'format':
                    self.ply_format = line[1].decode()
                    self.ply_version = line[2].decode()
            
            if self.ply_format in ['binary_little_endian', 'binary_big_endian']:
                endian:str = '>' if self.ply_format == 'binary_big_endian' else '<'

                for element_key, element_item in self.elements.items():
                    for dtype_key in self.properties_dtype[element_key].keys():
                        self.properties_dtype[element_key][dtype_key] = endian + self.properties_dtype[element_key][dtype_key]
                    self.data[element_key] = np.frombuffer(ply_file.read(element_item * self.properties_bytes[element_key]), dtype=np.dtype([(key, item) for key, item in self.properties_dtype[element_key].items()]), count=element_item)
            elif self.ply_format == 'ascii':
                for element_key, element_item in self.elements.items():
                    dtype = np.dtype([(key, item) for key, item in self.properties_dtype[element_key].items()])
                    self.data[element_key] = np.empty((0,), dtype=dtype)
                    for _ in range(element_item):
                        line:List[bytes] = ply_file.readline().split()
                        self.data[element_key] = np.append(self.data[element_key], np.array(line, dtype=dtype), axis=0)
            else:
                raise NotImplementedError('Format "{0}" is not supported.')

if __name__=='__main__':
    pd = PlyData('/data/KITTI-360/data_3d_semantics/2013_05_28_drive_0000_sync/static/000002_000385.ply')
    print(pd.ply_version)
    print(pd.ply_format)
    print(pd.elements)
    print(pd.properties_dtype)
    print(pd.properties_bytes)
    print(pd.data)
