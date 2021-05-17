# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='h5_kitti360',
    version='0.1.0',
    description='`http://git-docker.tasakilab:5050/shikishima/h5_kitti360`',
    long_description='`http://git-docker.tasakilab:5050/shikishima/h5_kitti360`',
    author='Junya Shikishima',
    author_email='160442065@ccalumni.meijo-u.ac.jp',
    url='http://git-docker.tasakilab:5050/shikishima/h5_kitti360',
    license='',
    packages=find_packages(),
    install_requires=[
        "numpy", "h5py", "scipy", "pointsmap"
    ],
    entry_points={
        'console_scripts': [
            'kitti2hdf5 = h5_kitti360.kitti2hdf5:main'
        ]
    },
    python_requires='>=3.6'
)