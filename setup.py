# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='h5dataviewer',
    version='0.1.1',
    description='`http://git-docker.tasakilab:5050/shikishima/h5_kitti360`',
    long_description='`http://git-docker.tasakilab:5050/shikishima/h5_kitti360`',
    author='Junya Shikishima',
    author_email='160442065@ccalumni.meijo-u.ac.jp',
    url='http://git-docker.tasakilab:5050/shikishima/h5_kitti360',
    license='',
    packages=find_packages(),
    install_requires=[
        "numpy", "h5py"
    ],
    entry_points={
        'console_scripts': [
        ]
    },
    python_requires='>=3.6'
)