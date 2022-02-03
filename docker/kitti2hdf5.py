import argparse
import os
import subprocess
from typing import Dict
import docker

DOCKER_IMAGE = 'h5_kitti360'

def build():
    docker_dir: str = os.path.dirname(os.path.abspath(__file__))
    print(docker_dir)

    proc = subprocess.run([
        'docker', 'build', '-t', DOCKER_IMAGE,
        '--ulimit', 'stack=8192', docker_dir
    ])
    if proc.returncode != 0:
        print(f'Failed: "{DOCKER_IMAGE}"')
        return
    print(f'Finished "{DOCKER_IMAGE}"')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-root-dir', type=str, metavar='PATH', required=True, help='Root directory of KITTI-360 Dataset.')
    parser.add_argument('-o', '--output-dir', type=str, metavar='PATH', required=True, help='Output Directory.')
    parser.add_argument('-s', '--sequence', type=int, choices=[0,2,3,4,5,6,7,9,10], required=True, help='Sequence of KITTI-360 Dataset.')
    args: Dict[str, str] = vars(parser.parse_args())

    if os.path.isdir(args['dataset_root_dir']) is False:
        raise ValueError(f'"{args["dataset_root_dir"]}" is not a directory.')
    if os.path.isdir(args['output_dir']) is False:
        raise ValueError(f'"{args["output_dir"]}" is not a directory.')

    docker_client = docker.from_env()

    try:
        docker_client.images.get(DOCKER_IMAGE)
    except docker.errors.ImageNotFound:
        build()

    container = docker_client.containers.run(
        DOCKER_IMAGE,
        command=f'kitti2hdf5 -d /workspace/KITTI-360 -o /workspace/HDF5 -s {args["sequence"]:d}',
        remove=True,
        stderr=True,
        detach=True,
        auto_remove=True,
        volumes={
            os.path.abspath(args['dataset_root_dir']): {'bind': '/workspace/KITTI-360', 'mode': 'ro'},
            os.path.abspath(args['output_dir']): {'bind': '/workspace/HDF5', 'mode': 'rw'}
        }
    )

    container_output = container.attach(stdout=True, stream=True, logs=True)

    for line in container_output:
        print(line.decode('utf-8'))

if __name__=='__main__':
    main()
