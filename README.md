# H5 KITTI-360

The script to convert the [KITTI-360 dataset](http://www.cvlibs.net/datasets/kitti-360/index.php) to HDF5.

## Preparation

1. [Download the KITTI-360 dataset](http://www.cvlibs.net/datasets/kitti-360/user_login.php)
1. Arrange the folders as shown in the [documentation](http://www.cvlibs.net/datasets/kitti-360/documentation.php).

## Run on Docker (strongly recommended)

1. Install "docker-py".
    ```bash
    pip install docker
    ```

1. Clone this repository with the following command.
    ```bash
    git clone https://github.com/shikishima-TasakiLab/h5_kitti360.git
    ```

1. Run the following command to perform the conversion. The first time, it takes a very long time to build the Docker image.
    ```bash
    python docker/kitti2hdf5.py -d /path/to/KITTI-360 -o /path/to/output/dir -s SEQUENCE
    ```

## Run on Local Environment

1. Install "[pointsmap-python](https://github.com/shikishima-TasakiLab/pointsmap-python)".

1. Install "[h5datacreator](https://github.com/shikishima-TasakiLab/h5datacreator)".

1. Install h5_kitti360 with the following command.
    ```bash
    pip install git+https://github.com/shikishima-TasakiLab/h5_kitti360
    ```

1. Run the following command to perform the conversion.
    ```bash
    kitti2hdf5 -d /path/to/KITTI-360 -o /path/to/output/dir -s SEQUENCE
    ```
