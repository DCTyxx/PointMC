#!/usr/bin/env bash

set -x

PWD=$(cd "$(dirname "$0")"/../;pwd)
cd $PWD

conda env create -f hack/environment.yml
source activate
conda deactivate
conda activate PointMC

if [ $? -ne 0 ]; then
    echo "Failed to create or activate Conda environment."
    exit 1
fi

wget -O pytorch3d-0.7.8-py310_cu118_pyt212.tar.bz2 https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu118_pyt212.tar.bz2
conda install pytorch3d-0.7.8-py310_cu118_pyt212.tar.bz2

wget -O mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

wget -O causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

if [ $? -ne 0 ]; then
    echo "Failed to get packages, check your network."
    exit 1
fi

pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
pip install numpy==1.26.4
pip install -r hack/requirements.txt

sudo apt update
sudo apt-get -y install python3-pybind11

cd utils/pointnet2_ops_lib
pip install .
cd ../..

sudo apt-get -y install libomp-dev
cd utils/pykdtree
pip install .
cd ../..
