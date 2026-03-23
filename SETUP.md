## Summary

- CUDA: 11.8
- cuDNN: 8.9.1
- Pytorch: 2.1.2
- Python: 3.10.8
- OS: Ubuntu22.04 LTS
- Mamba: 2.2.2
- causal-conv1d: 1.4.0

## Setup automatically

Run `make setup` for automatic installation. If issues arise, follow the steps below for manual setup.


## Setup manually
 
create environment
```shell
conda env create -f hack/environment.yml
```

activate environment
```shell
source activate
conda deactivate
conda activate PointMC
```

setup python packages manually
```shell
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu118_pyt212.tar.bz2
conda install pytorch3d-0.7.8-py312_cu121_pyt231.tarxf

wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```

degrade numpy to 1.x
```shell
pip install numpy==1.26.4
```

setup python packages by pypi
```shell
pip install -r hack/requirements.txt
```

install pybind
```shell
sudo apt update
sudo apt-get -y install python3-pybind11
```

setup pointnet2 libs
```shell
cd utils/pointnet2_ops_lib
pip install .
cd ../..
```

setup pykdtree libs
```shell
sudo apt-get -y install libomp-dev
cd utils/pykdtree
pip install .
cd ../..
```

## Solve dependency problems(optional)


install eigen (optional, if setup pykdtree failed)
```shell
sudo apt-get -y install libeigen3-dev
sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
```

setup openmp (optional, if setup pykdtree failed)
```shell
export PWD=$(cd "$(dirname "$0")"/../;pwd)

sudo apt-get -y install cmake autoconf automake libtool flex
autoreconf -f -i
mkdir $PWD/tmp
cd $PWD/tmp
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
tar -zxf openmpi-4.1.6.tar.gz
cd openmpi-4.1.6
./configure --prefix=$PWD/tmp/openmpi-4.1.6
make -j8
sudo make install
cd $PWD
sudo mv $PWD/tmp/openmpi-4.1.6 /usr/local/openmpi
export PATH="$PATH:/usr/local/openmpi/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/openmpi/lib"
```
