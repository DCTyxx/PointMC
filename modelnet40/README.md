
## Prepare

Download ModelNet40 dataset: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip

Link the dataset:

```shell
ln -s /home/user/Datasets/modelnet40_ply_hdf5_2048/ dataset_link
```

## Train

Run train script:

```shell
# train a small size model
python train.py
```

## Test

Run test script:

```shell
# test a large size model, test log output into './exp-test/'
python test.py --model_size l --ckpt <checkpoint_file>
```