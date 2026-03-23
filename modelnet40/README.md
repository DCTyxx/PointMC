
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

# train a large size model, experiment named 'test', train log output into './exp/'
python train.py --exp test --model_size l

# train a custom size model, experiment named 'test'
vim configs/modelnet40_custom.py
python train.py --exp test --model_size c

# more usage help
python train.py -h 
```

## Finetune or Resume

Run train script:

```shell
# resume the last experiment named 'test', for example 'test-001'
python train.py --exp test --model_size l --mode resume
# load the pretrained model, experiment named 'test'
python train.py --exp test --model_size l --mode finetune --ckpt <checkpoint_file>
```

## Test

Run test script:

```shell
# test a large size model, test log output into './exp-test/'
python test.py --model_size l --ckpt <checkpoint_file>
```