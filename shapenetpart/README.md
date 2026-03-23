
## Prepare

Download ShapeNetPart dataset: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

Run script:

```shell
python prepare_dataset.py -i /home/user/Datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal/ -o shapenetpart_presample.pt

```

The `shapenetpart_presample.pt` will be saved in `/home/user/Datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal/`. 

Link the dataset:

```shell
ln -s /home/user/Datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal/ dataset_link
```

## Train

Run train script:

```shell
# train a small size model
python train.py

# train a large size model, experiment named 'test', train log output into './exp/'
python train.py --exp test --model_size l

# train a custom size model, experiment named 'test'
vim configs/shapenetpart_custom.py
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
