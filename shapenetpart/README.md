
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
```


## Test

Run test script:

```shell
# test a large size model, test log output into './exp-test/'
python test.py --model_size l --ckpt <checkpoint_file>
```
