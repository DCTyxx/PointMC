
## Prepare

Download ScanObjectNN dataset: https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip

Link the dataset:

```shell
ln -s /home/user/Datasets/h5_files/ dataset_link
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