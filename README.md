# DeepQA
We propose a convolutional neural networks (CNN) based FR-IQA model, named Deep Image Quality Assessment (DeepQA), where the behavior of the HVS is learned from the underlying data distribution of IQA databases.

> Jongyoo Kim and Sanghoon Lee, “Deep learning of human visual sensitivity in image quality assessment framework,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1676–1684.


## Prerequisites
This code was developed and tested with Theano 0.9, CUDA 8.0, and Windows.

## Environment setting
### Setting database path:
For each database, set `BASE_PATH` to the actual root path of each database in the following files:
`IQA_DeepQA_FR_release/data_load/LIVE.py`,
`IQA_DeepQA_FR_release/data_load/TID2008.py`, and
`IQA_DeepQA_FR_release/data_load/TID2013.py`.

## Training DeepQA
We provide the demo code for training a DeepQA model.
```bash
python example.py
```

- `tr_te_file`: Store the randomly divided (training and testing) reference image indices in this file.
- `snap_path`: This indicates the path to store snapshot files


## Quantitative results
DeepQA was tested on the full-sets of LIVE IQA, CSIQ, TID2008, TID2013 databases. During the experiment, we randomly divided the reference images into two subsets, 80% for training and 20% for testing. The correlation coefﬁcients were averaged after the procedure was repeated 10 times while dividing the training and testing sets randomly.

|Database |SRCC  |PLCC  |
|---------|:----:|:----:|
|LIVE IQA |0.981 | 0.982|
|CSIQ     |0.961 | 0.965|
|TID2008  |0.947 | 0.951|
|TID2013  |0.939 | 0.947|

