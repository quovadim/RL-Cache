# RL-Cache

RL-Cache is an automatic algorithm for training cache admission for web caches. 
This code provides infrastructure for training and testing admission models

## Getting Started
first of all copy the project to your machine
```
git clone https://github.com/WVadim/RL-Cache
cd RL-Cache
```

### Prerequisites
To run the code, you will need
```
python 2.7
tensorflow library for python
keras library for python
numpy library for python
matplotlib library for python
hurry.filesize library for python
tqdm library for python

g++ compiler
boost library for C++
```
Now make all executables
```
make all
cd feature_collector
make all
cd ../rewards_collector
make all
```

### Dataset format
If you have some logs from the cache server or another data that is appropriate for this problem, you can user it.

Data format we require is bunch of .csv files named from 0.csv to N.csv where 0.csv is the first file.
Inside each file you need to have following format
```
timestamp object_id object_size
```
Take into account that columns should be separated with space, no headers should be into the data and all values should be integers.

If you don't have the data but still want to test our method, cheer up, just run script called example_data_generator.py

### Data preprocessing example

Here i give an example how to prepare your dataset for this project. If you already have your data, place all .csv files into the folder
data/example/ under project path, if you have no data just run
```
python example_data_generator.py example
```
Generation might take some time, but not too long.

Once you got your data, run following script to obtain precalculated features. That speeds up neural network training a lot, so it is better to do it once here.
```
./gather_data.sh data/example/ data/example_rewarded/
```

This process might be long, depending or your dataset size it might take up to 3 hrs for 500M requests.

Once the process is finished, you will need to get statistics data for the neural network to normalize features, to do that run the following code
```
python collect_statistics.py statistics/example -r=example -i=2
```
where option -r stands for the name of the folder where the data is stored and -i stands for the number of requests to use for statistics in millions, 
so -i=2 mean that first 2 millions of requests will be used for the statistics collection.

Now you are ready for first training.

### Training configurations format
To create experiment, you will need to create a folder into experiments/ and add 4 .json files in it.

You can always find an example of the experiments into experiments/example

So let's go for each .json file one by one

I will describe only usable parameters, all parameters that are not described should be the same as in example

#### model.json
```
dropout rate - probability of dropout during NN training
multiplier each - Maximal width of the network will be multiplier each * input size
layers each - Maximal depth of the network
use batch normalization - turn on batch normalization into the NN
admission lr - learning rate of the network
```
#### statistics.json
```
"statistics" - path to statistics file generated previously
"warmup" - number of lines into the statistics to skip
"split step" - width of the input vector

"normalization limit" - maximal allowed variance
"bias" - default bias

"save" - save statistics short description
"load" - load statistics short description

"show stat" - make statistics verbose
```
#### test.json
```
"batch size" - size of the batch for NN

"seed" - random seed

"algorithm type" - list of algorithms to test
"check size" - cache size to print

"min size" - minimal cache size,
"max size" - x where y ^ x  * minimal cache size,
"step" - y where y ^ x  * minimal cache size,

"period" - save period in seconds
"alpha" - list of metrics, 0 for OHR, 1 for BHR

"warmup" - number of requests that will be used for the warmup
```
#### train.json
```
"data" - name of the data folder
"cache size" - cache size in megabytes

"target" - algorithm to train
"batch size" - batch size for the neural network
"seed" - random seed

"samples" - number of samples each step
"percentile admission" - top n percentile for samples selection

"warmup" - number of requests to use for cache warmup
"runs" - maximal number of steps
"refresh period" - q value
"repetitions" - number of repetitions of each step
"overlap" - L value
"period" - K + L value

"algorithms" - list of algorithms to compare with
```

### First train

NOw we are ready for the first train, run train command
```
python train.py example -t=15 -v
```
where -t is number of threads and -v is to show training progress





