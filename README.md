# Dr. Agent

The source code for *Dr. Agent: Clinical Predictive Model via Simulated Second Opinions*

## Requirements

* Install python, pytorch. We use Python 3.7.3, Pytorch 1.1.
* If you plan to use GPU computation, install CUDA

## Data preparation
We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. To run MIMIC-III bechmark tasks, you should first build benchmark dataset according to https://github.com/YerevaNN/mimic3-benchmarks/.

After building the benchmark dataset, there will be a directory ```data/{task}``` for each created benchmark task. Then run ```extract_demo.py``` to extract demographics from the dataset (change ```TASK``` to specific task).

## Running Dr. Agent
1. You can train Dr. Agent on different tasks by running corresponding files.

2. The minimum input you need to run Dr. Agent is the dataset directory and the model save directory

    ```$ python train_decomp.py --data_path='./decompensation/data/' --save_path='./saved_weights/' ```

3. You can specify batch size ```--batch_size <integer> ```, learning rate ```--lr <float> ``` and epochs ```--epochs <integer> ```

4. Additional hyper-parameters can be specified such as the dimension of RNN, using LSTM or GRU, etc. Detailed information can be accessed by 

    ```$ python train_decomp.py --help```

5. When training is complete, it will output the performance of Dr. Agent on test dataset.