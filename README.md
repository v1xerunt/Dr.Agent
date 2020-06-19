# Dr. Agent

The source code for *Dr. Agent: Clinical Predictive Model via Mimicked Second Opinions*

## Requirements

* Install python, pytorch. We use Python 3.7.3, Pytorch 1.1.
* If you plan to use GPU computation, install CUDA

## Dr. Agent on MIMIC-III

### Fast way to test Dr. Agent with MIMIC-III

We provide the trained weights in ```./saved_weights/```. You can obtain the reported performance in our paper by simply load the weights to the model by using following codes:

```python
checkpoint = torch.load('./saved_weights/TASK_TO_TEST')
save_chunk = checkpoint['chunk']
model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

### Data preparation

We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. To run MIMIC-III bechmark tasks, you should first build benchmark dataset according to https://github.com/YerevaNN/mimic3-benchmarks/.

After building the benchmark dataset, there will be a directory ```data/{task}``` for each created benchmark task. Then run ```extract_demo.py``` to extract demographics from the dataset (change ```TASK``` to specific task).

### Running Dr. Agent

1. You can train Dr. Agent on different tasks by running corresponding files.

2. The minimum input you need to run Dr. Agent is the dataset directory and the model save directory

    ```$ python train_decomp.py --data_path='./decompensation/data/' --save_path='./saved_weights/' ```

3. You can specify batch size ```--batch_size <integer> ```, learning rate ```--lr <float> ``` and epochs ```--epochs <integer> ```

4. Additional hyper-parameters can be specified such as the dimension of RNN, using LSTM or GRU, etc. Detailed information can be accessed by 

    ```$ python train_decomp.py --help```

5. When training is complete, it will output the performance of Dr. Agent on test dataset.

## Dr. Agent on other EHR datasets

The minimal inputs to Dr. Agent model should contain:

1. EHR records (batch_size, time_step, feature_num): The EHR records for a mini-batch of patients.
2. Masks (batch_size, time_step): Since all patients' records are padding to the same length to form batches, masks are binary values indicating whether current timestep is a padding value or real value.
3. Demographics (batch_size, demo_features): Demographic features of patients. If demographics are not applicable for your dataset, you should use zeros.

You can directly use the model structure in ```./model/``` directory for different proposes:

1. ```model_decomp.py```: binary classification with outputs at each timestep
2. ```model_los.py```: multi-label prediction
3. ```model_mortality.py```: binary classification with output at the last timestep
4. ```model_phenotyping.py```: multi-class prediction

You can also modify the structure for you specific tasks.

## Citation
```
Junyi Gao, Cao Xiao, Lucas M Glass, Jimeng Sun, 
Dr. Agent: Clinical predictive model via mimicked second opinions, 
Journal of the American Medical Informatics Association, ocaa074, https://doi.org/10.1093/jamia/ocaa074
```
