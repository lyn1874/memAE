### memAE

This is an unofficial implementation of paper "Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder (MemAE) for Unsupervised Anomaly Detection". 

Majority of the code are based on the original repo https://github.com/donggong1/memae-anomaly-detection

| Dataset  | Paper | This implementation |
| -------- | ----- | ------------------- |
| UCSDped2 | 94.1  | 94.0                |
| Avenue   | 83.3  | 81.0                |

#### Requirements

- PyTorch == 1.4.0

- Python==3.7.6

- `./requirement.sh`

#### Prepare dataset

```pyth
prepare_data.sh dataset datapath
dataset: Avenue, UCSDped2
datapath: the path that you want to save the data, i.e., /project/anomaly_data/
```

#### Train the model

```pyth
./run.sh dataset datapath expdir
dataset: Avenue, UCSDped2 
datapath: the path that you want to save the data, i.e., /project/anomaly_data/
expdir: the path that you want to save the checkpoint
```

#### Evaluate the model

```pyth
./eval.sh dataset, datapath, version, ckpt, expdir
dataset: Avenue, UCSDped2
datapath: the path that you saved the data
version: experiment version
ckpt: the checkpoint step
expdir: the path that you saved the model checkpoint
```
If your Avenue dataset is saved under `/project/anomaly_data/Avenue/frames/testing/....`, run
`./eval.sh Avenue /project/anomaly_data/ 0 40 ckpt/` to get the reported performance






