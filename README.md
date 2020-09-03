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

- ./requirement.sh

#### Prepare dataset

The dataloader, transformation and other related functions are in the file data/utils.py

In my case, the data is arranged as ../Avenue/frames/training/training_video_10_frame_10.jpg, training_video_10 is the video string name and frame_10.jpg is the frame index in this video. Depend on your own data structure, you probably need to manually edit the function *setup* 

#### Train the model

```python
./run.sh
```

#### Evaluate the model

```pyth
./eval.sh 0 50
The first value is the experiment version and second value is the model ckpt step
```






