# Deep-Emotion For Facial Expression Recognition During Gambling Experiments

This repository provides a framework to conduct experiments on facial emotion recognition during gambling with the aim of building an emotional model of gambling. The used model is an implementation of the research paper [Deep-Emotion](https://arxiv.org/abs/1902.01019). The code and visualization technique is heavily based on [this](https://github.com/omarsayed7/Deep-Emotion) repository.

**Note**: This implementation is not the official one described in the paper.

## Architecture
- An end-to-end deep learning framework based on attentional convolutional networks.
- The attention mechanism is incorporated using spatial transformer networks.

<p align="center">
  <img src="image/net_arch.PNG" width="960" title="Deep-Emotion Architecture">
</p>

## Datasets
This implementation uses the following datasets:
- [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [CK+](https://ieeexplore.ieee.org/document/5543262)

*Note*: There are separate models trained on both datasets, and unfortunately I could not obtain the performance metrics specified in the paper.

## Training
The training methods and procedure are in the python notebooks. For both datasets, cross-entropy loss and an Adam optimizer were used.

## Usage
To conduct the experiments, run the ['live_experiment.py'](/live_experiment.py) file, but before, make sure to choose one of the trained models and the respective results csv file byt commenting out one of the imports, classnames and results file.