# ecg_representation_learning

This program is the source code of my thesis essay - Deep Metric Learning for Unsupervised Electrocardiogram Representation and Phenotyping.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

All experiment will upload to wandb.
Please follow the initialization instruction by [wandb](https://app.wandb.ai/).

Using Makefile to install the environment and the instruction is shown in the following section.
```
python_version = "3.7"
numpy = "==1.16.4"
torch = "==1.4.0"
torchvision = "==0.5.0"
tensorboard = "==1.14.0"
```

### Dataset
MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/.

Data preprocessing follows https://github.com/MousaviSajad/ECG-Heartbeat-Classification-seq2seq-model.

### Installing

Initial the environment.

```
Make init
```

Please double check the version of python, numpy, torch and torchvision!

## Running the tests

To run a test, please change the corresponding config in `run_experiment.py`.
```
Make run_experiment
```
The instruction for parameters in `ECGDL/experiment/config.py`.
#### Unsupervised Training Phase
`emotion_ssl: default_emotion_self_supervised_config` to train the unsupervised phase by Self-supervised ECG Representation Learning for Emotion Recognition (https://arxiv.org/abs/2002.03898).

`autoencoder: default_autoencoder_config` to train the unsupervised phase by autoencoder.

# Two types of autoencoder can be selected in default_autoencoder_config
```
self.algo_type = 'dae'  # Arrhythmia Detection from 2-lead ECG using Convolutional Denoising Autoencoders (https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_16.pdf)
or
self.algo_type = 'cae'  # An Automated ECG Beat Classification System sing Deep Neural Networks with an Unsupervised Feature Extraction Technique (https://www.mdpi.com/2076-3417/9/14/2921)
```

`unsup_mts: default_unsup_mts_config` to train the unsupervised phase by Unsupervised Scalable Representation Learning for Multivariate Time Series (https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries).


`MSwKM: default_MSwKM_config` to train the unsupervised phase by the proposed method.

#### Supervised Training Phase
Within each config, please first specify the corresponding algorithm trained in unsupervised phase and backbone model structure.
```
# The proposed method
unsup_method_type = "MSwKM"
backbone_model = "causalcnn"
# Unsupervised Scalable Representation Learning for Multivariate Time Series
unsup_method_type = "unsup_mts"
backbone_model = "causalcnn"
# Self-supervised ECG Representation Learning for Emotion Recognition
unsup_method_type = "emotion_ssl"
backbone_model = "original"
# Arrhythmia Detection from 2-lead ECG using Convolutional Denoising Autoencoders
unsup_method_type = "cdae"
backbone_model = "original"
# An Automated ECG Beat Classification System sing Deep Neural Networks with an Unsupervised Feature Extraction Technique
unsup_method_type = "dae"
backbone_model = "original"
```
Specify the pre-trained model path and select whether you want to freeze the backbone model weights.
```
self.model_args = {
                ...
                "model_path": xxx,
                ...
                "freeze": True/False,
            }
```
Specify whether classifier you wanted.
`self.algo_type = "non-linear" or "linear"`


`mitbih_downstream: default_mitbih_downstream_config` to train supervised classifier by MIT-BIH.


`lvh_downstream: default_lvh_downstream_config` to train supervised classifier by private dataset.

## Pretrained Models
Pretrained models are available on request.


## Authors

* **Crystal T. Wei** - *Initial work*
* **Micro** - *Codebase inital*
