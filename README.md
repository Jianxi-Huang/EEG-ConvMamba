# EEG-ConvMamba

#### EEG-ConvMamba: EEG Decoding and Visualization for Motor Imagery via Convolutional Neural Networks and Mamba

## Model architecture

<div align="center">
  <img src="https://github.com/Jianxi-Huang/EEG-ConvMamba/blob/main/Model%20Architecture.png" width="700px">
</div>


* We propose a novel end-to-end deep learning network architecture named EEG-ConvMamba for MI-EEG decoding, which combines mutil-branch convolution and Mamba structures for efficient integration of local-global information.
* Extensive experiments are conducted to explore the effects of the stacking depth, convolution kernel size, and convolution filter parameters of Mamba. The results show that the stacking depth of the Mamba module and the increase in the number of convolution         filters enhance the performance of the model. With increasing convolution kernels, the model performance tends to decrease.
* To enhance the interpretability of the network, we employ the Smooth Grad-CAM visualization technique to analyze the learned features via EEG topography.
## Requirements

#### The following are the required installation packages:
* python=3.10.13
* cudatoolkit=11.8
* torch=2.1.1
* torchvision=0.16.1
* torchaudio=2.1.1
* causal-conv1d=1.1.1
* mamba-ssm=2.2.1
  
## Datasets

#### The dataset we used and the average results obtained：

* BCI_competition_IV 2a — Average accuracy 80.06%
* High Gamma Dataset — Average accuracy 97.09%
* OpenBMI — Average accuracy 72.26%

### I hope this paper is useful to everyone, and wish you all success in your researches. :blush:
