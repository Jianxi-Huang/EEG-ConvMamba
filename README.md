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
* torchvision==0.16.1 torchaudio==2.1.1
* causal-conv1d=1.1.1
* mamba-ssm=2.2.1
  
## Datasets

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
