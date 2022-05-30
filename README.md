Auxiliary Classifier GAN for Covid & healthy X-rays image generation implementation in Tensorflow.
This is a non-official implementation of the CovidGAN 
architecture described in the paper CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved Covid-19 Detection (https://arxiv.org/abs/2103.05094).



# Abstract
--------------------------------

One of the biggest challenges in the medical field is the limitations of the data quantity, especially for training classifiers based on neural networks. The objective of this 
project is to develop a GAN (AC-GAN) to generate both Covid and healthy X-ray images. The GAN is used to generate a dataset that will be used to train a classifier.

## Common questions
- **Why not using data augmentations based on image manipulations (geometric/photometric)?** This type of transformations cannot be done without clinical considerations otherwise it will perform poorly on validation.
- **Why using an AC-GAN?** We have also tested a DC-GAN and compared the results. However, the AC-GAN allows to generate X-rays of multiple classes (covid, pneumonia ...)