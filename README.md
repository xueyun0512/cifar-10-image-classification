This repo is forked then modified from the initial repo for the course "Efficient Deep Learning" at IMT Atlantique | Brest | Spring 2025
--

The goal is to train a ResNet18 model in order to perform image classification on the dataset CIFAR10. Several fine-tuning and optimization method have been used:
--
1. Data Augmentation, Regularization
   - exploring hyper parameters on a fixed architecture
2. Quantization
   - quantification on a small convolutional network
3. Pruning
   - Pruning a small convolutional networks
4. Factorization
  - Factorizing Deep neural networks
5. Distillation
  - Distillation of knowledge and features between neural networks
6. Embedded Software and Hardware for Deep Learning
7. Final Session
  - Challenge final results


Objectives:
--
Make sure you have download the CIFAR10 dataset at /opt/img/effdl-cifar10/
To run the code, make sure you are inside "~\efficient-deep-learning\LAB\cifar10_image_classification\" 

TLDR : this course is mostly based on a long project. The overarching goal is to explore the tradeoff between the performances of neural networks (= Accuracy on the test set) and complexity. Complexity can be either computational complexity (number of arithmetic operations), or memory complexity (memory size necessary to use the network).

We tested combinations of the various ideas mentioned above. We explored the tradeoff between architecture, number of parameters, and accuracy. Then, we studied new notions that open new avenues to explore this tradeoff : quantization, pruning, factorization, distillation. A deeper insight on how to thing about specific software or hardware architecture in order to fully exploit all the optimizations that can be done.



General References
--

[List of references IMT Atlantique and AI](https://docs.google.com/document/d/1-IX-IO8DXYOZSiihOe0ttjvJvcEO9WLU2UtZgej86gQ/edit#heading=h.iueps2uhjocc)

Amazon Book - [Dive into Deep learning](https://d2l.ai/)

[Tutorial presentation on Efficient Deep Learning from NeurIPS'19](http://eyeriss.mit.edu/2019_neurips_tutorial.pdf)


Training Deep Networks
--

Here are some academic papers discussing learning rate strategies :

- [Cyclic learning rates](https://arxiv.org/abs/1506.01186)
- [Demystifying Learning Rate Policies for High Accuracy Training of Deep Neural Networks](https://arxiv.org/abs/1908.06477)
- [A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and Distillation](https://arxiv.org/abs/1810.13243)

Main strategies are [readily available in pytorch.](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

Pytorch
--

Start page to access the full [python API](https://pytorch.org/docs/stable/torch.html) of pytorch, to check all existing functions.

[A useful tutorial on Saving and Loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

[Pytorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html).

Data Augmentation
--
Popular methods :

[Cut Out](https://github.com/uoguelph-mlrg/Cutout)

[Auto Augment](https://github.com/DeepVoltaire/AutoAugment)

Other ressources :

[A list of papers and code for data augmentation](https://github.com/CrazyVertigo/awesome-data-augmentation)

[IMGAUG](https://imgaug.readthedocs.io/en/latest/index.html) and [Colab Notebook showing how to use IMGAUG with pytorch](https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll)

A popular python package in Kaggle competitions : [Albumentations](https://github.com/albumentations-team/albumentations)

Quantization
--
[Binary Connect](http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-b)

[XnorNet](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_32)

[BNN+](https://openreview.net/forum?id=SJfHg2A5tQ)

[Whitepaper of quantization](https://arxiv.org/abs/1806.08342)


Pruning
--
[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)

[ThiNet](https://arxiv.org/abs/1707.06342)


[AutoML for Model Compression (AMC)](https://arxiv.org/abs/1802.03494)

[Pruning Channel with Attention Statistics (PCAS)](https://arxiv.org/abs/1806.05382)

[BitPruning: Learning Bitlengths for Aggressive and Accurate Quantization](https://arxiv.org/abs/2002.03090)

Factorization and operators
--

[Deep Compression](https://arxiv.org/abs/1510.00149)

[Deep K-means](https://arxiv.org/abs/1806.09228)

[SqueezeNet](https://arxiv.org/abs/1602.07360)

[MobileNet](https://arxiv.org/abs/1704.04861)

[MobileNetV2](https://arxiv.org/abs/1801.04381)

[Shift Attention Layers](https://arxiv.org/abs/1905.12300)

Distillation
--
[Distilling the knowledge in a neural network](https://arxiv.org/abs/1503.02531)

[Fitnets: Hints for thin deep nets](https://arxiv.org/abs/1412.6550)

[LIT: Learned Intermediate Representation Training for Model Compression](http://proceedings.mlr.press/v97/koratana19a.html)

[A Comprehensive Overhaul of Feature Distillation](https://arxiv.org/abs/1904.01866)

[And the bit goes down: Revisiting the quantization of neural networks](https://arxiv.org/abs/1907.05686)

Self-Supervised Learning
--
[Pretext tasks used for learning Representations from data without labels](https://atcold.github.io/pytorch-Deep-Learning/en/week10/10-1/)


Embedded Software and Hardware
--

See references section of [Tutorial presentation on Efficient Deep Learning from NeurIPS'19](http://eyeriss.mit.edu/2019_neurips_tutorial.pdf).


Companies / private sector
--

[13 highest funded startups for hardware for DL](https://www.crunchbase.com/lists/relevant-ai-chip-startups/922b3cf5-b19d-4c28-9978-4e66ccb52337/organization.companies)

[More complete list of companies working on hardware DL](https://roboticsandautomationnews.com/2019/05/24/top-25-ai-chip-companies-a-macro-step-change-on-the-micro-scale/22704/)


Setting up on personal computer
--
Please see [here](getting_started.md) for instructions on how to setup your environment on your personal computer.
