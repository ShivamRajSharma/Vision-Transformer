# Vision-Transformer On CIFAR-10

PyTorch Implementation of ViT (Vision Transformer), an transformer based architecture for Computer-Vision tasks. In ViT the author converts an image into 16x16 patche embedding and applies visual transformers to find relationships between visual semantic concepts. The ViT achieves State Of the Art performance on all Computer-Vision task. This idea is persented in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/pdf?id=YicbFdNTTy)

In the paper  ViT was first trained on ImageNet and then used transfer learning to achieve SOTA. So to check how the ViT performs without any prior pretraining, we
directly train the ViT on CIFAR-10 dataset.

<p align="center">
  <img src="https://github.com/ShivamRajSharma/Vision-Transformer/blob/master/ViT.png" height="300"/>
</p>

## Usage

1) Install all the libraries required ```pip install -r requirements``` .
2) Download the dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place it inside the ```input/``` .
3) Run ```python3 train.py``` .
4) For inference run ```python3 predict.py``` .

## Results 

The ViT took about __ minutes to train on CIFAR-10 dataset. Without any prior pre-training we were able to achieve about 74% accuracy on validation dataset.
