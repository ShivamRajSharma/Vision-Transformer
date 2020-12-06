import math

image_path = 'input/'
Model_Path = 'model/model.bin'

Batch_Size = 64
Epochs = 60
LR = 8e-4
weight_decay = 0.1

image_height = 32
image_width = 32

patch_size = 8

embedding_dims = patch_size*patch_size*3
dropout = 0.1
heads = 4
num_layers = 4
forward_expansion = 4
max_len = math.ceil(image_height*image_width/patch_size**2) + 1
layer_norm_eps = 1e-5
num_classes = 10

mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)