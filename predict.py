import config
import ImageTransformer

import torch
import torch.nn as nn
import numpy as np
from PIL import Image 
import albumentations as alb

def predict(image_path):
    model = ImageTransformer.ImageTransformer(
        embedding_dims = config.embedding_dims,
        dropout = config.dropout,
        heads = config.heads,
        num_layers = config.num_layers,
        forward_expansion = config.forward_expansion,
        max_len = config.max_len,
        layer_norm_eps = config.layer_norm_eps,
        num_classes = config.num_classes,
    )

    model.load_state_dict(torch.load(config.Model_Path))
    model.eval()

    image = np.array(Image.open(image_path).convert('RGB'))
    transform = alb.Compose([
        alb.Resize(config.image_height, config.image_width, always_apply=True),
        alb.Normalize(config.mean, config.std, always_apply=True)
    ])

    image = transform(image=image)['image']
    
    image = torch.tensor(image, dtype=torch.float)

    image = image.unfold(0, config.patch_size, config.patch_size).unfold(1, config.patch_size, config.patch_size)
    image = image.reshape(image.shape[0], image.shape[1], image.shape[2]*image.shape[3]*image.shape[4])
    patch = image.view(-1, image.shape[-1])
    patch = patch.unsqueeze(0)

    idx_to_class = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    output = model(patch)
    
    prediction_class = torch.softmax(output, dim=-1)[0].argmax(dim=-1).item()
    prediction = idx_to_class[prediction_class]
    print(f'THE IMAGE CONTAINS A {prediction.upper()}')

if __name__ == "__main__":
    image_path = 'image.jpg'
    predict(image_path)


