import config
import dataloader
import engine 
import ImageTransformer

import transformers
import torch 
import torch.nn as nn
import numpy as np 
import torchvision

import albumentations as alb


def run():
    train_dataset = torchvision.datasets.CIFAR10(root='input/train', train=True)
    val_dataset = torchvision.datasets.CIFAR10(root='input/val', train=False)

    train_transform = alb.Compose([
        alb.Resize(config.image_height, config.image_width, always_apply=True),
        alb.Normalize(config.mean, config.std, always_apply=True),
        alb.HorizontalFlip(p=0.1),
        alb.RandomBrightness(p=0.1),
        alb.RandomContrast(p=0.1),
        alb.RGBShift(p=0.1),
        alb.GaussNoise(p=0.1),
    ])

    val_transforms = alb.Compose([
        alb.Resize(config.image_height, config.image_width, always_apply=True),
        alb.Normalize(config.mean, config.std, always_apply=True)
    ])

    
    train_data = dataloader.dataloader(train_dataset, train_transform)
    val_data = dataloader.dataloader(val_dataset, val_transforms)


    train_loader = torch.utils.data.DataLoader(
        train_data,
        num_workers=4,
        pin_memory=True,
        batch_size=config.Batch_Size
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=4,
        pin_memory=True,
        batch_size=config.Batch_Size
    )

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
    
    if torch.cuda.is_available():
        accelarator = 'cuda'
    else:
        accelarator = 'cpu'
    
    device = torch.device(accelarator)
    torch.backends.cudnn.benchmark = True

    model = model.to(device)

    optimizer = transformers.AdamW(model.parameters(), lr=config.LR, weight_decay=config.weight_decay)

    num_training_steps = int((config.Epochs*len(train_dataset))/config.Batch_Size)

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = int(0.1*num_training_steps),
        num_training_steps = num_training_steps
    )
    
    best_acc = 0
    best_model = 0

    for epoch in range(config.Epochs):
        train_acc, train_loss = engine.train_fn(model, train_loader, optimizer, scheduler, device)
        val_acc, val_loss = engine.eval_fn(model, val_loader, device)
        print(f'\nEPOCH     =  {epoch+1} / {config.Epochs} | LR =  {scheduler.get_last_lr()[0]}')
        print(f'TRAIN ACC = {train_acc*100}% | TRAIN LOSS = {train_loss}')
        print(f'VAL ACC   = {val_acc*100}% | VAL LOSS = {val_loss}')
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict()

        torch.save(best_model, config.Model_Path)    

if __name__ == "__main__":
    run()