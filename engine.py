import torch
import torch.nn as nn
from tqdm import tqdm

def loss_fn(target, output):
    return nn.CrossEntropyLoss()(output, target)

def accuracy_fn(target, output):
    output = torch.softmax(output, dim=-1)
    output = output.argmax(dim=-1)
    return ((target==output)*1.0).mean()

def train_fn(model, dataloader, optimizer, scheduler, device):
    running_loss = 0
    running_acc = 0 
    model.train()
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None
        patches = data['patches'].to(device)
        label = data['label'].to(device)
        output = model(patches)
        loss = loss_fn(label, output)
        running_loss += loss.item()
        running_acc += accuracy_fn(label, output).item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    epoch_acc = running_acc/len(dataloader)
    epoch_loss = running_loss/len(dataloader)

    return epoch_acc, epoch_loss

def eval_fn(model, dataloader, device):
    running_loss = 0
    running_acc = 0
    model.eval()
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            patches = data['patches'].to(device)
            label = data['label'].to(device)
            output = model(patches)
            loss = loss_fn(label, output)
            running_loss += loss.item()
            running_acc += accuracy_fn(label, output).item()
    epoch_loss = running_loss/len(dataloader)
    epoch_acc = running_acc/len(dataloader)
    
    return epoch_acc, epoch_loss

if __name__ == "__main__":
    a = torch.randint(0, 10, (100,))
    b = torch.randint(0, 10, (100,))
    acc = accuracy_fn(a, b).item()
    print(acc*100)