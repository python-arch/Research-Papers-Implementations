from tqdm import tqdm
import torch
from torchsummary import summary
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose , Resize , ToTensor , Normalize , RandomHorizontalFlip,RandomCrop
import yaml
import time
import torchvision
import sys
sys.path.append("/home/abdelrahman.elsayed/VIT_from_scratch/modules")
from VIT import *
from InputEmbedding import *
from EncoderBlock import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# load the config

with open('model_config.yml' , 'r') as f:
    config = yaml.safe_load(f)


print(config)

model = VIT(model_config=config).to(device)

# testing discrete components of the model
# input_embeds = InputEmbedding(config).to(device)
# test_input = torch.randn((1,3,224,224))
# projs = input_embeds(test_input)

# print(projs.size())

# summary(model , input_size=(3,224,224))

# training loop

optimizer = optim.Adam(model.parameters(), lr=config['training']['base_lr'], weight_decay=config['training']['weight_decay'])
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.LinearLR(optimizer)

# get the data

transform_training_data = Compose(
    [RandomCrop(32, padding=4), Resize((224)), RandomHorizontalFlip(), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_data = torchvision.datasets.CIFAR10(
    root='/home/abdelrahman.elsayed/VIT_from_scratch/data', train=True, download=True, transform=transform_training_data)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=config['training']['batch_size'],
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def main():
    model.train().to(device)
    epochs = config['training']['epochs']
    for epoch in tqdm(range(epochs), total=epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):

            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print('Batch {} epoch {} has loss = {}'.format(batch_idx, epoch, running_loss/200))                
                running_loss = 0

        scheduler.step()

if __name__ == "__main__":
    main()







