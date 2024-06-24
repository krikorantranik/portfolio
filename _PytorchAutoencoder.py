import numpy as np
import torch
import torch.nn as nn
import time, copy
import matplotlib.pyplot as plt
import pandas as pd
import time, copy
import random
import torchvision.utils as vutils
import copy

#CNN and Autoencoder

# Number of channels in the training images. For color images this is 3
nc = 3
#size of the representation
nr = 1000
#size of the starting point of the decoder
nz = 50

#CNN definition
class Encdec(nn.Module):
    def __init__(self, nc, nz, nr):
        super(Encdec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = nc, out_channels = 10, kernel_size = 5, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 5, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 5, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 5, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 1, kernel_size = 5, stride = 1, padding=1),
            nn.Flatten(),
            nn.Linear(2916, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, nr),
         )
        self.decoder = nn.Sequential(
            nn.Linear(nr, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, nz*64*64),
            nn.Unflatten(1, torch.Size([nz, 64, 64])),
            nn.Conv2d(in_channels = nz, out_channels = 10, kernel_size = 5, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 5, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = nc, kernel_size = 5, stride = 1, padding=1),      
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10092, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, nc*64*64),   
            nn.Unflatten(1, torch.Size([nc, 64, 64])),         
            nn.Tanh()            
         )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, input):
        return self.decoder(self.encoder(input))

netEncDec = Encdec(nc, nz, nr)
print(netEncDec)

#training function for autoencoder
# From https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def train_autoencoder(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # keep the best weights stored separately
    best_loss = np.inf
    best_epoch = 0

    # Each epoch has a training, validation, and test phase
    phases = ['train', 'val']

    # Keep track of how loss evolves during training
    training_curves = {}
    for phase in phases:
        training_curves[phase+'_loss'] = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data - for the autoencoder we don't care about the
            # labels, we are training the input against itself!
            for inputs in dataloaders[phase]:
                inputs = inputs
                # our targets are the same as our inputs!
                targets = inputs
                # print(inputs.shape)
                # print(targets.shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    loss = criterion(outputs, targets)

                    # backward + update weights only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                print(f'partial loss: {loss.item():4f}')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            training_curves[phase+'_loss'].append(epoch_loss)

            print(f'{phase:5} Loss: {epoch_loss:.4f}')

            # deep copy the model if it's the best loss
            if phase == 'val' and epoch_loss < best_loss:
              best_epoch = epoch
              best_loss = epoch_loss
              best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f} at epoch {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, training_curves

#train autoencoder
learning_rate = 0.001
num_epochs = 25
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(netEncDec.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
netEncDec, training_curves = train_autoencoder(netEncDec, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=num_epochs)

#display loss
plt.figure(figsize=(10,5))
plt.title("Loss During Training")
plt.plot(training_curves['train_loss'] ,label="train")
plt.plot(training_curves['val_loss'] ,label="val")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#generate images (fakes)
rand_list=[]
n=24
for i in range(n):
    rand_list.append(random.randint(0,200))
img_list = []
with torch.no_grad():
    regen = netEncDec(torch.stack(smpl, dim=0)).detach().cpu()
    img_list.append(vutils.make_grid(regen, padding=2, normalize=True))
# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Regenerated Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()