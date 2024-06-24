import numpy as np
import torch
import torch.nn as nn
import time, copy
import pandas as pd
import time, copy
import copy


#GAN: adversarial networks (generate)

# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

#discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.pipeline(input)

#generator
class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()
        self.pipeline = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            nn.ConvTranspose2d( ngf * 4, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.pipeline(input)

# custom weights initialization called on netG and netD
# This function initializes the weights of certain layers according to the distributions described in the original paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#Discriminator and Generator
netD = Discriminator(nc, ndf)
netG = Generator(nc, nz, ngf)

# Apply the weights_init function to randomly initialize all weights
netD.apply(weights_init)
netG.apply(weights_init)

# Print the model
print(netD)
print(netG)

#training for GAN:
# Training adapted from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
real_label = 1.
fake_label = 0.
fixed_noise = torch.randn(64, nz, 1, 1)
def train_GANS(netD, netG, phase, dataloaders, dataset_sizes, criterion, optimizerD, optimizerG, num_epochs=25):
    since = time.time()
    # Keep track of how loss and accuracy evolves during training
    training_curves = {}
    training_curves['G'] = []
    training_curves['D'] = []
    best_model_wts = copy.deepcopy(netG.state_dict()) # keep the best weights stored separately
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        running_loss = 0.0
        for _, inputs in enumerate(dataloaders[phase], 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            optimizerD.zero_grad()
            # Format batch
            real_cpu = inputs
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            running_loss = running_loss + errG.item()
            training_curves['D'].append(errD.item())
            training_curves['G'].append(errG.item())
        if epoch < num_epochs/2:
              best_epoch = epoch
              best_loss = running_loss
              best_model_wts = copy.deepcopy(netG.state_dict())            
           
        if running_loss < best_loss:
              best_epoch = epoch
              best_loss = running_loss
              best_model_wts = copy.deepcopy(netG.state_dict())
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # load best model weights
    netG.load_state_dict(best_model_wts)
    return netD, netG, training_curves

#actual training
# Number of training epochs
num_epochs = 500 #Try 100 or so, once working
# Learning rate for optimizers
lrD = 0.0005
lrG = 0.0005
# Beta1 hyperparam for Adam optimizers
beta1D = 0.5
beta1G = 0.5
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=lrD, betas=(beta1D, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG, betas=(beta1G, 0.999))
# Train the model. We also will store the results of training to visualize
netD, netG, training_curves = train_GANS(netD, netG, 'zero', dataloaders, dataset_sizes, criterion, optimizerD, optimizerG, num_epochs)

