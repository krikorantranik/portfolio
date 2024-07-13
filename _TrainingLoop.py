import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
import torch.nn.functional as func
import math
import copy


model = ....
learning_rate = 0.1
num_epochs = 10

# loss and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#variable learning rate!!
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2, threshold=0.0001, min_lr=0.000001, cooldown=1)
print(model)

#training

#to keep the best model
best_model_wts = copy.deepcopy(model.state_dict()) 
best_acc = 0.0
best_epoch = 0

# Each epoch has a training and validation phase. I'll save the test as unseen data
phases = ['train', 'val']

# Keep track of how loss and accuracy evolves during training
training_curves = {}
for phase in phases:
        training_curves[phase+'_loss'] = []
        training_curves[phase+'_acc'] = []

#epoch loop
for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward step
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())
                    disploss = math.log10(loss.item()+0.00001)
                    #print step loss
                    learningrate = optimizer.param_groups[0]["lr"]
                    print(f'loss: {disploss:4f} LR: {learningrate:.8f} ')
                    

                    # backward + update weights (only in training)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #NOTE: learning rate inside the loop for epoch: because of the variable manual settings, should be classically outside
                if phase == 'train':
                    scheduler.step(loss)

                # stats
                running_loss += math.log10(loss.item()+0.00001) * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data).double()


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            training_curves[phase+'_loss'].append(epoch_loss)
            training_curves[phase+'_acc'].append(epoch_acc)

            print(f'{phase:5} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # copy the model if it is the best yet
            if phase == 'val' and epoch_acc > best_acc:
              best_epoch = epoch
              best_acc = epoch_acc
              best_model_wts = copy.deepcopy(model.state_dict())

print(f'Best val Acc: {best_acc:4f} at epoch {best_epoch}')

# load best model
model.load_state_dict(best_model_wts)
