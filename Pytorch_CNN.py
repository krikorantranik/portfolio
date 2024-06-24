import numpy as np
import torch
import torch.nn as nn
import time, copy



#CNN classifier
from torch.nn.modules.flatten import Flatten
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.dropout = nn.Dropout(0.05) 
        self.pipeline = nn.Sequential(
            #in channels is 1, because it is grayscale
            nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 5, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 5, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 5, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 5, kernel_size = 5, stride = 1, padding=1),
            nn.ReLU(),
            self.dropout,
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Linear(500, 50),
            nn.ReLU(),
            self.dropout,
            nn.Linear(50, 50),
            nn.ReLU(),
            self.dropout,
            nn.Linear(50, 10),
            nn.ReLU(),
            self.dropout,
            nn.Linear(10, 10),
            nn.ReLU(),
            self.dropout,
            nn.Linear(10, 5),
        )

    def forward(self, x):
        return self.pipeline(x)

model = CNNClassifier()
print(model)


#train CNN classifier
#model training
import copy

model.train()  

num_epochs=50
learning_rate = 0.0001
regularization = 0.0000001


#loss function
criterion = nn.CrossEntropyLoss()

#determine gradient values
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization) 
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

best_model_wts = copy.deepcopy(model.state_dict()) 
best_acc = 0.0
best_epoch = 0
phases = ['train', 'val']
training_curves = {}
epoch_loss = 1
epoch_acc = 0

for phase in phases:
    training_curves[phase+'_loss'] = []
    training_curves[phase+'_acc'] = []

for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in phases:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs
                labels = labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.type(torch.LongTensor))

                    # backward + update weights only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)  
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            training_curves[phase+'_loss'].append(epoch_loss)
            training_curves[phase+'_acc'].append(epoch_acc)
            print(f'Epoch {epoch+1}, {phase:5} Loss: {epoch_loss:.7f} Acc: {epoch_acc:.7f} ')

            # deep copy the model if it's the best accuracy (based on validation)
            if phase == 'val' and epoch_acc >= best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

print(f'Best val Acc: {best_acc:5f} at epoch {best_epoch}')

# load best model weights
model.load_state_dict(best_model_wts)

