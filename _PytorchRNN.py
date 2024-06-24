import numpy as np
import torch
import torch.nn as nn
import time, copy
import copy




#RNN
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.RNN = nn.RNN(input_size, hidden_size, num_layers = 2, dropout = 0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        pass
    def forward(self, input):
        output, hn = self.RNN(input)
        output = self.fc(output)
        return output, hn
    
model = RNNClassifier(insize,8,2)

#model training
import copy
model.train()  
num_epochs=15
learning_rate = 0.007
regularization = 0.001
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
                    
                    loss = 0
                    output, hidden = model(inputs)
                    output = torch.mean(output,1)
                    _, predictions = torch.max(output, 1)
                    loss = criterion(output, labels.type(torch.LongTensor))
                    
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


