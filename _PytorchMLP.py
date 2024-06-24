import numpy as np
import torch
import torch.nn as nn
import time, copy
import sklearn.metrics as metrics
import copy

#pandas data to tensors (Pytorch):
X_train = X_train.astype(float).to_numpy()
X_test = X_test.astype(float).to_numpy()
X_val = X_val.astype(float).to_numpy()
y_train = y_train.astype(float).to_numpy()
y_test = y_test.astype(float).to_numpy()
y_val = y_val.astype(float).to_numpy()
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

#straight forward MLP
    
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.dropout = nn.Dropout(0.03) 
        self.layers = nn.Sequential(
            nn.Linear(input_size, 250),
            nn.Linear(250, 500),
            nn.Linear(500, 1000),
            nn.Linear(1000, 1500),
            nn.ReLU(),
            self.dropout,
            nn.Linear(1500, 1500),
            nn.Sigmoid(),
            self.dropout,
            nn.Linear(1500, 1500),
            nn.ReLU(),
            self.dropout,
            nn.Linear(1500, 1500),
            nn.Sigmoid(),
            self.dropout,
            nn.Linear(1500, 1500),
            nn.ReLU(),
            self.dropout,
            nn.Linear(1500, 1500),
            nn.Sigmoid(),
            self.dropout,
            nn.Linear(1500, 1500),
            nn.ReLU(),
            self.dropout,
            nn.Linear(1500, 1500),
            nn.Sigmoid(),
            self.dropout,
            nn.Linear(1500, 2),
        )
    def forward(self, x):
        return self.layers(x)

#define model
model = SimpleClassifier()
print(model)

#MLP training
model = SimpleClassifier()
model.train()  
num_epochs=800
learning_rate = 0.00001
regularization = 0.0000001
#loss function
criterion = nn.CrossEntropyLoss()
#determine gradient values
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization) 
best_model_wts = copy.deepcopy(model.state_dict()) 
best_acc = 0.0
best_f1 = 0.0
best_epoch = 0
phases = ['train', 'val']
training_curves = {}
epoch_loss = 1
epoch_f1 = 0
epoch_acc = 0

for phase in phases:
    training_curves[phase+'_loss'] = []
    training_curves[phase+'_acc'] = []
    training_curves[phase+'_f1'] = []

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
            running_fp = 0
            running_tp = 0
            running_tn = 0
            running_fn = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.view(inputs.shape[0],-1)
                inputs = inputs
                labels = labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    #loss = criterion(predictions, labels)
                    #loss.requires_grad = True

                    # backward + update weights only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)
                running_fp += torch.sum((predictions != labels.data) & (predictions >= 0.5))   
                running_tp += torch.sum((predictions == labels.data) & (predictions >= 0.5))    
                running_fn += torch.sum((predictions != labels.data) & (predictions < 0.5))   
                running_tn += torch.sum((predictions == labels.data) & (predictions < 0.5))    
                print(f'Epoch {epoch+1}, {phase:5} Loss: {epoch_loss:.7f} F1: {epoch_f1:.7f} Acc: {epoch_acc:.7f} Partial loss: {loss.item():.7f} Best f1: {best_f1:.7f} ') 

            #if phase == 'train':
            #    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = (2*running_tp.double()) / (2*running_tp.double() + running_fp.double() + running_fn.double() + 0.0000000000000000000001)
            training_curves[phase+'_loss'].append(epoch_loss)
            training_curves[phase+'_acc'].append(epoch_acc)
            training_curves[phase+'_f1'].append(epoch_f1)

            print(f'Epoch {epoch+1}, {phase:5} Loss: {epoch_loss:.7f} F1: {epoch_f1:.7f} Acc: {epoch_acc:.7f} Best f1: {best_f1:.7f} ')

            # deep copy the model if it's the best accuracy (based on validation)
            if phase == 'val' and epoch_f1 >= best_f1:
                best_epoch = epoch
                best_acc = epoch_acc
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())

print(f'Best val F1: {best_f1:5f}, Best val Acc: {best_acc:5f} at epoch {best_epoch}')

# load best model weights
model.load_state_dict(best_model_wts)

#plot results on VALIDATION 
class_labels = ['0','1']
def classify_predictions(model, dataloader, cutpoint):
    model.eval()   # Set model to evaluate mode
    all_labels = torch.tensor([])
    all_scores = torch.tensor([])
    all_preds = torch.tensor([])
    for inputs, labels in dataloader:
        inputs = inputs
        labels = labels
        outputs = torch.softmax(model(inputs),dim=1)
        scores = torch.div(outputs[:,1],(outputs[:,1] + outputs[:,0])  )
        preds = (scores>=cutpoint).float()
        all_labels = torch.cat((all_labels, labels), 0)
        all_scores = torch.cat((all_scores, scores), 0)
        all_preds = torch.cat((all_preds, preds), 0)
    return all_preds.detach(), all_labels.detach(), all_scores.detach()

def plot_metrics(model, dataloaders, phase='val', cutpoint=0.5):
    preds, labels, scores = classify_predictions(model, dataloaders[phase], cutpoint)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    auc = metrics.roc_auc_score(labels, preds)

    disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    ind = np.argmin(np.abs(thresholds - 0.5))
    ind2 = np.argmin(np.abs(thresholds - 0.1))
    ind3 = np.argmin(np.abs(thresholds - 0.25))
    ind4 = np.argmin(np.abs(thresholds - 0.75))
    ind5 = np.argmin(np.abs(thresholds - 0.1))
    ax = disp.plot().ax_
    ax.scatter(fpr[ind], tpr[ind], color = 'red')
    ax.scatter(fpr[ind2], tpr[ind2], color = 'blue')
    ax.scatter(fpr[ind3], tpr[ind3], color = 'black')
    ax.scatter(fpr[ind4], tpr[ind4], color = 'orange')
    ax.scatter(fpr[ind5], tpr[ind5], color = 'green')
    ax.set_title('ROC Curve (green=0.1, orange=0.25, red=0.5, black=0.75, blue=0.9)')

    f1sc = metrics.f1_score(labels, preds)

    cm = metrics.confusion_matrix(labels, preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    ax = disp.plot().ax_
    ax.set_title('Confusion Matrix -- counts, f1: ' + str(f1sc))

    ncm = metrics.confusion_matrix(labels, preds, normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=ncm)
    ax = disp.plot().ax_
    ax.set_title('Confusion Matrix -- rates, f1: ' + str(f1sc))

    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    N, P = TN + FP, TP + FN
    ACC = (TP + TN)/(P+N)
    TPR, FPR, FNR, TNR = TP/P, FP/N, FN/P, TN/N
    print(f'\nAt default threshold:')
    print(f' TN = {TN:5},  FP = {FP:5} -> N = {N:5}')
    print(f' FN = {FN:5},  TP = {TP:5} -> P = {P:5}')
    print(f'TNR = {TNR:5.3f}, FPR = {FPR:5.3f}')
    print(f'FNR = {FNR:5.3f}, TPR = {TPR:5.3f}')
    print(f'ACC = {ACC:6.3f}')

    return cm, fpr, tpr, thresholds, auc

res = plot_metrics(model, dataloaders, phase='val', cutpoint=0.5)

#plot results on TEST 
bestcut = 0.1
import sklearn.metrics as metrics
class_labels = ['0','1']
def classify_predictions(model, dataloader, cutpoint):
    model.eval()   # Set model to evaluate mode
    all_labels = torch.tensor([])
    all_scores = torch.tensor([])
    all_preds = torch.tensor([])
    for inputs, labels in dataloader:
        inputs = inputs
        labels = labels
        outputs = torch.softmax(model(inputs),dim=1)
        scores = torch.div(outputs[:,1],(outputs[:,1] + outputs[:,0])  )
        preds = (scores>=cutpoint).float()
        all_labels = torch.cat((all_labels, labels), 0)
        all_scores = torch.cat((all_scores, scores), 0)
        all_preds = torch.cat((all_preds, preds), 0)
    return all_preds.detach(), all_labels.detach(), all_scores.detach()

def plot_metrics(model, dataloaders, phase='test', cutpoint=bestcut):
    preds, labels, scores = classify_predictions(model, dataloaders[phase], cutpoint)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    auc = metrics.roc_auc_score(labels, preds)
    disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    ind = np.argmin(np.abs(thresholds - 0.5))
    ind2 = np.argmin(np.abs(thresholds - 0.1))
    ind3 = np.argmin(np.abs(thresholds - 0.25))
    ind4 = np.argmin(np.abs(thresholds - 0.75))
    ind5 = np.argmin(np.abs(thresholds - 0.1))
    ax = disp.plot().ax_
    ax.scatter(fpr[ind], tpr[ind], color = 'red')
    ax.scatter(fpr[ind2], tpr[ind2], color = 'blue')
    ax.scatter(fpr[ind3], tpr[ind3], color = 'black')
    ax.scatter(fpr[ind4], tpr[ind4], color = 'orange')
    ax.scatter(fpr[ind5], tpr[ind5], color = 'green')
    ax.set_title('ROC Curve (green=0.1, orange=0.25, red=0.5, black=0.75, blue=0.9)')
    f1sc = metrics.f1_score(labels, preds)
    cm = metrics.confusion_matrix(labels, preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    ax = disp.plot().ax_
    ax.set_title('Confusion Matrix -- counts, f1: ' + str(f1sc))
    ncm = metrics.confusion_matrix(labels, preds, normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=ncm)
    ax = disp.plot().ax_
    ax.set_title('Confusion Matrix -- rates, f1: ' + str(f1sc))
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    N, P = TN + FP, TP + FN
    ACC = (TP + TN)/(P+N)
    TPR, FPR, FNR, TNR = TP/P, FP/N, FN/P, TN/N
    print(f'\nAt default threshold:')
    print(f' TN = {TN:5},  FP = {FP:5} -> N = {N:5}')
    print(f' FN = {FN:5},  TP = {TP:5} -> P = {P:5}')
    print(f'TNR = {TNR:5.3f}, FPR = {FPR:5.3f}')
    print(f'FNR = {FNR:5.3f}, TPR = {TPR:5.3f}')
    print(f'ACC = {ACC:6.3f}')
    return cm, fpr, tpr, thresholds, auc

res = plot_metrics(model, dataloaders, phase='test', cutpoint=bestcut)

