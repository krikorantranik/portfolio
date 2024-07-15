import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
import torch.nn.functional as func
import math
import copy

#one hot encodding, train and val

sizeTrain = 10000
sizeVandT = 500
batch_size=1

newds = newds.sample(frac=1).reset_index(drop=True)
train_sentences = newds.newtext[:sizeTrain].reset_index(drop=True)
val_sentences = newds.newtext[sizeTrain:sizeTrain+sizeVandT].reset_index(drop=True)
test_sentences = newds.newtext[sizeTrain+sizeVandT:sizeTrain+sizeVandT+sizeVandT].reset_index(drop=True)

train_y = newds['label'][:sizeTrain].reset_index(drop=True)
val_y = newds['label'][sizeTrain:sizeTrain+sizeVandT].reset_index(drop=True)
test_y = newds['label'][sizeTrain+sizeVandT:sizeTrain+sizeVandT+sizeVandT].reset_index(drop=True)

# Using this function we will create a dictionary to use for our one hot encoding vectors
def add_words_to_dict(word_dictionary, word_list, sentences):
    max_sentencelength = 0
    for sentence in sentences:
        if (len(sentence.split(" ")) > max_sentencelength):
            max_sentencelength = len(sentence.split(" "))
        for word in sentence.split(" "):
            if word in word_dictionary:
                continue
            else:
                word_list.append(word)
                word_dictionary[word] = len(word_list)-1
    return max_sentencelength

english_dictionary = {}
english_list = []
len1 = add_words_to_dict(english_dictionary, english_list, train_sentences)
len2 = add_words_to_dict(english_dictionary, english_list, val_sentences)
len3 = add_words_to_dict(english_dictionary, english_list, test_sentences)

max_sentencelength = max([len1, len2, len3])

#dataset for the model

def create_input_tensor(sentence, word_dictionary):
    words = sentence.split(" ")
    tensor = torch.zeros(len(words), len(word_dictionary)+1)
    for idx in range(len(words)):
        word = words[idx]
        tensor[idx][word_dictionary[word]] = 1
    tensor = torch.nn.functional.pad(input=tensor, pad=(0, 0, 0,max_sentencelength-tensor.size()[0]), mode='constant', value=-1)
    return tensor

train_tensors = []
for i in range(0,len(train_sentences)):
  train_tensors.append([create_input_tensor(train_sentences[i], english_dictionary),torch.Tensor(train_y)[i]])

val_tensors = []
for i in range(0,len(val_sentences)):
  val_tensors.append([create_input_tensor(val_sentences[i], english_dictionary),torch.Tensor(val_y)[i]])

test_tensors = []
for i in range(0,len(test_sentences)):
  test_tensors.append([create_input_tensor(test_sentences[i], english_dictionary),torch.Tensor(test_y)[i]])

dataloaders = {'train': DataLoader(train_tensors, batch_size=batch_size, shuffle=True),
               'val': DataLoader(val_tensors, batch_size=batch_size, shuffle=True),
               'test': DataLoader(test_tensors, batch_size=batch_size, shuffle=True)}

dataset_sizes = {'train': len(train_tensors),
                 'val': len(val_tensors),
                 'test': len(test_tensors)}

i = 0
samples =[]
for smpinput, smplabels in dataloaders["train"]: 
    samples.append(smpinput)

print(f'dataset_sizes = {dataset_sizes}')


#Transformer:

#embedding from one hot to specific size
class TokenEmbedding(nn.Module):
    def __init__(self, dict_size, embedding_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Linear(dict_size, embedding_size)
        torch.nn.init.normal_(self.embedding.weight,mean=0, std=1) 
        self.dimm = embedding_size
    def forward(self, tokens):
        emb = self.embedding(tokens) * math.sqrt(self.dimm)
        return emb

#positional encodding, added to the previous encodding (classical sine and cosine)
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, maxlen):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, embedding_size, 2)* math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, embedding_size))
        #creates positional information using cosine and sine functions
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2).transpose(0,1)
        #add dropout, as always
        self.dropout = nn.Dropout(dropout)
        self.maxlen = maxlen
        self.register_buffer('pos_embedding', pos_embedding)
    def forward(self, token_embedding):
        #add positional information to the original embedding
        b = self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        return b

#multi head attention layer
class AttentionLayer(nn.Module):
    def __init__(self, embedding_size, nhead):
        super(AttentionLayer, self).__init__()
        self.embedding_size = embedding_size
        self.nhead = nhead
        #embeddings per head
        self.emb_head  = int(self.embedding_size / self.nhead)
        self.Q = nn.Linear(self.emb_head , self.emb_head)
        torch.nn.init.normal_(self.Q.weight,mean=0, std=1) 
        self.K = nn.Linear(self.emb_head , self.emb_head)
        torch.nn.init.normal_(self.K.weight,mean=0, std=1) 
        self.V = nn.Linear(self.emb_head , self.emb_head)
        torch.nn.init.normal_(self.V.weight,mean=0, std=1) 
        self.O = nn.Linear(self.nhead*self.emb_head ,self.embedding_size) 
        torch.nn.init.normal_(self.O.weight,mean=0, std=1) 
        self.norm1 = nn.LayerNorm(embedding_size) 
    def forward(self, mQ, mK, mV):
        batch_size = mK.size(0)
        seq_length = mK.size(1)
        #matrix of keys
        key = mK.view(batch_size, seq_length, self.nhead, self.emb_head)  
        K_ = self.K(key) 
        K_ = K_.transpose(1,2)
        #matrix of queries
        query = mQ.view(batch_size, seq_length, self.nhead, self.emb_head)
        Q_ = self.Q(query) 
        Q_ = Q_.transpose(1,2)  
        #matrix of values
        value = mV.view(batch_size, seq_length, self.nhead, self.emb_head) 
        V_ = self.V(value)
        V_ = V_.transpose(1,2)  
        #perform QKt/sqrt(dimk)
        prod = torch.matmul(Q_, K_.transpose(-1,-2))
        prod = prod / math.sqrt(self.emb_head) 
        #apply softmax (turn to probabilities)
        scores = func.softmax(prod, dim=-1)
        #multiply by the values
        scores = torch.matmul(scores, V_)
        #format and normalize
        concOut = scores.transpose(1,2).contiguous().view(batch_size, seq_length, self.emb_head*self.nhead)
        output = self.O(concOut)
        normout = self.norm1(output)
        return normout + mV

#feed forward MLP
class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout, internal_size):
        super(FeedForward, self).__init__() 
        self.l0 = nn.Linear(embedding_size, internal_size)
        torch.nn.init.normal_(self.l0.weight,mean=0, std=1) 
        self.n0 = nn.ReLU()
        self.l1 = nn.Linear(internal_size, internal_size)
        torch.nn.init.normal_(self.l1.weight,mean=0, std=1) 
        self.n1 = nn.ReLU()
        self.l2 = nn.Linear(internal_size, 2*internal_size)
        torch.nn.init.normal_(self.l2.weight,mean=0, std=1) 
        self.n2 = nn.ReLU()
        self.l3 = nn.Linear(2*internal_size, 2*internal_size)
        torch.nn.init.normal_(self.l3.weight,mean=0, std=1) 
        self.n3 = nn.ReLU()
        self.l4 = nn.Linear(2*internal_size, 2*internal_size)
        torch.nn.init.normal_(self.l4.weight,mean=0, std=1) 
        self.n4 = nn.ReLU()
        self.l5 = nn.Linear(2*internal_size, internal_size)
        torch.nn.init.normal_(self.l5.weight,mean=0, std=1) 
        self.n5 = nn.ReLU()
        self.l6 = nn.Linear(internal_size, internal_size)
        torch.nn.init.normal_(self.l6.weight,mean=0, std=1) 
        self.n6 = nn.ReLU()
        self.l7 = nn.Linear(internal_size, embedding_size)
        torch.nn.init.normal_(self.l7.weight,mean=0, std=1) 
        self.n7 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_size) 
    def forward(self,ds):
        nn_out = self.n7(self.l7(self.n6(self.l6(self.n5(self.l5(self.n4(self.l4(self.n3(self.l3(self.n2(self.l2(self.n1(self.l1(self.n0(self.l0(ds)))))))))))))))) 
        norm_out = self.dropout(self.norm1(nn_out))
        return norm_out + ds

#encoder wrapper
class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, nhead, dropout, internal_size):
        super(EncoderBlock, self).__init__()
        self.attentionlayer = AttentionLayer(embedding_size=embedding_size, nhead=nhead)
        self.feedforward = FeedForward(embedding_size=embedding_size, dropout=dropout, internal_size=internal_size)
    def forward(self,mK,mQ,mV):
        attn_out = self.attentionlayer(mQ,mK,mV)
        nn_out = self.feedforward(attn_out) 
        return nn_out
class Encoder(nn.Module):
    def __init__(self, dict_size, max_sentencelength, embedding_size, layers, nhead, dropout, internal_size):
        super(Encoder, self).__init__()
        self.rlayers = nn.ModuleList([EncoderBlock(embedding_size=embedding_size, nhead=nhead, dropout=dropout,internal_size=internal_size) for i in range(layers)])
    def forward(self, x):
        for layer in self.rlayers:
            output = layer(x,x,x)
        return output    

#decoder wrapper
class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, nhead, dropout, internal_size):
        super(DecoderBlock, self).__init__()
        self.attentionlayer = AttentionLayer(embedding_size=embedding_size, nhead=nhead)
        self.feedforward = FeedForward(embedding_size=embedding_size, dropout=dropout, internal_size=internal_size)
    def forward(self,mK,mQ,mV):
        attn_out = self.attentionlayer(mQ,mK,mV)
        nn_out = self.feedforward(attn_out) 
        return nn_out
class Decoder(nn.Module):
    def __init__(self, embedding_size, layers, nhead, dropout, internal_size):
        super(Decoder, self).__init__()
        self.rlayers = nn.ModuleList([DecoderBlock(embedding_size=embedding_size, nhead=nhead, dropout=dropout,internal_size=internal_size) for i in range(layers)])
    def forward(self, x, y):
        for layer in self.rlayers:
            output = layer(x,x,y)
        return output  

#final MLP (feedforward)
class Final(nn.Module):
    def __init__(self, embedding_size, dropout, internal_size, output_size):
        super(Final, self).__init__()
        self.l0 = nn.Linear(embedding_size, internal_size)
        torch.nn.init.normal_(self.l0.weight,mean=0, std=1) 
        self.n0 = nn.ReLU()
        self.l1 = nn.Linear(internal_size, 2*internal_size)
        torch.nn.init.normal_(self.l1.weight,mean=0, std=1) 
        self.n1 = nn.ReLU()
        self.l2 = nn.Linear(2*internal_size, output_size)
        torch.nn.init.normal_(self.l2.weight,mean=0, std=1) 
        #no softmax! it is accounted for in the loss
        self.dropout = nn.Dropout(dropout)
    def forward(self,ds):
        dss = ds.mean(dim=1)
        dsss = self.l2(self.n1(self.l1(self.n0(self.l0(dss)))))
        nn_out = self.dropout(dsss) 
        return nn_out

#the actual model
class TransformerModel(nn.Module):
    def __init__(self, dict_sizef, max_sentencelengthf, embedding_sizef, EncoderLayersf, DecoderLayersf, nheadf, dropoutf, Encoder_internal_sizef, Decoder_internal_sizef, Final_internal_sizef, output_sizef):
        super(TransformerModel, self).__init__()
        self.src_tok_emb = TokenEmbedding(dict_size=dict_sizef, embedding_size=embedding_sizef)
        self.positional_encoding = PositionalEncoding(embedding_size=embedding_sizef, maxlen=max_sentencelengthf, dropout=dropoutf)
        self.encoderObj1 = Encoder(dict_size=dict_sizef, max_sentencelength=max_sentencelengthf, embedding_size=embedding_sizef, layers=EncoderLayersf, nhead=nheadf, dropout=dropoutf, internal_size=Encoder_internal_sizef)
        self.encoderObj2 = Encoder(dict_size=dict_sizef, max_sentencelength=max_sentencelengthf, embedding_size=embedding_sizef, layers=EncoderLayersf, nhead=nheadf, dropout=dropoutf, internal_size=Encoder_internal_sizef)
        self.decoderObj = Decoder(embedding_size=embedding_sizef, layers=DecoderLayersf, nhead=nheadf, dropout=dropoutf, internal_size=Decoder_internal_sizef)
        self.final = Final(embedding_size=embedding_sizef, dropout=dropoutf, internal_size=Final_internal_sizef, output_size=output_sizef)
    def forward(self, x, y):
        encoded1 = self.positional_encoding(self.src_tok_emb(x))
        encoded2 = self.positional_encoding(self.src_tok_emb(y))
        encoded1 = self.encoderObj1(encoded1)
        encoded2 = self.encoderObj2(encoded2)
        decoded = self.decoderObj(encoded1, encoded2)
        outs = self.final(decoded)
        return outs

#declaration and params
dict_size = len(english_dictionary)+1
max_sentencelength = max([len1, len2, len3])
embedding_size = 50
EncoderLayers = 50
DecoderLayers = 50
nhead = 5
dropout = 0.0000001
Encoder_internal_size = 500
Decoder_internal_size = 300
Final_internal_size = 300
output_size = 2
model = TransformerModel(dict_sizef=dict_size, max_sentencelengthf=max_sentencelength, embedding_sizef=embedding_size, EncoderLayersf=EncoderLayers, DecoderLayersf=DecoderLayers, nheadf=nhead, dropoutf=dropout, Encoder_internal_sizef=Encoder_internal_size, Decoder_internal_sizef=Decoder_internal_size, Final_internal_sizef=Final_internal_size, output_sizef=output_size)
learning_rate = 0.1
num_epochs = 10
# loss and optimizer
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss for classification!
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2, threshold=0.0001, min_lr=0.000001, cooldown=1)
print(model)

#train (variable learning rate)

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
                    outputs = model(inputs,inputs)
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


