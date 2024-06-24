import numpy as np
import torch
import torch.nn as nn
import time, copy




#NLP -> transformers

nds = nds.sample(frac=1).reset_index(drop=True)
train_sentences = nds.NLPtext[:100].reset_index(drop=True)
val_sentences = nds.NLPtext[100:120].reset_index(drop=True)
test_sentences = nds.NLPtext[120:220].reset_index(drop=True)

train_y = nds[1][:100].reset_index(drop=True)
val_y = nds[1][100:120].reset_index(drop=True)
test_y = nds[1][120:220].reset_index(drop=True)

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

max_sentencelength = max([len1, len2, len3])+3

#tensors
def create_input_tensor(sentence, word_dictionary):
    words = sentence.split(" ")
    tensor = torch.zeros(len(words), len(word_dictionary)+1)
    for idx in range(len(words)):
        word = words[idx]
        tensor[idx][word_dictionary[word]] = 1
    tensor = torch.nn.functional.pad(input=tensor, pad=(0, 0, 0,max_sentencelength-tensor.size()[0]), mode='constant', value=0)
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


#dataloaders
dataloaders = {'train': train_tensors,
               'val': val_tensors,
               'test': test_tensors}

dataset_sizes = {'train': len(train_tensors),
                 'val': len(val_tensors),
                 'test': len(test_tensors)}
print(f'dataset_sizes = {dataset_sizes}')

#positional encoder: information about position
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, maxlen):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, embedding_size, 2)* math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, embedding_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
    def forward(self, token_embedding, maxlen):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

#embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, dict_size, embedding_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(dict_size, embedding_size, max_norm=True)
        self.emb_size = embedding_size
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

#attention
class AttentionLayer(nn.Module):
    def __init__(self, embedding_size, nhead):
        super(AttentionLayer, self).__init__()
        self.embedding_size = embedding_size
        self.nhead = nhead
        self.emb_head  = int(self.embedding_size / self.nhead)
        self.Q = nn.Linear(self.emb_head , self.emb_head)
        self.K = nn.Linear(self.emb_head , self.emb_head)
        self.V = nn.Linear(self.emb_head , self.emb_head)
        self.O = nn.Linear(self.nhead*self.emb_head ,self.embedding_size) 
    def forward(self, mQ, mK, mV):
        batch_size = mK.size(0)
        seq_length = mK.size(1)
        key = mK.view(batch_size, seq_length, self.nhead, self.emb_head)  
        K_ = self.K(key) 
        K_ = K_.transpose(1,2) 
        query = mQ.view(batch_size, seq_length, self.nhead, self.emb_head)
        Q_ = self.Q(query) 
        Q_ = Q_.transpose(1,2)  
        value = mV.view(batch_size, seq_length, self.nhead, self.emb_head) 
        V_ = self.V(value)
        V_ = V_.transpose(1,2)  
        prod = torch.matmul(Q_, K_.transpose(-1,-2))
        prod = prod / math.sqrt(self.emb_head)
        scores = func.softmax(prod, dim=-1)
        scores = torch.matmul(scores, V_)
        concOut = scores.transpose(1,2).contiguous().view(batch_size, seq_length, self.emb_head*self.nhead)
        output = self.O(concOut)
        return output

#details of transformer
class TransfBlock(nn.Module):
    def __init__(self, embedding_size, nhead, dropout, internal_size):
        super(TransfBlock, self).__init__()
        self.attentionlayer = AttentionLayer(embedding_size, nhead)
        self.norm1 = nn.LayerNorm(embedding_size) 
        self.norm2 = nn.LayerNorm(embedding_size)
        self.nnetwork = nn.Sequential(
                          nn.Linear(embedding_size, internal_size),
                          nn.ReLU(),
                          nn.Linear(internal_size, internal_size),
                          nn.ReLU(),
                          nn.Linear(internal_size, embedding_size)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,mK,mQ,mV):
        attn_out = self.attentionlayer(mQ,mK,mV)
        attn_residual_out = attn_out + mV
        norm1_out = self.dropout(self.norm1(attn_residual_out)) 
        nn_out = self.nnetwork(norm1_out) 
        nn_residual_out = nn_out + norm1_out
        norm2_out = self.dropout(self.norm2(nn_residual_out))
        return norm2_out

#encoder object    
class Encoder(nn.Module):
    def __init__(self, embedding_size, layers, nhead, dropout, internal_size):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([TransfBlock(embedding_size=embedding_size, nhead=nhead, dropout=dropout,internal_size=internal_size) for i in range(layers)])
    def forward(self, x):
        for layer in self.layers:
            output = layer(x,x,x)
        return output     
    
#main transformer
class SentenceTransformer(nn.Module):
    def __init__(self, num_encoder_layers, embedding_size, nhead, dict_size, max_sentencelength, dropout,internal_size):
        super(SentenceTransformer, self).__init__()
        self.transformer = Encoder(embedding_size=embedding_size, layers=num_encoder_layers, nhead=nhead, dropout=dropout,internal_size=internal_size)
        self.generator = nn.Linear(embedding_size, 2)
        self.src_tok_emb = TokenEmbedding(dict_size=dict_size, embedding_size=embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size=embedding_size, maxlen=max_sentencelength, dropout=dropout)
    def forward(self, src):
        src_emb = self.positional_encoding(self.src_tok_emb(src), maxlen=max_sentencelength)
        outs = self.transformer(src_emb)
        outs = outs.mean(dim=1)
        return self.generator(outs)
    
#model
model = SentenceTransformer(num_encoder_layers=50, embedding_size=12, nhead=6, dict_size=len(english_dictionary)+3, max_sentencelength=max_sentencelength, dropout=0.00001,internal_size=20)
learning_rate = 0.01
num_epochs = 20
# loss and optimizer
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss for classification!
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
print(model)

#train model
best_model_wts = copy.deepcopy(model.state_dict()) # keep the best weights stored separately
best_acc = 0.0
best_epoch = 0

# Each epoch has a training, validation, and test phase
phases = ['train', 'val']

# Keep track of how loss and accuracy evolves during training
training_curves = {}
for phase in phases:
        training_curves[phase+'_loss'] = []
        training_curves[phase+'_acc'] = []

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
                # No need to flatten the inputs!
                inputs = inputs
                labels = labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.repeat(max_sentencelength).long())
                    disploss = loss.item()
                    #print(f'loss: {disploss:4f}')

                    # backward + update weights only if in training phase
                    if phase == 'train':
                        loss.backward()
                        #to avoid the exploding gradients issue and the loss become unstable
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.repeat(max_sentencelength).data).double() / inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            training_curves[phase+'_loss'].append(epoch_loss)
            training_curves[phase+'_acc'].append(epoch_acc)

            print(f'{phase:5} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model if it's the best accuracy (bas
            if phase == 'val' and epoch_acc > best_acc:
              best_epoch = epoch
              best_acc = epoch_acc
              best_model_wts = copy.deepcopy(model.state_dict())

print(f'Best val Acc: {best_acc:4f} at epoch {best_epoch}')

# load best model weights
model.load_state_dict(best_model_wts)





