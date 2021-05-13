# -*- coding: utf-8 -*-
"""
Created on May 10 2021
@author: Agam Chopra
"""
import pandas as pd
import nltk
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
#%%
# Load and Preprocess Data:
    # definitions->
def load_data(file_path):  
    return pd.read_csv (r'%s'%(file_path)).values[:,1:]
    
def processed_data(data):
    return data[:,0], [sents.lower() for sents in data[:,1]]

def tokenize_sen(sen):
    stop_words = set(stopwords.words("english"))
    return [word for word in [''.join(e for e in string if e.isalnum()) for string in [x.replace('\\n','') for x in nltk.word_tokenize(sen)]] if word not in stop_words and word != '']
#%%
title,text = processed_data(load_data(file_path = "https://raw.githubusercontent.com/AgamChopra/bia660-team4-spring-2021/main/ICOtext.csv"))
tok_sens = [tokenize_sen(sen) for sen in text]
#%%
# Embeddings:
    # Setup
CONTEXT_SIZE = 4
EMBEDDING_DIM = 8
test_sentence = []
[[test_sentence.append(word) for word in tok_sens[i]]for i in range(6)] # First 6 documents
grams = [[([test_sentence[i], test_sentence[i + 1], test_sentence[i + 3], test_sentence[i + 4]], test_sentence[i + 2]) for i in range(len(tok_sens[j]) - 5)] for j in range(6)]
ngrams = []
[[ngrams.append(i) for i in j]for j in grams]
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
#%%
    # Definition
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
#%%
#import sys 
#sys.path.append("PATH")
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
#model.load_state_dict(torch.load(r'PATH.pth')) # Load pretrained model parameters
#%%   
# Train Embeddings:
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(20):
    total_loss = 0
    for context, target in ngrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 1 == 0:
        print('epoch:', epoch,'loss:', total_loss)
    losses.append(total_loss)
plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Embedding Loss')
plt.show()
#torch.save(model.state_dict(),r'PATH.pth') # Save the model parameters
#%%
# Extract vectors:
    # definitions
def extract_docs(x, doc_number = 6, max_words=5000): # x = tok_sens #looks up trained vector representation of words in the range of docs provided.
    docs = [x[i][:max_words] for i in range(doc_number)]
    docx = [[model.embeddings.weight[word_to_ix[word]]for word in doc]for doc in docs]
    xx = [[j.detach() for j in i] for i in docx]
    return xx
def data_vectorizer(xx, num_examples, embd_dim, max_words): # converts data into readable format for the models.
    x = torch.zeros(num_examples, embd_dim, max_words)
    for i in range(num_examples):
        for j in range(len(xx[i])):
                x[i,:,j] = xx[i][j]
    return(x)
#%%
xx = extract_docs(tok_sens, 6, 5000)
x = data_vectorizer(xx, 6, EMBEDDING_DIM, 5000)
#%%
#CNN:
def conv_block(in_c, out_c, ker, std, kp, sp):
    out = nn.Sequential(nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=ker, stride=std),nn.BatchNorm1d(out_c),nn.ReLU(inplace=True),nn.MaxPool1d(kernel_size=kp, stride=sp))
    return out

def conv1_block(in_c, out_c):
    out = nn.Sequential(nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1),nn.BatchNorm1d(out_c))#,nn.Softmax(dim=1))
    return out

class classifier_conv(nn.Module):
    def __init__(self):
        super(classifier_conv, self).__init__()
        self.f1 = conv_block(8,16,8,2,9,2)
        self.f2 = conv_block(16,32,9,2,9,5)
        self.f3 = conv_block(32,64,4,2,4,3)
        self.f4 = conv_block(64,128,6,3,3,1)
        self.f5 = conv1_block(128, 3)
        
    def forward(self, x):
        print(x.shape)
        y = self.f1(x)
        print(y.shape)
        y = self.f2(y)
        print(y.shape)
        y = self.f3(y)
        print(y.shape)
        y = self.f4(y)
        print(y.shape)
        y = self.f5(y)
        print(y.shape)
        return y

class conv_model():
    def __init__(self):
        self.model = classifier_conv()
            
    def train(self,x, y, epochs = 100, lr = 0.001): # x and y are in torch tensor format
        # prediction b4 training
        self.model.eval()
        y_pred = self.model(x)
        print(y_pred)
        self.model.train()
        loss_list = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay= lr/100)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Forward pass
            y_pred = self.model(x)
            loss = criterion(y_pred.squeeze(), torch.reshape(y,[len(y_pred.squeeze())]).float())
            loss_list.append(loss)
            if epoch%10 == 0:
                print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            # Backward pass
            loss.backward()
            optimizer.step()
        return loss_list
    
    def evaluate(self, x):
        self.model.eval()
        y_pred = self.model(x)
        return y_pred
#%%
y = [] # need labels in torch.tensor format!
conv_model = conv_model()
_ = conv_model.train(x,y,10,0.001)
conv_model.evaluate(x)
#%%
#LSTM:
    #...