#!/usr/bin/env python

### new model
import os
import pickle
import pandas as pd
import numpy as np
import math
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

class time_series(Dataset):
    def __init__(self, df_series, seq_length=480):
        self.X = torch.tensor(np.array(df_series), requires_grad=False).float()
        self.seq_length = seq_length
    def __len__(self):
        return len(self.X) - self.seq_length - 5
    def __getitem__(self, idx):
        seq = self.X[idx:(idx + self.seq_length),:]
        pred = self.X[(idx + self.seq_length),:], self.X[(idx + self.seq_length + 1),:],\
                self.X[(idx + self.seq_length + 2),:], self.X[(idx + self.seq_length +3),:],\
                self.X[(idx + self.seq_length + 4),:]
        return seq, pred
    
class gcn_block(nn.Module):
    def __init__(self, X_size, A_hat, batch_size, cuda, bias=True):
        super(gcn_block, self).__init__()
        self.batch_size = batch_size
        self.cuda1 = cuda
        self.A_hat = A_hat
        self.weight = nn.parameter.Parameter(torch.FloatTensor(X_size, math.ceil(X_size/2)))
        var = 2./(self.weight.size(1)+self.weight.size(0))
        self.weight.data.normal_(0,var)
        self.weight2 = nn.parameter.Parameter(torch.FloatTensor(math.ceil(X_size/2), math.ceil(X_size/4)))
        var2 = 2./(self.weight2.size(1)+self.weight2.size(0))
        self.weight2.data.normal_(0,var2)
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(math.ceil(X_size/2)))
            self.bias.data.normal_(0,var)
            self.bias2 = nn.parameter.Parameter(torch.FloatTensor(math.ceil(X_size/4)))
            self.bias2.data.normal_(0,var2)
        else:
            self.register_parameter("bias", None)
        self.fc1 = nn.Linear(math.ceil(X_size/4),1)
        
    def forward(self, X): ### 2-layer GCN architecture, input = batch_size X num_nodes X seq_features
        A = []
        for batch in range(self.batch_size):
            X1 = X[batch,:,:].squeeze(dim=0); #print(X.shape)
            X1 = torch.mm(X1, self.weight)
            if self.bias is not None:
                X1 = (X1 + self.bias)
            X1 = F.relu(torch.mm(self.A_hat, X1))
            X1 = torch.mm(X1, self.weight2)
            if self.bias2 is not None:
                X1 = (X1 + self.bias2)
            X1 = F.relu(torch.mm(self.A_hat, X1))
            X1 = self.fc1(X1)
            A.append(X1)
        A = torch.stack([b for b in A], dim=0).squeeze(dim=2)
        return A    # output = batch_size X num_nodes

class gcn_lstm(nn.Module):
    def __init__(self, X_size, seq_len, A_hat, batch_size, cuda, bias=True,\
                 input_size=1329, lstm_input=256, lstm_hidden=128): # X_size = num features
        super(gcn_lstm, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lstm_size = lstm_hidden
        self.lstm_input = lstm_input
        self.input_size = input_size
        self.cuda1 = cuda
        ### gcn
        if self.cuda1:
            self.A_hat = A_hat.cuda()
        else:
            self.A_hat = A_hat
        for block in range(self.seq_len):
            setattr(self, "gcn_%i" % block, gcn_block(X_size=(block + 1), A_hat=self.A_hat,\
                                                      batch_size=self.batch_size, cuda=self.cuda1))
        ### LSTM
        self.hidden1 = self.init_hidden_lstm()
        ## input batch, seq_len, hidden_size, output batch, seq_len, hidden_size
        self.lstm1 = nn.LSTM(input_size=lstm_input, hidden_size=lstm_hidden,\
                             batch_first=True)
        self.fc_embed = nn.Linear(self.input_size, self.lstm_input)
        self.fc1 = nn.Linear(lstm_hidden*seq_len, self.input_size)
        self.fc2 = nn.Linear(lstm_hidden*seq_len, self.input_size)
        self.fc3 = nn.Linear(lstm_hidden*seq_len, self.input_size)
        self.fc4 = nn.Linear(lstm_hidden*seq_len, self.input_size)
        self.fc5 = nn.Linear(lstm_hidden*seq_len, self.input_size)
        
    def init_hidden_lstm(self):
        grad = True
        if self.cuda1:
            return Variable(torch.randn(1, self.batch_size, self.lstm_size),\
                        requires_grad=grad).cuda(),\
                Variable(torch.randn(1, self.batch_size, self.lstm_size),\
                        requires_grad=grad).cuda()
        else:
            return Variable(torch.randn(1, self.batch_size, self.lstm_size),\
                            requires_grad=grad),\
                    Variable(torch.randn(1, self.batch_size, self.lstm_size),\
                            requires_grad=grad)
        
    def forward(self, seq): 
        ### input = batch_size X seq_length X num_nodes
        A = []
        for q in range(self.seq_len):
            X = seq[:, :(q+1), :].permute(0,2,1) # batch_size X num_nodes X seq_len subsequence
            X = getattr(self, "gcn_%i" % q)(X)
            A.append(X)
        A = torch.stack([b for b in A], dim=1) # batch_size X seq_length X num_nodes
        A = torch.sigmoid(self.fc_embed(A)) 
        #print(A[0,1,:]); print(A[1,1,:])
        ### LSTM
        # stack into batch_size X seq_length X nodes
        A, _ = self.lstm1(A, self.hidden1) # input = batch X seq_len X num_nodes
        A = torch.sigmoid(A)
        A = A.reshape(self.batch_size,-1)
        X1 = torch.tanh(self.fc1(A))
        X2 = torch.tanh(self.fc2(A))
        X3 = torch.tanh(self.fc3(A))
        X4 = torch.tanh(self.fc4(A))
        X5 = torch.tanh(self.fc5(A))
        return X1, X2, X3, X4, X5

class seriesloss(torch.nn.Module):
    def __init__(self):
        super(seriesloss, self).__init__()
    
    def forward(self, X1, X2, X3, X4, X5, y1, y2, y3, y4, y5): # X = net output, y = actual
        c1 = nn.MSELoss(reduction='mean')
        loss1 = c1(X1, y1)
        c2 = nn.MSELoss(reduction='mean')
        loss2 = c2(X2, y2)
        c3 = nn.MSELoss(reduction='mean')
        loss3 = c3(X3, y3)
        c4 = nn.MSELoss(reduction='mean')
        loss4 = c4(X4, y4)
        c5 = nn.MSELoss(reduction='mean')
        loss5 = c5(X5, y5)
        ### higher weights for loss at more recent time-steps
        total_loss = 5*loss1 + 4*loss2 + 3*loss3 + 2*loss4 + loss5
        return total_loss
    
### log-transforms, then standardize features
def transform_data(df_series, scaler, inverse=False):
    if inverse:
        ### inverse standardize
        df_series = scaler.inverse_transform(df_series)
        df = load_pickle("df_processed.pkl")
        cols = df["geohash6"].unique()
        start = df.index.min()
        indexes = [start + 15*i for i in\
                   range(int((df.index.max() - start)/15) + 1)]
        del df
        df_series = pd.DataFrame(data=df_series,index=indexes,columns=cols)
        ### inverse log-transform
        for col in df_series.columns:
            df_series[col] = df_series[col].apply(lambda x: 10**x)
    else:
        ### log-transform the demand values
        for col in df_series.columns:
            df_series[col] = df_series[col].apply(lambda x: math.log10(x))
        ### standardize the log-transformed features
        df_series = scaler.fit_transform(df_series)
    return df_series
    
### Loads model and optimizer states
def load(net, optimizer, load_best=True):
    base_path = "./data/"
    if load_best == False:
        checkpoint = torch.load(os.path.join(base_path,"checkpoint.pth.tar"))
    else:
        checkpoint = torch.load(os.path.join(base_path,"model_best.pth.tar"))
    start_epoch = checkpoint['epoch']
    best_pred = checkpoint['best_acc']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch, best_pred

if __name__ == "__main__":
    data = "./data/"
    df_series = load_pickle("df_series_relative.pkl")
    ### Loads graph and get adjacency matrix
    G = load_pickle("Graph.pkl")
    A = nx.to_numpy_matrix(G, weight="weight"); A = A + np.eye(G.number_of_nodes())
    degrees = []
    for d in G.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1]**(-0.5))
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes()) # Features are just identity matrix
    X = torch.from_numpy(X).float()
    A_hat = degrees@A@degrees; del degrees
    A_hat = torch.tensor(A_hat, requires_grad=False).float()
    cuda = torch.cuda.is_available()
    #cuda = False
    '''
    ### transform data
    scaler = StandardScaler()
    df_series = transform_data(df_series, scaler, inverse=False)
    save_as_pickle("standardscaler.pkl", scaler)
    '''
    df_series = np.array(df_series[1:])
    batch_size = 1
    seq_len = 480
    trainset = time_series(df_series, seq_length=seq_len)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    ## initialize net, loss function and optimizer
    net = gcn_lstm(X_size=X.shape[1], seq_len=seq_len, A_hat=A_hat, batch_size=batch_size, cuda=cuda)
    del A_hat, X, df_series, A
    criterion = seriesloss()
    optimizer = optim.Adam(net.parameters(), lr=0.02)
    if cuda:
        net.cuda()
    try:
        start_epoch, best_loss = load(net, optimizer, load_best=False)
    except:
        start_epoch = 0; best_loss = 999
    stop_epoch = 100; end_epoch = 7000
    
    net.train()
    try:
        losses_per_epoch = load_pickle("losses_per_epoch.pkl")
    except:
        losses_per_epoch = []
    total_loss = 0.0
    for e in range(start_epoch, end_epoch):
        losses_per_batch = []
        for i, (seq, pred) in enumerate(train_loader):
            if cuda:
                seq = seq.cuda(); pred = [p.cuda() for p in pred]
            optimizer.zero_grad()
            output = net(seq)
            loss = criterion(output[0],output[1],output[2],output[3],output[4],\
                             pred[0],pred[1],pred[2],pred[3],pred[4])
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
            if i % 100 == 99: # print every 50 mini-batches of size = batch_size
                losses_per_batch.append(total_loss/100)
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.7f' %
                      (e, (i + 1)*batch_size, len(trainset), total_loss/100))
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        print("Losses at Epoch %d: %.7f" % (e, losses_per_epoch[-1]))
        if losses_per_epoch[-1] < best_loss:
            best_loss = losses_per_epoch[-1]
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': best_loss,\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("./data/" ,"model_best.pth.tar"))
        if (e % 1) == 0:
            save_as_pickle("losses_per_epoch.pkl", losses_per_epoch)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': best_loss,\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("./data/" ,"checkpoint.pth.tar"))
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(losses_per_epoch))], losses_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.set_title("Loss vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/", "loss_vs_epoch.png"))