from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle

from PrepareData import PrepareData
import Models.test as CNN
from params import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def Data_Load():
     X = np.load('Dataset/data_samples.npy') # (samples' nember, 1024)
     Y = np.load('Dataset/data_labels.npy')  # (samples' number, 1)
     X, Y = torch.Tensor(X), torch.LongTensor(Y)
     # X, Y = torch.Tensor(X), torch.Tensor(Y)
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

     train_set = TensorDataset(X_train, Y_train)
     test_set = TensorDataset(X_test, Y_test)

     train_batch = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
     test_batch = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
     pickle.dump(train_batch, open('Dataset/train_batch.dill', 'wb'))
     pickle.dump(test_batch, open('Dataset/test_batch.dill', 'wb'))

     return train_batch, test_batch

train_batch, test_batch = Data_Load()
# train_batch = pickle.load(open('Dataset/train_batch.dill', 'rb')).to(device)
# test_batch = pickle.load(open('Dataset/test_batch.dill', 'rb')).to(device)
model = CNN.CNN1D().to(device)
def train(max_epoch = args.max_epoch):
     for epoch in range(max_epoch):
          model.train()
          loss_fn = nn.CrossEntropyLoss()
          optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
          for step, (sample, label) in enumerate(train_batch):
               sample, label = sample.to(device), label.to(device)
               pred = model(sample)
               loss = loss_fn(pred, label.view(-1))

               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

               print('Loss: {:.4f} Acc: {}'.format(loss.item(), sum(torch.max(pred, dim=1).indices == label.view(1, -1))))

               break

if __name__ == '__main__':
     train()