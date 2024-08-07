from PrepareData import PrepareData
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import pickle

import Models.test as CNN
import params

def Data_Load():
     X = np.load('Dataset/data_samples.npy') # (samples' nember, 1024)
     Y = np.load('Dataset/data_labels.npy')  # (samples' number, 1)
     X, Y = torch.Tensor(X), torch.LongTensor(Y)
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

     train_set = TensorDataset(X_train, Y_train)
     test_set = TensorDataset(X_test, Y_test)

     train_batch = DataLoader(train_set, batch_size=32, shuffle=True)
     test_batch = DataLoader(test_set, batch_size=32, shuffle=True)
     pickle.dump(train_batch, open('Dataset/train_batch.dill', 'wb'))
     pickle.dump(test_batch, open('Dataset/test_batch.dill', 'wb'))

     return train_batch, test_batch

if __name__ == '__main__':
     train_batch, test_batch = Data_Load()
     # train_batch = pickle.load(open('Dataset/train_batch.dill', 'rb'))
     # test_batch = pickle.load(open('Dataset/test_batch.dill', 'rb'))
     model = CNN.CNN1D()
     for step, (sample, label) in enumerate(train_batch):
          feature = model(sample)
          print(feature.shape)
          break