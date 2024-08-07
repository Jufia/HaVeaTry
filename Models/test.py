import torch.nn as nn
import numpy as np
import pickle


class CNN1D(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # (1, 1024)
            nn.BatchNorm1d(32),
            nn.ReLU(),  # (32, 1024)
            nn.MaxPool1d(2)  # (32, 512)
        )

    def forward(self, x):  # x:(batch, seq_long)
        # input size 2D (unbatched) (channel, seq_long) or 3D (batched) (batch_size, channel, seq_long)
        x = x[:, np.newaxis]
        x = self.layer1(x)
        return x


if __name__ == '__main__':
    train_batch = pickle.load(open('../Dataset/train_batch.dill', 'rb'))
    model = CNN1D()
    for step, (sample, label) in enumerate(train_batch):
        feature = model(sample)
        print(feature.shape)
        break
