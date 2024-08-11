import torch.nn as nn
import numpy as np
import pickle


class CNN1D(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # (batch, 1, 1024)
            nn.BatchNorm1d(32),
            nn.ReLU(),  # (bs, 32, 1024)
            nn.MaxPool1d(2)  # (bs 32, 512)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),    # (bs, 64, 512)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)  # (bs, 64, 216)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)  # (bs, 64, 128)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # output (bs, 64, 1)
        self.classify = nn.Sequential(nn.Linear(64, num_classes), nn.Dropout(p=0.5))

    def forward(self, x):  # x:(batch, seq_long)
        # input size 2D (unbatched) (channel, seq_long) or 3D (batched) (batch_size, channel, seq_long)
        x = x[:, np.newaxis]
        x = self.layer1(x)  # (bs 32, 512)
        x = self.layer2(x)  # (bs, 64, 216)
        x = self.layer3(x)  # (bs, 64, 128)
        x = self.avgpool(x)  # output (bs, 64, 1) !! 觉得这里不合理，128个数直接取平均池化
        y = x.view(x.size(0), -1)  # (batch, 64)
        y = self.classify(y)
        return y


if __name__ == '__main__':
    train_batch = pickle.load(open('../Dataset/train_batch.dill', 'rb'))
    model = CNN1D()
    for step, (sample, label) in enumerate(train_batch):
        feature = model(sample)
        print(feature.shape)
        break
