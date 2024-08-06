import torch
import pandas
import numpy as np
from scipy.io import loadmat


def Data_read(path: str, loca: str, label: int):
    data = loadmat(path)  # (244739, 1)
    for k in data.keys():
        if loca in k:
            data = data[k]  # shape:(244739, 1)
            break

    Window = 128
    EndPoint = data.shape[0]
    length = 1024

    sub_data = data[0: length]
    for start in range(Window, EndPoint - length, Window):
        ss = data[start: start + length]  # (1024, 1)
        sub_data = np.append(sub_data, ss, axis=1)

    samples = sub_data.transpose()  # (n, 1024)
    samples_num = len(samples)
    labels = (np.ones(samples_num) * label).reshape(-1, 1)  # (n, 1)

    return samples, labels


if __name__ == "__main__":
    path = '../datasets/CWRU/Drive_end_3/241.mat'
    loca = '_DE_time'
    label = 0
    samples, labels = Data_read(path, loca, label)

    np.savetxt('../Dataset/data_samples.csv', samples, delimiter=',')
    np.savetxt('../Dataset/data_labels.csv', labels, delimiter=',')
