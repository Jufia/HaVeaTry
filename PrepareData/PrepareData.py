import numpy as np
from scipy.io import loadmat
from params import args
import os

def Normalize(sub_data, normalization):
    if normalization == "0-1":
        sub_data = (sub_data - sub_data.min()) / (sub_data.max() - sub_data.min())
    elif normalization == "mean-std":
        sub_data = (sub_data - sub_data.mean()) / sub_data.std()

    return sub_data

def Data_read(path: str, loca: str, label: int):
    data = loadmat(path)  # (244739, 1)
    for k in data.keys():
        if loca in k:
            data = data[k]  # shape:(244739, 1)
            break

    Window = 128
    EndPoint = data.shape[0]
    length = args.length

    sub_data = data[0: length]
    for start in range(Window, EndPoint - length, Window):
        ss = data[start: start + length]  # (1024, 1)
        ss = Normalize(ss, args.normalization)
        sub_data = np.append(sub_data, ss, axis=1)

    samples = sub_data.transpose()  # (n, 1024)
    samples_num = len(samples)
    labels = (np.ones(samples_num) * label).reshape(-1, 1)  # (n, 1)

    return samples, labels


def Data_Merge_Save(loca = '_DE_time'):
    paths = ['../datasets/CWRU/Drive_end_0/',
             '../datasets/CWRU/Drive_end_1/',
             '../datasets/CWRU/Drive_end_2/',
             '../datasets/CWRU/Drive_end_3/']
    dataname_dict = {0: [97, 109, 122, 135, 173, 189, 201, 213, 226, 238],  # 1797rpm
                     1: [98, 110, 123, 136, 175, 190, 202, 214, 227, 239],  # 1772rpm
                     2: [99, 111, 124, 137, 176, 191, 203, 215, 228, 240],  # 1750rpm
                     3: [100, 112, 125, 138, 177, 192, 204, 217, 229, 241]}
    label_set = args.label_set
    samples = np.empty((0, args.length))
    labels = np.empty((0, 1))
    for i, filename in enumerate(dataname_dict[args.load]):
        path = paths[args.load] + str(filename) + '.mat'
        label = label_set[i]
        sub_sample, sub_label = Data_read(path, loca, label)
        samples = np.append(samples, sub_sample, axis=0)
        labels = np.append(labels, sub_label, axis=0)

    # os.makedirs("../Dataset/")
    np.save('../Dataset/data_samples.npy', samples)
    np.save('../Dataset/data_labels.npy', labels)
    # np.savetxt('../Dataset/data_samples.csv', samples, delimiter=',')
    # np.savetxt('../Dataset/data_labels.csv', labels, delimiter=',')

if __name__ == "__main__":
    Data_Merge_Save()
