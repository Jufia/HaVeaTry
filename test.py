from PrepareData import PrepareData
import numpy as np
import torch
import params

args = params.parse_args()

label_set = args.label_set
paths = ['datasets/CWRU/Drive_end_0/',
         'datasets/CWRU/Drive_end_1/',
         'datasets/CWRU/Drive_end_2/',
         'datasets/CWRU/Drive_end_3/']

dataname_dict = {0: [97, 109, 122, 135, 173, 189, 201, 213, 226, 238],  # 1797rpm
                 1: [98, 110, 123, 136, 175, 190, 202, 214, 227, 239],  # 1772rpm
                 2: [99, 111, 124, 137, 176, 191, 203, 215, 228, 240],  # 1750rpm
                 3: [100, 112, 125, 138, 177, 192, 204, 217, 229, 241]}
path = paths[args.load]
data_name = dataname_dict[args.load]
loca = '_DE_time'

samples = np.empty((0, 1024))
labels = np.empty((0, 1))
for i, filename in enumerate(dataname_dict[args.load]):
    path = paths[args.load] + str(filename) + '.mat'
    label = label_set[i]
    sub_sample, sub_label = PrepareData.Data_read(path, loca, label)
    samples = np.append(samples, sub_sample, axis=0)
    labels = np.append(labels, sub_label, axis=0)

np.savetxt('Dataset/data_samples.csv', samples, delimiter=',')
np.savetxt('Dataset/data_labels.csv', labels, delimiter=',')





