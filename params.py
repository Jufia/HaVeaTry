"""
定义一些变量
"""

import argparse

parser = argparse.ArgumentParser(description='classification task')

# dataset information
parser.add_argument('--datadir', type=str, default='./datasets', help='path to dataset')
parser.add_argument('--load', type=int, default=3, help="working condition")
parser.add_argument("--label_set", type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="label set")
# parser.add_argument("--val_rat", type=float, default=0.3, help="training-validation rate")
# parser.add_argument("--test_rat", type=float, default=0.5, help="validation-test rate")
# parser.add_argument("--seed", type=int, default="29")

# pre-prosessing
parser.add_argument("--length", type=int, default=1024, help="length of sequence")
parser.add_argument("--fft", type=bool, default=False, help="FFT preprocessing")
parser.add_argument("--window", type=int, default=128, help="time window, if not augment data, window=1024")
parser.add_argument("--normalization", type=str, default="0-1", choices=["None", "0-1", "mean-std"], help="normalization option")
# parser.add_argument("--savemodel", type=bool, default=False, help="whether save pre-trained model in the classification task")
# parser.add_argument("--pretrained", type=bool, default=False, help="whether use pre-trained model in transfer learning tasks")

# backbone
parser.add_argument("--backbone", type=str, default="ResNet1D", choices=["ResNet1D", "ResNet2D", "MLPNet", "CNN1D"])

# optimization & training
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--max_epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.003, help="learning rate")

args = parser.parse_args()


