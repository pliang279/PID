import os
import sys
sys.path.append(os.getcwd())
import math
import numpy as np
import torch
from torch import nn
import pickle
from itertools import chain, combinations
from collections import namedtuple
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-data", default=10000, type=int)
parser.add_argument("--modality-number", default=2, type=int)
parser.add_argument("--feature-dim", default=5, type=int)
parser.add_argument("--feature-sep", default=1.5, type=float)
parser.add_argument("--label-dim", nargs='+', default=[5], type=int)
parser.add_argument("--transform-dim", default=32, type=int)
parser.add_argument('--setting', default='redundancy', type=str)
parser.add_argument('--mix-ratio', nargs='+', default=None, type=float)
parser.add_argument('--num-classes', default=32, type=int)
parser.add_argument('--out-path', default='MultiBench/synthetic', type=str)
args = parser.parse_args()


def save_data(data, filename):
    with open(os.path.join(args.out_path, filename), 'wb') as f:
        pickle.dump(data, f)


num_data = args.num_data
n_modality = args.modality_number
num_classes = args.num_classes
mix_ratio = args.mix_ratio if args.mix_ratio else np.random.rand(3,)
dim_info = {'redundancy':[0, 0, 1], 'uniqueness0':[1, 0, 0], 'uniqueness1':[0, 1, 0], 'synergy':[1, 1, 0]}

intersections = chain.from_iterable(combinations(np.arange(n_modality), r) for r in range(1, n_modality+1))
intersections = [''.join([str(i) for i in x]) for x in intersections]
feature_dim = [args.feature_dim] * len(intersections)
feature_sep = args.feature_sep
DimInfo = namedtuple('DimInfo', ['dim', 'sep'])
if args.setting in dim_info:
    label_dim = np.array(dim_info[args.setting]) * np.array(feature_dim)
    feature_dim_info = dict()
    label_dim_info = dict()
    for (i, x) in enumerate(intersections):
        feature_dim_info[x] = DimInfo(feature_dim[i], feature_sep)
        label_dim_info[x] = label_dim[i]
else:
    assert(len(mix_ratio) == len(intersections))
    mix_ratio = np.array(mix_ratio)
    feature_dim_info = dict()
    for (i, x) in enumerate(intersections):
        feature_dim_info[x] = DimInfo(feature_dim[i], feature_sep)

total_data = [[] for _ in range(n_modality)]
total_labels = []

transforms = dict()
for x in intersections:
    transforms[x] = np.random.uniform(0.0,1.0,(feature_dim_info[x].dim, args.transform_dim))
label_transform = nn.Sequential(nn.Dropout(0.2))

dataset = []
for _ in range(num_data):
    raw_features = dict()
    for k, d in feature_dim_info.items():
        raw_features[k] = np.random.multivariate_normal(np.zeros((d.dim,)), np.eye(d.dim)*d.sep, (1,))[0]
    modality_data = []
    for i in range(n_modality):
        modality_data.append([])
        for k, v in raw_features.items():
            if str(i) in str(k):
                modality_data[-1].append(v @ transforms[k])
    modality_data = [np.concatenate(data) for data in modality_data]

    label_components = []
    if args.setting in dim_info:
        for k, d in label_dim_info.items():
            label_components.append(raw_features[k][:d])
        label_vector = np.concatenate(label_components)
        label_vector = label_transform(torch.Tensor(label_vector)).detach().numpy()
        label_prob = 1 / (1 + math.exp(-np.mean(label_vector)))
        label = int(label_prob*32) - int(int(label_prob*32)==32)
        dataset.append((label_prob, modality_data))
    else:
        for (i, k) in enumerate(intersections):
            idx = np.sort(np.random.choice(np.arange(len(raw_features[k])), int(mix_ratio[i]*len(raw_features[k])), replace=False))
            label_components.append(raw_features[k][idx])
        label_vector = np.concatenate(label_components)
        label_vector = label_transform(torch.Tensor(label_vector)).detach().numpy()
        label_prob = 1 / (1 + math.exp(-np.mean(label_vector)))
        label = int(label_prob*32) - int(int(label_prob*32)==32)
        dataset.append((label_prob, modality_data))
dataset = sorted(dataset, key=lambda x: x[0])
for label in range(num_classes):
    tmp = dataset[label*num_data//num_classes:(label+1)*num_data//num_classes]
    for i in range(n_modality):
        for x in tmp:
            total_data[i].append(x[1][i])
    total_labels.append([label] * (num_data//num_classes))
total_labels = np.array(total_labels).reshape(-1,1)
for i in range(n_modality):
    total_data[i] = np.vstack(total_data[i])
    assert(len(total_data[i]) == len(total_labels))

data = dict()
data['train'] = dict()
data['valid'] = dict()
# data['valid2'] = dict()
data['test'] = dict()
X_train, X_test, y_train, y_test = train_test_split(np.array(total_data).transpose((1,0,2)), total_labels, test_size=0.3, stratify=total_labels)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)
# X_valid1, X_valid2, y_valid1, y_valid2 = train_test_split(X_valid, y_valid, test_size=0.5, stratify=y_valid)
for i in range(n_modality):
    data['train'][str(i)] = np.array(X_train)[:,i,:]
    data['valid'][str(i)] = np.array(X_valid)[:,i,:]
    # data['valid2'][str(i)] = np.array(X_valid2)[:,i,:]
    data['test'][str(i)] = np.array(X_test)[:,i,:]
data['train']['label'] = np.array(y_train)
data['valid']['label'] = np.array(y_valid)
# data['valid2']['label'] = np.array(y_valid2)
data['test']['label'] = np.array(y_test)
save_data(data, "DATA_{}.pickle".format(args.setting))
# for k, v in total_raw_features.items():
#     total_raw_features[k] = np.array(v)
# save_data(total_raw_features, "RAW_{}.pickle".format(args.setting))
