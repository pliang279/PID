from ce_alignment_information import critic_ce_alignment
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

x1 = np.array([[0], [0], [1], [1]])
x2 = np.array([[0], [1], [0], [1]])
data = {
    'and': (x1, x2, 1 * np.logical_and(x1, x2)),
    'or': (x1, x2, 1 * np.logical_or(x1, x2)),
    'xor': (x1, x2, 1 * np.logical_xor(x1, x2)),
    'unique1': (x1, x2, x1),
    'redundant': (x1, x1, x1),
    'redundant_and_unique1': (np.concatenate([x1, x2], axis=1), x2, 1 * np.logical_and(x1, x2)),
    'redundant_or_unique1': (np.concatenate([x1, x2], axis=1), x2, 1 * np.logical_or(x1, x2)),
    'redundant_xor_unique1': (np.concatenate([x1, x2], axis=1), x2, 1 * np.logical_xor(x1, x2)),
}

class MultimodalDataset(Dataset):
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels
    self.num_modalities = len(self.data)
  
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return tuple([self.data[i][idx] for i in range(self.num_modalities)] + [self.labels[idx]])

batch_size = 256

def prob_set(x):
    ret = {}
    for y in x:
        if y not in ret:
            ret[y] = 0
        ret[y] += 1
    ret = {k: (v / len(x)) for k, v in ret.items()}
    return ret

# 1. original: 1004.2515

def get_min_mi(x1, x2, y):
    p1 = prob_set(x1)
    p1y = prob_set(list(zip(x1, y)))
    p2 = prob_set(x2)
    p2y = prob_set(list(zip(x2, y)))
    py = prob_set(y)
    p12 = prob_set(list(zip(x1, x2)))
    p12y = prob_set(list(zip(x1, x2, y)))

    value_given_y = {}
    x1_, x2_, y_ = x1, x2, y
    for x1, x2, y in zip(x1, x2, y):
        if y not in value_given_y:
            value_given_y[y] = []
        value1 = np.log2(p1y[x1, y] / (p1[x1] * py[y]))
        value2 = np.log2(p2y[x2, y] / (p2[x2] * py[y]))
        value_given_y[y].append([value1, value2])
    result = 0.0
    for y in value_given_y:
        result += (len(value_given_y[y]) / len(y_)) * \
            np.min(np.mean(np.array(value_given_y[y]), axis=0))
    return result

# over q

def get_mi_terms(x1, x2, y):
    p1 = prob_set(x1)
    p1y = prob_set(list(zip(x1, y)))
    p2 = prob_set(x2)
    p2y = prob_set(list(zip(x2, y)))
    py = prob_set(y)
    p12 = prob_set(list(zip(x1, x2)))
    p12y = prob_set(list(zip(x1, x2, y)))
    sum = 0

    x1_, x2_, y_ = x1, x2, y
    for x1, x2, y in zip(x1, x2, y):
        value = np.log2(p1y[x1, y] / (p1[x1] * py[y])) + \
            np.log2(p2y[x2, y] / (p2[x2] * py[y])) - \
            np.log2(p12y[x1, x2, y] / (p12[x1, x2] * py[y]))
        sum += value
    return sum / len(y_)

from itertools import permutations

def enumerate_alignments(x1, x2, y):
    x10 = [x1[i] for i in range(len(x1)) if not y[i]]
    x11 = [x1[i] for i in range(len(x1)) if y[i]]
    x20 = [x2[i] for i in range(len(x2)) if not y[i]]
    x21 = [x2[i] for i in range(len(x2)) if y[i]]

    perm0, perm1 = list(permutations(range(len(x10)))), list(permutations(range(len(x11))))

    max_mi, max_p0, max_p1 = -10000000, None, None
    for p0 in perm0:
        for p1 in perm1:
            x10p = x10
            x11p = x11
            x20p = [x20[i] for i in p0]
            x21p = [x21[i] for i in p1]
            yp = [0] * len(x10p) + [1] * len(x11p)
            mi = get_mi_terms(x10p + x11p, x20p + x21p, yp)
            # print(f'MI {mi} p0 {p0} p1 {p1}')
            if mi > max_mi:
                max_mi, max_p0, max_p1 = mi, p0, p1
    return max_mi, max_p0, max_p1

# fixed q'

def q_x1_x2(x1, x2, p1y, p2y, py):
    total = 0.
    for y in py.keys():
        total += py[y] * p1y.get((x1, y), 0) / py[y] * p2y.get((x2, y), 0) / py[y]
    return total

def q_x1(x1, p1y, p2y, py):
    total = 0.
    for x2, y in p2y.keys():
        total += py[y] * p1y.get((x1, y), 0) / py[y] * p2y.get((x2, y), 0) / py[y]
    return total

def q_x2(x2, p1y, p2y, py):
    total = 0.
    for x1, y in p1y.keys():
        total += py[y] * p1y.get((x1, y), 0) / py[y] * p2y.get((x2, y), 0) / py[y]
    return total

def get_analytic_upperbound(x1_, x2_, y_):
    p1 = prob_set(x1_)
    p2 = prob_set(x2_)
    p12 = prob_set(list(zip(x1_, x2_)))
    py = prob_set(y_)
    p1y = prob_set(list(zip(x1_, y_)))
    p2y = prob_set(list(zip(x2_, y_)))

    total = 0
    for y in py.keys():
        for x1 in p1.keys():
            for x2 in p2.keys():
                weight = py[y] * (p1y.get((x1, y), 0) / py[y]) * (p2y.get((x2, y), 0) / py[y])
                if weight == 0:
                    continue
                value = np.log2(q_x1_x2(x1, x2, p1y, p2y, py) / \
                                (q_x1(x1, p1y, p2y, py) * \
                                 q_x2(x2, p1y, p2y, py)))
                total += weight * value
    return total

def get_mi_1_1(x1, x2):
    p1 = prob_set(x1)
    p2 = prob_set(x2)
    p12 = prob_set(list(zip(x1, x2)))

    x1_, x2_ = x1, x2
    sum = 0
    for x1, x2 in zip(x1, x2):
        value = np.log2(p12[x1, x2] / (p1[x1] * p2[x2]))
        sum += value
    return sum / len(x1_)

def get_mi_2_1(x1, x2, y):
    p12 = prob_set(list(zip(x1, x2)))
    p12y = prob_set(list(zip(x1, x2, y)))
    py = prob_set(y)

    x1_, x2_, y_ = x1, x2, y
    sum = 0
    for x1, x2, y in zip(x1, x2, y):
        value = np.log2(p12y[x1, x2, y] / (p12[x1, x2] * py[y]))
        sum += value
    return sum / len(y_)

def get_cmi_1_1_1(x1, x2, y):
    p12y = prob_set(list(zip(x1, x2, y)))
    py = prob_set(y)
    p1y = prob_set(list(zip(x1, y)))
    p2y = prob_set(list(zip(x2, y)))

    x1_, x2_, y_ = x1, x2, y
    sum = 0
    for x1, x2, y in zip(x1, x2, y):
        value = np.log2(p12y[x1, x2, y] * py[y] / (p1y[x1, y] * p2y[x2, y]))
        sum += value
    return sum / len(y_)

def get_ii(x1, x2, y):
    return get_mi_1_1(x1, x2) - get_cmi_1_1_1(x1, x2, y)

def get_u1(r, x1, y):
    return get_mi_1_1(x1, y) - r

def get_u2(r, x2, y):
    return get_mi_1_1(x2, y) - r

def get_s_v1(r, x1, x2, y):
    return get_mi_2_1(x1, x2, y) - r - get_u1(r, x1, y) - get_u2(r, x2, y)

def get_s_v2(r, x1, x2, y):
    return r - get_ii(x1, x2, y)

def get_all(f_r, train_ds):
    x1 = [tuple(x[0].tolist()) for x in train_ds]
    x2 = [tuple(x[1].tolist()) for x in train_ds]
    y = [x[2].item() for x in train_ds]
    r = f_r(x1, x2, y)
    u1 = get_u1(r, x1, y)
    u2 = get_u2(r, x2, y)
    s_v1 = get_s_v1(r, x1, x2, y)
    s_v2 = get_s_v2(r, x1, x2, y)
    return r, u1, u2, s_v1, s_v2

# Examples:

def example_data_norepeat(name):
    x1, x2, y = data[name]
    x1 = torch.tensor(x1)
    x2 = torch.tensor(x2)
    y = torch.tensor(y)

    ds = MultimodalDataset([x1.float(), x2.float()], y)
    torch.manual_seed(42)
    train_ds, test_ds = ds, ds

    print(get_all(lambda x1, x2, y: enumerate_alignments(x1, x2, y)[0], train_ds))
    return x1, x2, y, train_ds, test_ds

def example_data(name):
    print(name)
    x1, x2, y = data[name]
    x1 = torch.tensor(x1)[None].expand(1024, -1, -1).flatten(0, 1)
    x2 = torch.tensor(x2)[None].expand(1024, -1, -1).flatten(0, 1)
    y = torch.tensor(y)[None].expand(1024, -1, -1).flatten(0, 1)

    ds = MultimodalDataset([x1.float(), x2.float()], y)
    torch.manual_seed(42)
    train_ds, test_ds = ds, ds

    print(get_all(get_min_mi, train_ds))
    example_data_norepeat(name)
    print(get_all(get_analytic_upperbound, train_ds))
    return x1, x2, y, train_ds, test_ds

# Run on Generative Synthetic Datasets:

import pickle
with open('DATA_mix1.pickle', 'rb') as f:
    data = pickle.load(f)

x1 = torch.tensor(data['train']['0'], dtype=torch.float)
x2 = torch.tensor(data['train']['1'], dtype=torch.float)
y = torch.tensor(data['train']['label'])

x1_test = torch.tensor(data['test']['0'], dtype=torch.float)
x2_test = torch.tensor(data['test']['1'], dtype=torch.float)
y_test = torch.tensor(data['test']['label'])

train_ds = MultimodalDataset([x1, x2], y)
test_ds = MultimodalDataset([x1_test, x2_test], y_test)
results, aligns, models = critic_ce_alignment(x1, x2, y, 2, train_ds, test_ds, discrim_epochs=1, ce_epochs=10)

results_agg = torch.mean(results, dim=0) / torch.log(torch.tensor(2, device='cuda'))
print(', '.join(['%s: %.2f' % x for x in zip(['redundancy', 'unique 1', 'unique 2', 'synergy'], results_agg.tolist())]))

# Run on Binary Synthetic Datasets:

for name in data:
    x1, x2, y, train_ds, test_ds = example_data(name)
    results, aligns, models = critic_ce_alignment(x1, x2, y, 2, train_ds, test_ds)

    models[0].eval()
    _, results, aligns = models[0](x1[:4].cuda().float(), x2[:4].cuda().float(), y[:4].cuda())
    results = results / torch.log(torch.tensor(2, device='cuda'))
    print(', '.join(['%s: %.2f' % x for x in zip(['redundancy', 'unique 1', 'unique 2', 'synergy'], results.tolist())]))
    print()
