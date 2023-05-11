import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.Supervised_Learning import train, test
from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader as affect_get_dataloader
from datasets.avmnist.get_data import get_dataloader as avmnist_get_dataloader
from fusions.common_fusions import Concat
import numpy as np
import torch

import argparse
import json


def _affect_train_process(encoders, fusion, head, traindata, validdata, saved_model, modalities):
    return train(encoders, fusion, head, traindata, validdata, 50, task="regression", optimtype=torch.optim.AdamW, early_stop=False, is_packed=True, lr=1e-3, save=saved_model, weight_decay=0.01, objective=torch.nn.L1Loss(), modalities=modalities)

def _affect_test_process(model, testdata, modalities):
    return test(model=model, test_dataloaders_all=testdata, dataset=args.dataset, is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True, modalities=modalities)

def _avmnist_train_process(encoders, fusion, head, traindata, validdata, saved_model, modalities):
    return train(encoders, fusion, head, traindata, validdata, 50, optimtype=torch.torch.optim.AdamW, is_packed=True, lr=1e-3, save=saved_model, weight_decay=0.01, modalities=modalities)

def _avmnist_test_process(model, testdata, modalities):
    return test(model, testdata, is_packed=True, no_robust=True, modalities=modalities)


###############################################################
###############################################################


parser = argparse.ArgumentParser()
parser.add_argument("--modalities", default='[0,1,2]', type=str)
parser.add_argument("--dataset", default='mosi', type=str)
parser.add_argument("--dataset-path", default='/usr0/home/yuncheng/MultiBench/data/mosi_raw.pkl', type=str)
args = parser.parse_args()

affect = ['mosi', 'mosei', 'sarcasm', 'humor']
if args.dataset in affect:
    traindata, validdata, testdata = affect_get_dataloader(args.dataset_path, robust_test=False, data_type=args.dataset)
    # mosi/mosei
    if args.dataset == 'mosi':
        d_v = (35, 70)
        d_a = (74, 200)
        d_l = (300, 600)
    elif args.dataset == 'mosei':
        d_v = (713, 70)
        d_a = (74, 200)
        d_l = (300, 600)
    # humor/sarcasm
    elif args.dataset == 'humor' or args.dataset == 'sarcasm':
        d_v = (371, 70)
        d_a = (81, 200)
        d_l = (300, 600)
    train_process = _affect_train_process
    test_process = _affect_test_process
    num_class = 1
    config = [d_v, d_a, d_l]
elif args.dataset == 'avmnist':
    traindata, validdata, testdata = avmnist_get_dataloader(args.dataset_path, unsqueeze_channel=False)
    d_v = (28, 70)
    d_a = (112, 200)
    train_process = _avmnist_train_process
    test_process = _avmnist_test_process
    num_class = 10
    config = [d_v, d_a]

modalities = json.loads(args.modalities)
d_modalities = [config[i] for i in modalities]
out_dim = sum([d[1] for d in d_modalities])
encoders = [GRU(d[0], d[1], dropout=True, has_padding=True, batch_first=True).cuda() for d in d_modalities]
head = MLP(out_dim, 512, num_class).cuda()

fusion = Concat().cuda()

traintimes = []
mems = []
testtimes = []
accs = []
saved_model = '{}_models/{}_lf_{}.pt'.format(args.dataset, args.dataset, ''.join(list(map(str, modalities))))
for _ in range(3):
    traintime, mem, num_params = train_process(encoders, fusion, head, traindata, validdata, saved_model, modalities)
    traintimes.append(traintime)
    mems.append(mem)

    print("Testing:")
    model = torch.load(saved_model).cuda()

    testtime, acc = test_process(model, testdata, modalities)
    testtimes.append(testtime)
    accs.append(acc)

print("Average Training Time: {}, std: {}".format(np.mean(traintimes), np.std(traintimes)))
print("Average Training Memory: {}, std: {}".format(np.mean(mems), np.std(mems)))
print("Number of Parameters: {}".format(num_params))
print("Average Inference Time: {}, std: {}".format(np.mean(testtimes), np.std(testtimes)))
print("Average Performance: {}, std: {}".format(np.mean(accs), np.std(accs)))
