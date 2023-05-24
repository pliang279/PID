import numpy
import sys
import os
import pickle
import torch
sys.path.append(os.getcwd())
from training_structures.unimodal import test
from get_data import get_dataloader
from rus import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

saved_dir = 'MultiBench/synthetic/experiments2/'
results_path = saved_dir + 'results.pickle'
if os.path.isfile(results_path):
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
else:
    results = dict()
results['unimodal'] = dict()
for SETTING in ['redundancy', 'uniqueness0', 'uniqueness1', 'synergy', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6', 'synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5']:
    print(SETTING)
    results['unimodal'][SETTING] = dict()
    for i in range(2):
        print('unimodal', i)
        data_path = saved_dir + 'DATA_{}.pickle'.format(SETTING)
        saved_model = saved_dir + '{}/{}_unimodal{}'.format(SETTING, SETTING, i)
        saved_encoder = saved_model + '_encoder.pt'
        saved_head = saved_model + '_head.pt'
        saved_preds = saved_dir + '{}/{}_unimodal{}_preds.pickle'.format(SETTING, SETTING, i)
        encoder = torch.load(saved_encoder).to(device)
        head = torch.load(saved_head).to(device)
        _, _, _, testdata = get_dataloader(path=data_path, keys=['0','1','label'], modalities=[1,1], batch_size=128, num_workers=4)
        acc = test(encoder, head, testdata, no_robust=True, auprc=False, modalnum=i, save_preds=saved_preds)
        results['unimodal'][SETTING]['acc{}'.format(i)] = acc
    print()

with open(results_path, 'wb') as f:
    pickle.dump(results, f)