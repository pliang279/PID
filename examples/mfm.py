import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from unimodals.MVAE import TSEncoder, TSDecoder # noqa
from utils.helper_modules import Sequential2 # noqa
from objective_functions.objectives_for_supervised_learning import MFM_objective # noqa
from torch import nn # noqa
from unimodals.common_models import MLP # noqa
from training_structures.Supervised_Learning import train, test # noqa
from datasets.affect.get_data import get_dataloader as affect_get_dataloader
from datasets.avmnist.get_data import get_dataloader as avmnist_get_dataloader
from fusions.common_fusions import Concat # noqa

import numpy as np
import argparse
import json


def _affect_train_process(encoders, fuse, head, traindata, validdata, additional_modules, saved_model, modalities):
    return train(encoders, fuse, head, traindata, validdata, 50, additional_modules, objective=objective, objective_args_dict=argsdict, save=saved_model, modalities=modalities)

def _affect_test_process(model, testdata, modalities):
    return test(model=model, test_dataloaders_all=testdata, dataset=args.dataset, is_packed=False, no_robust=True, modalities=modalities)

def _avmnist_train_process(encoders, fuse, head, traindata, validdata, additional_modules, objective, argsdict, saved_model, modalities):
    return train(encoders, fuse, head, traindata, validdata, 50, additional_modules, objective=objective, objective_args_dict=argsdict, save=saved_model, modalities=modalities)

def _avmnist_test_process(model, testdata, modalities):
    return test(model=model, test_dataloaders_all=testdata, dataset=args.dataset, is_packed=False, no_robust=True, modalities=modalities)


###############################################################
###############################################################


parser = argparse.ArgumentParser()
parser.add_argument("--modalities", default='[0,1,2]', type=str)
parser.add_argument("--dataset", default='mosi', type=str)
parser.add_argument("--dataset-path", default='/usr0/home/yuncheng/MultiBench/data/mosi_raw.pkl', type=str)
args = parser.parse_args()


affect = ['mosi', 'mosei', 'sarcasm', 'humor']
if args.dataset in affect:
    classes = 2
    n_latent = 256
    timestep = 50
    traindata, validdata, test_robust = get_dataloader(args.dataset_path, task='classification', robust_test=False, max_pad=True, max_seq_len=timestep, data_type=args.dataset)
    # mosi/mosei
    if args.dataset == 'mosi':
        dim_0 = 35
        dim_1 = 74
        dim_2 = 300
    elif args.dataset == 'mosei':
        dim_0 = 713
        dim_1 = 74
        dim_2 = 300
    # humor/sarcasm
    elif args.dataset == 'humor' or args.dataset == 'sarcasm':
        dim_0 = 371
        dim_1 = 81
        dim_2 = 300
    train_process = _affect_train_process
    test_process = _affect_test_process
    config = [dim_0, dim_1, dim_2]
elif args.dataset == 'avmnist':
    classes = 10
    n_latent = 256
    timestep = 50
    traindata, validdata, testdata = avmnist_get_dataloader(args.dataset_path, unsqueeze_channel=False, max_pad=True, max_seq_len=timestep)
    dim_0 = 28
    dim_1 = 112
    train_process = _avmnist_train_process
    test_process = _avmnist_test_process
    config = [dim_0, dim_1]

modalities = json.loads(args.modalities)
d_modalities = [config[i] for i in modalities]
encoders = [TSEncoder(d, 30, n_latent, timestep, returnvar=False).cuda() for d in d_modalities]
decoders = [TSDecoder(d, 30, n_latent, timestep).cuda() for d in d_modalities]
fuse = Sequential2(Concat(), MLP(len(modalities)*n_latent, n_latent, n_latent//2)).cuda()
intermediates = [MLP(n_latent, n_latent//2, n_latent//2).cuda() for _ in modalities]
head = MLP(n_latent//2, 20, classes).cuda()

argsdict = {'decoders': decoders, 'intermediates': intermediates}

additional_modules = decoders+intermediates

objective = MFM_objective(2.0, [torch.nn.MSELoss(
), torch.nn.MSELoss(), torch.nn.MSELoss()], [1.0, 1.0, 1.0])

traintimes = []
mems = []
testtimes = []
accs = []
saved_model = '{}_models/{}_mfm_{}.pt'.format(args.dataset, args.dataset, ''.join(list(map(str, modalities))))
for _ in range(3):
    traintime, mem, num_params = train_process(encoders, fuse, head, traindata, validdata, additional_modules, objective, argsdict, saved_model, modalities)

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
