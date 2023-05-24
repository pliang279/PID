import torch
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from utils.helper_modules import Sequential2 # noqa
from unimodals.common_models import MLP, Linear
from get_data import get_dataloader
from supervised_learning import train, test
from fusions.common_fusions import Concat

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", nargs='+', default=['a','b','c','d','e','label'], type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", nargs='+', default=30, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--n-latent", default=512, type=int)
parser.add_argument("--rank", default=32, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight-decay", default=0.01, type=float)
parser.add_argument("--weight", default=1, type=float)
parser.add_argument("--saved-model", default=None, type=str)
args = parser.parse_args()


def EncourageAlignment(ce_weight=2, align_weight=args.weight, criterion=torch.nn.CrossEntropyLoss()):
    def _actualfunc(pred, truth, args):
        ce_loss = criterion(pred, truth)
        outs = args['reps']
        outs[0] = outs[0].view(-1,).cpu().detach().numpy()
        outs[1] = outs[1].view(-1,).cpu().detach().numpy()
        align_loss = np.dot(outs[0], outs[1])/(np.linalg.norm(outs[0])*np.linalg.norm(outs[1]))
        return align_loss * align_weight + ce_loss * ce_weight
    return _actualfunc


# Load data
traindata, validdata, _, testdata = get_dataloader(path=args.data_path, keys=args.keys, modalities=args.modalities, batch_size=args.bs, num_workers=args.num_workers)

# Specify model
if len(args.input_dim) == 1:
    input_dims = args.input_dim * len(args.modalities)
else:
    input_dims = args.input_dim
encoders = [Linear(input_dim, args.hidden_dim).to(device) for input_dim in input_dims]
fusion = Sequential2(Concat(), MLP(len(args.modalities)*args.hidden_dim, args.n_latent, args.n_latent)).to(device)
head = MLP(args.n_latent, args.hidden_dim, args.num_classes).to(device)

# Training
train(encoders, fusion, head, traindata, validdata, args.epochs, objective=torch.nn.CrossEntropyLoss(), optimtype=torch.optim.AdamW, is_packed=False, lr=args.lr, save=args.saved_model, weight_decay=args.weight_decay)

# Testing
print("Testing:")
model = torch.load(args.saved_model).to(device)
test(model, testdata, is_packed=False, no_robust=True, criterion=torch.nn.CrossEntropyLoss())
