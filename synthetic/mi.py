import torch
import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import MLP, Linear
from get_data import get_dataloader
from supervised_learning import train, test
from fusions.common_fusions import MultiplicativeInteractions2Modal

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="/home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", nargs='+', default=['a','b','c','d','e','label'], type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=16, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", nargs='+', default=30, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--output-dim", default=512, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight-decay", default=0.01, type=float)
parser.add_argument("--saved-model", default=None, type=str)
args = parser.parse_args()

# Load data
traindata, validdata, _, testdata = get_dataloader(path=args.data_path, keys=args.keys, modalities=args.modalities, batch_size=args.bs, num_workers=args.num_workers)

# Specify model
if len(args.input_dim) == 1:
    input_dims = args.input_dim * len(args.modalities)
else:
    input_dims = args.input_dim
encoders = [Linear(input_dim, args.output_dim).to(device) for input_dim in input_dims]
head = MLP(args.output_dim, args.hidden_dim, args.num_classes).to(device)
fusion = MultiplicativeInteractions2Modal([args.output_dim, args.output_dim], args.output_dim, 'matrix', flatten=True).to(device)

train(encoders, fusion, head, traindata, validdata, args.epochs, optimtype=torch.optim.AdamW, early_stop=False, is_packed=False, lr=args.lr, save=args.saved_model, weight_decay=args.weight_decay, objective=torch.nn.CrossEntropyLoss())

# Testing
print("Testing:")
model = torch.load(args.saved_model).to(device)
test(model=model, test_dataloaders_all=testdata, is_packed=False, criterion=torch.nn.CrossEntropyLoss(), no_robust=True)
