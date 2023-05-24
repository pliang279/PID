import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import Linear, MLP  # noqa
from ensemble import train, test  # noqa
from get_data import get_dataloader  # noqa
from fusions.ensemble_fusions import AdditiveEnsemble  # noqa

import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", nargs='+', default=['a','b','c','d','e','label'], type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=128, type=int)
parser.add_argument("--input-dim", nargs='+', default=30, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--output-dim", default=512, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight-decay", default=0, type=float)
parser.add_argument("--eval", default=True, type=int)
parser.add_argument("--setting", default='redundancy', type=str)
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
heads = [MLP(args.output_dim, args.hidden_dim, args.num_classes, dropout=False).to(device) for _ in args.modalities]
# encoders = [torch.load(f'synthetic/model_selection/{args.setting}/{args.setting}_unimodal{i}_encoder.pt') for i in args.modalities]
# heads = [torch.load(f'synthetic/model_selection/{args.setting}/{args.setting}_unimodal{i}_head.pt') for i in args.modalities]
ensemble = AdditiveEnsemble().to(device)

# Training
train(encoders, heads, ensemble, traindata, validdata, args.epochs, optimtype=torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay, criterion=torch.nn.CrossEntropyLoss(), save_model=args.saved_model, modalities=args.modalities)

# Testing
print("Testing:", args.saved_model)
model = torch.load(args.saved_model).to(device)
save_acc = ['synthetic/experiments2/results.pickle', args.setting]
saved_dir = 'synthetic/experiments2/'
saved_cluster = saved_dir + '{}/{}_additive_cluster.pickle'.format(args.setting, args.setting)
test(model, testdata, no_robust=True, criterion=torch.nn.CrossEntropyLoss(), save_acc=save_acc, save_preds=saved_cluster)
