import torch
import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import MLP # noqa
from get_data import get_dataloader # noqa
from training_structures.unimodal import train, test # noqa

import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", nargs='+', default=['a','b','c','d','e','label'], type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modality", default=0, type=int)
parser.add_argument("--bs", default=256, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", default=30, type=int)
parser.add_argument("--output-dim", default=128, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight-decay", default=0, type=float)
parser.add_argument("--eval", default=True, type=int)
parser.add_argument("--saved-model", default=None, type=str)
args = parser.parse_args()

# Load data
traindata, validdata, _, testdata = get_dataloader(path=args.data_path, keys=args.keys, modalities=[1,1], batch_size=args.bs, num_workers=args.num_workers)

# Specify unimodal model
encoder = MLP(args.input_dim, args.hidden_dim, args.output_dim).to(device)
head = MLP(args.output_dim, args.hidden_dim, args.num_classes, dropout=False).to(device)

# train
saved_encoder = args.saved_model + '_encoder.pt'
saved_head = args.saved_model + '_head.pt'
train(encoder, head, traindata, validdata, 20, auprc=False, modalnum=args.modality, save_encoder=saved_encoder, save_head=saved_head)

# test
print("Testing: ")
encoder = torch.load(saved_encoder).to(device)
head = torch.load(saved_head).to(device)
test(encoder, head, testdata, no_robust=True, auprc=False, modalnum=args.modality)
