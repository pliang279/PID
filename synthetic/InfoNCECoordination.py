import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import MLP, Linear
from get_data import get_dataloader
from supervised_learning import train, test
from fusions.common_fusions import Concat
from cca_zoo.deepmodels import (
    DCCA,
    architectures,
    DCCA_NOI,
    DCCA_SDL,
    BarlowTwins,
)
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import default_collate
from torchinfo import summary
import argparse
import pickle

device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="/home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", nargs='+', default=['a','b','c','d','e','label'], type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", default=30, type=int)
parser.add_argument("--output-dim", default=128, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--rank", default=32, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--weight-decay", default=0.01, type=float)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--saved-model", default='/home/yuncheng/lrf_best.pt', type=str)
args = parser.parse_args()

with open(args.data_path, 'rb') as f:
  dicta = pickle.load(f)


train_tensor_list = [torch.tensor(dicta['train'][key]).float() for key in args.keys if key != 'label']
valid_tensor_list = [torch.tensor(dicta['valid'][key]).float() for key in args.keys if key != 'label']

train_dataset = torch.utils.data.TensorDataset(*train_tensor_list)
val_dataset = torch.utils.data.TensorDataset(*valid_tensor_list)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, collate_fn= lambda x: {"views": default_collate(x)})
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, collate_fn= lambda x: {"views": default_collate(x)})

from info_nce import InfoNCE

encoders = nn.ModuleList([MLP(args.input_dim, args.output_dim, args.output_dim).to(device)]*len(args.modalities))
head = MLP(args.output_dim*len(args.modalities), args.hidden_dim, args.num_classes).to(device)
fusion = Concat().cuda()
criterion = InfoNCE()
optimizer = torch.optim.Adam(encoders.parameters())
from tqdm import tqdm
for idx in tqdm(range(args.epochs)):
  losses = 0
  for datum in train_dataloader:
    datum = datum['views']
    datum = [data.to(device) for data in datum]
    encoded = [encoders[idx](elem) for idx, elem in enumerate(datum)]
    loss = 0.0
    count = 0.0
    for idx, a in enumerate(encoded):
      for id2, b in enumerate(encoded):
        if idx <= id2: 
          continue
        loss += criterion(a,b)
        count += 1.0
    loss = loss / count
    optimizer.zero_grad()
    loss.backward()
    losses += loss.item()
    optimizer.step()
  print(losses/len(train_dataloader))

for encoder in encoders:
  for param in encoder.parameters():
    param.requires_grad = False


encoders = [encoder for _ in range(len(args.modalities))]
# Load data
traindata, validdata, testdata = get_dataloader(path=args.data_path, keys=args.keys, modalities=args.modalities, batch_size=args.bs, num_workers=args.num_workers)


train(encoders, fusion, head, traindata, validdata, args.epochs, optimtype=torch.optim.AdamW, early_stop=False, is_packed=False, lr=args.lr, save=args.saved_model, weight_decay=args.weight_decay, objective=torch.nn.CrossEntropyLoss())

# Testing
print("Testing:")
model = torch.load(args.saved_model).to(device)
test(model, testdata, is_packed=False, no_robust=True, criterion=torch.nn.CrossEntropyLoss())
