import torch
import sys
import os
import pickle
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import MLP, Linear
from get_data import get_dataloader
from supervised_learning import train, test
from fusions.common_fusions import Concat

import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--train-data", nargs='+', default="/usr0/home/yuncheng/MultiBench/synthetic/DATA_mix.pickle", type=str, help="input path of synthetic training datasets")
parser.add_argument("--test-data", default="/usr0/home/yuncheng/MultiBench/synthetic/DATA_mix.pickle", type=str, help="input path of synthetic test datasets")
parser.add_argument("--test-cluster", default="/usr0/home/yuncheng/MultiBench/synthetic/DATA_mix_cluster.pickle", type=str, help="preprocessed cluster version of the test dataset")
parser.add_argument("--keys", nargs='+', default=['0', '1','label'], type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", nargs='+', default=[1,1], type=int, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=128, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", default=200, type=int)
parser.add_argument("--output-dim", default=512, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight-decay", default=0, type=float)
parser.add_argument("--pretrained-model", default=None, type=str)
parser.add_argument("--saved-model", default=None, type=str)
parser.add_argument("--sub-acc", nargs='+', default=None, type=int, help='id of misclassified examples from last cycle for sub-accuracy computation')
args = parser.parse_args()
torch.multiprocessing.set_sharing_strategy('file_system')

# Load data
traindata, validdata1, validdata2, testdata = get_dataloader(path=args.train_data, keys=args.keys, modalities=args.modalities, batch_size=args.bs, num_workers=args.num_workers, test_path=args.test_data)

# Specify late fusion model
out_dim = args.output_dim * len(args.modalities)
encoders = [Linear(args.input_dim, args.output_dim).to(device) for _ in args.modalities]
head = MLP(out_dim, args.hidden_dim, args.num_classes).to(device)
fusion = Concat().cuda()

# Training
train(encoders, fusion, head, traindata, validdata1, args.epochs, optimtype=torch.optim.AdamW, early_stop=False, is_packed=False, lr=args.lr, save=args.saved_model, weight_decay=args.weight_decay, objective=torch.nn.CrossEntropyLoss(), pretrain=args.pretrained_model)

# Testing
print("Testing:")
model = torch.load(args.saved_model).to(device)
with open(args.test_cluster, 'rb') as f:
    cluster = pickle.load(f)
test(model, testdata, no_robust=True, criterion=torch.nn.CrossEntropyLoss(), sub_acc=args.sub_acc)
test(model, validdata2, no_robust=True, criterion=torch.nn.CrossEntropyLoss(), cluster=cluster['valid2'])

