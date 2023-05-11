import torch
import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import GRU, MLP
from training_structures.Supervised_Learning import train, test  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import MultiplicativeInteractions3Modal


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_senti_data.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, testdata = get_dataloader('/content/CoLearningTestBed/data/mosi_raw.pkl', robust_test=False, data_type='mosi')

encoders = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).cuda(),
            GRU(74, 100, dropout=True, has_padding=True, batch_first=True).cuda(),
            GRU(300, 400, dropout=True, has_padding=True, batch_first=True).cuda()]
head = MLP(100, 100, 1).cuda()
fusion = MultiplicativeInteractions3Modal([70, 100, 400], 100, task='affect').cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save='mosi_tensor_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_tensor_best.pt').cuda()

test(model=model, test_dataloaders_all=testdata, dataset='mosi', is_packed=True,
     criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)
