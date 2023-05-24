import numpy
import sys
import os
import pickle
import torch
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from supervised_learning import test
from ensemble import test as test_ensemble
from get_data import get_dataloader
from scipy.stats import entropy
from rus import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

saved_dir = 'MultiBench/synthetic/experiments2/'
results_path = saved_dir + 'results.pickle'
if os.path.isfile(results_path):
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
else:
    results = dict()
# for method in ['additive', 'agree', 'align', 'recon', 'early_fusion', 'elem', 'mi', 'outer', 'lower', 'mult']:
for method in ['additive']:
# for method in ['recon']:
    w_method = method
    print(w_method)
    if w_method not in results:
        results[w_method] = dict()
    for setting in ['redundancy', 'uniqueness0', 'uniqueness1', 'synergy', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6', 'synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5']:
    # for setting in ['synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5']:
        print(setting)
        data_path = saved_dir + 'DATA_{}.pickle'.format(setting)
        saved_model = saved_dir + '{}/{}_{}_best.pt'.format(setting, setting, w_method)
        saved_cluster = saved_dir + '{}/{}_{}_cluster.pickle'.format(setting, setting, w_method)
        _, _, _, testdata = get_dataloader(path=data_path, keys=['0','1','label'], modalities=[1,1], batch_size=128, num_workers=4)
        model = torch.load(saved_model).cuda()
        # num_params = sum([p.numel() for p in model.parameters()])
        if method in ['additive']:
            acc = test_ensemble(model, testdata, no_robust=True, criterion=torch.nn.CrossEntropyLoss(), save_preds=saved_cluster)
        else:
            acc = test(model, testdata, no_robust=True, criterion=torch.nn.CrossEntropyLoss(), save_preds=saved_cluster)
        # with open(saved_cluster, 'rb') as f:
        #     preds = pickle.load(f)
        # with open(saved_dir + 'DATA_{}_cluster.pickle'.format(setting), 'rb') as f:
        #     cluster = pickle.load(f)
        # pred_results = (cluster['test']['0'], cluster['test']['1'], preds.reshape(-1,1))
        # P, maps = convert_data_to_distribution(*pred_results)
        # tmp = get_measure(P)
        # if setting not in results[w_method]:
            # results[w_method][setting] = tmp
            # results[w_method][setting] = dict()
        # else:
        # for k in tmp:
        #     results[w_method][setting][k] = tmp[k]
        # results[w_method][setting]['entropy'] = entropy(preds)
        results[w_method][setting]['acc'] = acc
        # results[w_method][setting]['params'] = num_params
        print()

    with open(results_path, 'wb') as f:
        pickle.dump(results, f)