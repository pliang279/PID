"""Implements training pipeline for ensemble with unimodal models."""
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from utils.AUPRC import AUPRC
from eval_scripts.performance import eval_affect
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from tqdm import tqdm
import pickle
softmax = nn.Softmax()


def deal_with_objective(objective, pred, truth, args):
    """Alter inputs depending on objective function, to deal with different objective arguments."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size())-1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().to(device))
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(objective) == nn.L1Loss:
        return objective(pred, truth.float().to(device))
    else:
        return objective(pred, truth, args)


def train(encoders, heads, ensemble, train_dataloader, valid_dataloader, total_epochs, early_stop=False, optimtype=torch.optim.RMSprop, 
            lr=0.001, weight_decay=0.0, criterion=nn.CrossEntropyLoss(), auprc=False, 
            save_model='ensemble.pt', modalities=[0], task='classification', track_complexity=True, objective_args_dict={}):
    """Train ensemble module.

    Args:
        encoders (nn.Module): Unimodal encodders for each modality
        heads (nn.Module): Takes in the unimodal encoder output and produces the final prediction.
        ensemble (nn.Module): Module to combine outputs from unimodal models.
        train_dataloader (torch.utils.data.DataLoader): Training data dataloader
        valid_dataloader (torch.utils.data.DataLoader): Validation set dataloader
        total_epochs (int): Total number of epochs
        early_stop (bool, optional): Whether to apply early-stopping or not. Defaults to False.
        optimtype (torch.optim.Optimizer, optional): Type of optimizer to use. Defaults to torch.optim.RMSprop.
        lr (float, optional): Learning rate. Defaults to 0.001.
        weight_decay (float, optional): Weight decay of optimizer. Defaults to 0.0.
        criterion (nn.Module, optional): Loss module. Defaults to nn.CrossEntropyLoss().
        auprc (bool, optional): Whether to compute AUPRC score or not. Defaults to False.
        save_model (str, optional): Path of file to save model with best validation performance, modality prepended. Defaults to 'encoder.pt'.
        modalities (int List, optional): Which modalities to apply encoder to. Defaults to [0].
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        track_complexity (bool, optional): Whether to track the model's complexity or not. Defaults to True.
    """
    assert(len(modalities) == len(encoders) and len(modalities) == len(heads)), "length of modalities doesn't match"
    def _trainprocess():
        models = nn.ModuleList()
        for modality in modalities:
            model = nn.Sequential(encoders[modality], heads[modality])
            models.append(model) 

        model = ensemble._initialize(models)
        op = optimtype(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            for j in train_dataloader:
                op.zero_grad()
                out, outs = model(j)
                objective_args_dict['outs'] = outs
                if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                    loss = deal_with_objective(criterion, out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), objective_args_dict)
                else:
                    loss = deal_with_objective(criterion, out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), objective_args_dict)
                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    out, outs = model(j)
                    objective_args_dict['outs'] = outs
                    if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                        loss = deal_with_objective(criterion, out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), objective_args_dict)
                    else:
                        loss = deal_with_objective(criterion, out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), objective_args_dict)
                    totalloss += loss*len(j[-1])
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0).cpu().numpy()

            true = torch.cat(true, 0).cpu().numpy()
            totals = true.shape[0]
            valloss = totalloss/totals
            acc = accuracy_score(true, pred)
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                    " acc: "+str(acc))
            if acc > bestacc:
                patience = 0
                bestacc = acc
                print("Saving Best")
                torch.save(model, save_model)
            else:
                patience += 1
            if early_stop and patience > 7:
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
    if track_complexity:
        all_in_one_train(_trainprocess, encoders+heads)
    else:
        _trainprocess()


def single_test(model, test_dataloader, auprc=False, task='classification', criterion=None, save_preds=None, save_acc=None):
    """Test ensemble model on one dataloader.

    Args:
        encoder (nn.Module List): Unimodal encoder module per modality
        head (nn.Module List): Module which takes in encoded unimodal input and predicts output per modality.
        test_dataloader (torch.utils.data.DataLoader): Data Loader for test set.
        auprc (bool, optional): Whether to output AUPRC or not. Defaults to False.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        criterion (nn.Module, optional): Loss module. Defaults to None.

    Returns:
        dict: Dictionary of (metric, value) relations.
    """
    objective_args_dict={}
    with torch.no_grad():
        pred = []
        true = []
        totalloss = 0
        pts = []
        for j in test_dataloader:
            out, outs = model(j)
            # print('out', outs[0].shape, outs[1].shape)
            objective_args_dict['outs'] = outs
            if criterion is not None:
                loss = criterion(out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                totalloss += loss*len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0).cpu().numpy()
        true = torch.cat(true, 0).cpu().numpy()
        totals = true.shape[0]
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if criterion is not None:
            print("loss: " + str(totalloss / totals))
        acc = accuracy_score(true, pred)
        print("acc: "+str(acc))
        if save_acc:
            with open(save_acc[0], 'rb') as f:
                results = pickle.load(f)
            results['additive'][save_acc[1]]['acc'] = acc
            with open(save_acc[0], 'wb') as f:
                pickle.dump(results, f)
            print('Write to', save_acc[0])
        if save_preds:
            with open(save_preds, 'wb') as f:
                pickle.dump(pred, f)
        return acc


def test(model, test_dataloaders_all, dataset='default', method_name='My method', auprc=False, task='classification', criterion=None, no_robust=False, save_preds=None, save_acc=None):
    """Test unimodal model on all provided dataloaders.

    Args:
        encoder (nn.Module): Encoder module
        head (nn.Module): Module which takes in encoded unimodal input and predicts output.
        test_dataloaders_all (dict): Dictionary of noisetype, dataloader to test.
        dataset (str, optional): Dataset to test on. Defaults to 'default'.
        method_name (str, optional): Method name. Defaults to 'My method'.
        auprc (bool, optional): Whether to output AUPRC scores or not. Defaults to False.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        criterion (nn.Module, optional): Loss module. Defaults to None.
        no_robust (bool, optional): Whether to not apply robustness methods or not. Defaults to False.
    """
    if no_robust:
        return single_test(model, test_dataloaders_all, auprc, task, criterion, save_preds=save_preds, save_acc=save_acc)

    def _testprocess():
        single_test(model, test_dataloaders_all[list(
            test_dataloaders_all.keys())[0]][0], auprc, task, criterion)
    all_in_one_test(_testprocess, [model])

    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print("Testing on noisy data ({})...".format(noisy_modality))
        robustness_curve = dict()
        for test_dataloader in tqdm(test_dataloaders):
            single_test_result = single_test(
                model, test_dataloader, auprc, task, criterion)
            for k, v in single_test_result.items():
                curve = robustness_curve.get(k, [])
                curve.append(v)
                robustness_curve[k] = curve
        for measure, robustness_result in robustness_curve.items():
            robustness_key = '{} {}'.format(dataset, noisy_modality)
            print("relative robustness ({}, {}): {}".format(noisy_modality, measure, str(
                relative_robustness(robustness_result, robustness_key))))
            if len(robustness_curve) != 1:
                robustness_key = '{} {}'.format(robustness_key, measure)
            print("effective robustness ({}, {}): {}".format(noisy_modality, measure, str(
                effective_robustness(robustness_result, robustness_key))))
            fig_name = '{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure)
            single_plot(robustness_result, robustness_key, xlabel='Noise level',
                        ylabel=measure, fig_name=fig_name, method=method_name)
            print("Plot saved as "+fig_name)