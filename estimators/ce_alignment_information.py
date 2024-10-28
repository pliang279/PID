import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Datasets

class MultimodalDataset(Dataset):
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels
    self.num_modalities = len(self.data)
  
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return tuple([self.data[i][idx] for i in range(self.num_modalities)] + [self.labels[idx]])

# Models

batch_size = 256

def sinkhorn_probs(matrix, x1_probs, x2_probs):
    matrix = matrix / (torch.sum(matrix, dim=0, keepdim=True) + 1e-8) * x2_probs[None]
    sum = torch.sum(matrix, dim=1)
    if torch.allclose(sum, x1_probs, rtol=0, atol=0.01):
        return matrix, True
    matrix = matrix / (torch.sum(matrix, dim=1, keepdim=True) + 1e-8) * x1_probs[:, None]
    sum = torch.sum(matrix, dim=0)
    if torch.allclose(sum, x2_probs, rtol=0, atol=0.01):
        return matrix, True
    return matrix, False

def mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

def simple_discrim(xs, y, num_labels):
    shape = [x.size(1) for x in xs] + [num_labels]
    p = torch.ones(*shape) * 1e-8
    for i in range(len(y)):
        p[tuple([torch.argmax(x[i]).item() for x in xs] + [y[i].item()])] += 1
    p /= torch.sum(p)
    p = p.cuda()
    
    def f(*x):
        x = [torch.argmax(xx, dim=1) for xx in x]
        return torch.log(p[tuple(x)])

    return f

class Discrim(nn.Module):
    def __init__(self, x_dim, hidden_dim, num_labels, layers, activation):
        super().__init__()
        self.mlp = mlp(x_dim, hidden_dim, num_labels, layers, activation)
    def forward(self, *x):
        x = torch.cat(x, dim=-1)
        return self.mlp(x)

class CEAlignment(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation):
        super().__init__()

        self.num_labels = num_labels
        self.mlp1 = mlp(x1_dim, hidden_dim, embed_dim * num_labels, layers, activation)
        self.mlp2 = mlp(x2_dim, hidden_dim, embed_dim * num_labels, layers, activation)

    def forward(self, x1, x2, x1_probs, x2_probs):
        x1_input = x1
        x2_input = x2

        q_x1 = self.mlp1(x1).unflatten(1, (self.num_labels, -1))
        q_x2 = self.mlp2(x2).unflatten(1, (self.num_labels, -1))

        q_x1 = (q_x1 - torch.mean(q_x1, dim=2, keepdim=True)) / torch.sqrt(torch.var(q_x1, dim=2, keepdim=True) + 1e-8)
        q_x2 = (q_x2 - torch.mean(q_x2, dim=2, keepdim=True)) / torch.sqrt(torch.var(q_x2, dim=2, keepdim=True) + 1e-8)

        align = torch.einsum('ahx, bhx -> abh', q_x1, q_x2) / math.sqrt(q_x1.size(-1))
        align_logits = align
        align = torch.exp(align)

        normalized = []
        for i in range(align.size(-1)):
            current = align[..., i]
            for j in range(500):
                current, stop = sinkhorn_probs(current, x1_probs[:, i], x2_probs[:, i])
                if stop:
                    break
            normalized.append(current)
        normalized = torch.stack(normalized, dim=-1)

        if torch.any(torch.isnan(normalized)):
            print(align_logits)
            print(align)
            print(normalized)
            raise Exception('nan')

        return normalized

class CEAlignmentInformation(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels,
                 layers, activation, discrim_1, discrim_2, discrim_12, p_y):
        super().__init__()
        self.num_labels = num_labels
        self.align = CEAlignment(x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation)
        self.discrim_1 = discrim_1
        if isinstance(self.discrim_1, nn.Module):
            self.discrim_1.eval()
        self.discrim_2 = discrim_2
        if isinstance(self.discrim_2, nn.Module):
            self.discrim_2.eval()
        self.discrim_12 = discrim_12
        if isinstance(self.discrim_12, nn.Module):
            self.discrim_12.eval()
        self.register_buffer('p_y', p_y)

    def align_parameters(self):
        return list(self.align.parameters())

    def forward(self, x1, x2, y):
        with torch.no_grad():
            p_y_x1 = nn.Softmax(dim=-1)(self.discrim_1(x1))
            p_y_x2 = nn.Softmax(dim=-1)(self.discrim_2(x2))
        align = self.align(torch.flatten(x1, 1, -1), torch.flatten(x2, 1, -1), p_y_x1, p_y_x2)

        y = nn.functional.one_hot(y.squeeze(-1), num_classes=self.num_labels)

        # sample method: P(X1)
        # coeff: P(Y | X1) Q(X2 | X1, Y)
        # log term: log Q(X2 | X1, Y) - logsum_Y' Q(X2 | X1, Y') Q(Y' | X1)

        q_x2_x1y = align / (torch.sum(align, dim=1, keepdim=True) + 1e-8)
        log_term = torch.log(q_x2_x1y + 1e-8) - torch.log(torch.einsum('aby, ay -> ab', q_x2_x1y, p_y_x1) + 1e-8)[:, :, None]
        # That's all we need for optimization purposes
        loss = torch.mean(torch.sum(torch.sum(p_y_x1[:, None, :] * q_x2_x1y * log_term, dim=-1), dim=-1))

        # Now, we calculate the MI terms
        p_y_x1_sampled = torch.sum(p_y_x1 * y, dim=-1)
        p_y_x2_sampled = torch.sum(p_y_x2 * y, dim=-1)
        with torch.no_grad():
            p_y_x1x2 = nn.Softmax(dim=-1)(self.discrim_12(x1, x2))
        p_y_x1x2_sampled = torch.sum(p_y_x1x2 * y, dim=-1)
        p_y_sampled = torch.sum(self.p_y[None] * y, dim=-1)

        mi_y_x1 = torch.mean(torch.sum(p_y_x1 * (torch.log(p_y_x1) - torch.log(self.p_y)[None]), dim=-1))
        mi_y_x2 = torch.mean(torch.sum(p_y_x2 * (torch.log(p_y_x2) - torch.log(self.p_y)[None]), dim=-1))
        mi_y_x1x2 = torch.mean(torch.sum(p_y_x1x2 * (torch.log(p_y_x1x2) - torch.log(self.p_y)[None, None]), dim=-1))
        mi_q_y_x1x2 = p_y_x1[:, None, :] * q_x2_x1y * (log_term + torch.log(p_y_x1 + 1e-8)[:, None, :] - torch.log(self.p_y + 1e-8)[None, None, :])
        mi_q_y_x1x2 = torch.sum(torch.sum(mi_q_y_x1x2, dim=-1), dim=-1) # anchored by x1 -- take mean to get MI
        mi_q_y_x1x2 = torch.mean(mi_q_y_x1x2)

        print('m', torch.stack([mi_y_x1, mi_y_x2, mi_y_x1x2, mi_q_y_x1x2]))

        redundancy = mi_y_x1 + mi_y_x2 - mi_q_y_x1x2
        unique1 = mi_q_y_x1x2 - mi_y_x2
        unique2 = mi_q_y_x1x2 - mi_y_x1
        synergy = mi_y_x1x2 - mi_q_y_x1x2

        print('r', torch.stack([redundancy, unique1, unique2, synergy]))

        return loss, torch.stack([redundancy, unique1, unique2, synergy], dim=0), align

# Training Loops

def train_discrim(model, train_loader, optimizer, data_type, num_epoch=40):
    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = []
            for j in range(len(data_type)):
                xs = [data_batch[data_type[j][i] - 1] for i in range(len(data_type[j]))]
                x_batch = torch.cat(xs, dim=1).cuda()
                if j != len(data_type) - 1:
                    x_batch = x_batch.float()
                inputs.append(x_batch)
            y = inputs[-1]
            inputs = inputs[:-1]

            logits = model(*inputs)
            loss = nn.CrossEntropyLoss()(logits, y.squeeze(-1))
            loss.backward()

            optimizer.step()

            if (_iter + 1) % 20 == 0 and i_batch % 1024 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())

def eval_discrim(model, test_loader, data_type):
    losses = []
    for i_batch, data_batch in enumerate(test_loader):
        inputs = []
        for j in range(len(data_type)):
            xs = [data_batch[data_type[j][i] - 1] for i in range(len(data_type[j]))]
            x_batch = torch.cat(xs, dim=1).cuda()
            if j != len(data_type) - 1:
                x_batch = x_batch.float()
            inputs.append(x_batch)
        y = inputs[-1]
        inputs = inputs[:-1]

        logits = model(*inputs)
        loss = nn.CrossEntropyLoss()(logits, y.squeeze(-1))
        losses.append(loss.item())

        if i_batch % 1024 == 0:
            print('i_batch: ', i_batch, ' loss: ', loss.item())
    print('Eval loss:', sum(losses) / len(losses))

def train_ce_alignment(model, train_loader, opt_align, data_type, num_epoch=10):
    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
            opt_align.zero_grad()

            x1s = [data_batch[data_type[0][i] - 1] for i in range(len(data_type[0]))]
            x2s = [data_batch[data_type[1][i] - 1] for i in range(len(data_type[1]))]
            ys = [data_batch[data_type[2][i] - 1] for i in range(len(data_type[2]))]

            x1_batch = torch.cat(x1s, dim=1).float().cuda()
            x2_batch = torch.cat(x2s, dim=1).float().cuda()
            y_batch = torch.cat(ys, dim=1).cuda()

            loss, _, _ = model(x1_batch, x2_batch, y_batch)
            loss.backward()

            opt_align.step()

            if (_iter + 1) % 1 == 0 and i_batch % 1 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' align_loss: ', loss.item())

def eval_ce_alignment(model, test_loader, data_type):
    results = []
    aligns = []

    for i_batch, data_batch in enumerate(test_loader):
        x1s = [data_batch[data_type[0][i] - 1] for i in range(len(data_type[0]))]
        x2s = [data_batch[data_type[1][i] - 1] for i in range(len(data_type[1]))]
        ys = [data_batch[data_type[2][i] - 1] for i in range(len(data_type[2]))]

        x1_batch = torch.cat(x1s, dim=1).float().cuda()
        x2_batch = torch.cat(x2s, dim=1).float().cuda()
        y_batch = torch.cat(ys, dim=1).cuda()

        with torch.no_grad():
            _, result, align = model(x1_batch, x2_batch, y_batch)
        results.append(result)
        aligns.append(align)

    results = torch.stack(results, dim=0)
 
    return results, aligns

def critic_ce_alignment(x1, x2, labels, num_labels, train_ds, test_ds, discrim_1=None, discrim_2=None, discrim_12=None, learned_discrim=True, shuffle=True, discrim_epochs=40, ce_epochs=10):
    """
    Usage: critic_ce_alignment(x1, x2, labels, num_labels, train_ds, test_ds, discrim_1=None, discrim_2=None, discrim_12=None, learned_discrim=True, shuffle=True, discrim_epochs=40, ce_epochs=10)
    x1: N * K
    x2: N * K
    labels: N * 1 (torch.long)
    num_labels: The maximum label + 1
    train_ds: Dataset consisting of (x1, x2, y) tuples
    test_ds: Dataset consisting of (x1, x2, y) tuples
    discrim_1, discrim_2, discrim_12: Unimodal models and a multimodal model. Multimodal model is invoked as `discrim_12(x1, x2)` and is expected to output logits as N * C where C is num_labels. If these are set to None (not recommended), a simple model is learned instead.

    You can get train and test datasets by e.g.:
    ```
    from ce_alignment_information import MultimodalDataset
    train_ds = MultimodalDataset([x1, x2], y)
    ```
    """

    if discrim_1 is not None:
        model_discrim_1, model_discrim_2, model_discrim_12 = discrim_1, discrim_2, discrim_12
    elif learned_discrim:
        model_discrim_1 = Discrim(x_dim=x1.size(1), hidden_dim=32, num_labels=num_labels, layers=3, activation='relu').cuda()
        model_discrim_2 = Discrim(x_dim=x2.size(1), hidden_dim=32, num_labels=num_labels, layers=3, activation='relu').cuda()
        model_discrim_12 = Discrim(x_dim=x1.size(1) + x2.size(1), hidden_dim=32, num_labels=num_labels, layers=3, activation='relu').cuda()

        for model, data_type in [
            (model_discrim_1, ([1], [0])),
            (model_discrim_2, ([2], [0])),
            (model_discrim_12, ([1], [2], [0])),
        ]:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            train_loader1 = DataLoader(train_ds, shuffle=shuffle, drop_last=True,
                                    batch_size=batch_size,
                                    num_workers=1)
            train_discrim(model, train_loader1, optimizer, data_type=data_type, num_epoch=discrim_epochs)
            model.eval()
            test_loader1 = DataLoader(test_ds, shuffle=False, drop_last=False,
                                      batch_size=batch_size, num_workers=1)
            eval_discrim(model, test_loader1, data_type=data_type)
    else:
        model_discrim_1 = simple_discrim([x1], labels, num_labels)
        model_discrim_2 = simple_discrim([x2], labels, num_labels)
        model_discrim_12 = simple_discrim([x1, x2], labels, num_labels)

    p_y = torch.sum(nn.functional.one_hot(labels.squeeze(-1)), dim=0) / len(labels)

    def product(x):
        return x[0] * product(x[1:]) if x else 1

    model = CEAlignmentInformation(x1_dim=product(x1.shape[1:]), x2_dim=product(x2.shape[1:]),
        hidden_dim=32, embed_dim=10, num_labels=num_labels, layers=3, activation='relu',
        discrim_1=model_discrim_1, discrim_2=model_discrim_2, discrim_12=model_discrim_12,
        p_y=p_y).cuda()
    opt_align = optim.Adam(model.align_parameters(), lr=1e-3)

    train_loader1 = DataLoader(train_ds, shuffle=shuffle, drop_last=True,
                               batch_size=batch_size,
                               num_workers=1)
    test_loader1 = DataLoader(test_ds, shuffle=False, drop_last=True,
                              batch_size=batch_size,
                              num_workers=1)

    # Train and estimate mutual information
    model.train()
    train_ce_alignment(model, train_loader1, opt_align, data_type=([1], [2], [0]), num_epoch=ce_epochs)

    model.eval()
    results, aligns = eval_ce_alignment(model, test_loader1, data_type=([1], [2], [0]))
    return results, aligns, (model, model_discrim_1, model_discrim_2, model_discrim_12, p_y)


class EncoderAndHead(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
    def forward(self, *args, **kwargs):
        x = self.head(self.encoder(*args, **kwargs))
        return x

class MM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, *x):
        x = self.model(x)
        return x
