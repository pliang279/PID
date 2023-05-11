"""Implements ensemble of outputs of unimodal models."""

import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torch.autograd import Variable

class AdditiveEnsemble(nn.Module):
    """Adds outputs of each modality."""
    def __init__(self):
        """Initialize Additive Ensemble Module.
        """
        super(AdditiveEnsemble, self).__init__()

    def _initialize(self, models):
        """Initialize Additive Ensemble Module.
        :param models: List of unimodal models
        """
        print('initializing ensemble model')

        self.models = models
        self.modelnum = len(self.models)
        # self.vars = []
        # for i in range(self.modelnum):
        #     # self.vars.append(self.models[i])
        #     self.vars.append(nn.Parameter(torch.tensor(1.0), requires_grad=True).cuda())
        return self

    def forward(self, j):
        """
        Forward Pass of Stack.
        
        :param j: Data input of all modalities
        """
        outs = []
        sum = 0
        for i in range(self.modelnum):
          model = self.models[i]
          out = model(j[i].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
          outs.append(out)
          sum += out
        return sum / self.modelnum , outs