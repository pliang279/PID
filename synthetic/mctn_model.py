import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1):
        super(MLP, self).__init__()
        self.indim = indim
        self.hiddim = hiddim
        self.outdim = outdim
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.fc3 = nn.Linear(outdim, outdim*2)
        self.fc4 = nn.Linear(outdim*2, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        if len(x.shape) > 1 and x.shape[1] == self.indim:
            output = F.relu(self.fc(x))
            if self.dropout:
                output = self.dropout_layer(output)
            output2 = self.fc2(output)
        else:
            output = F.relu(self.fc3(x))
            if self.dropout:
                output = self.dropout_layer(output)
            output2 = self.fc4(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        return output2


class Translation(nn.Module):
    
    def __init__(self, encoder, decoder, i):
        super(Translation, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        joint_embbed = self.encoder(src)
        out = self.decoder(joint_embbed)
        return out, joint_embbed


class MCTN(nn.Module):

    def __init__(self, translations, head, p=0.2):
        super(MCTN, self).__init__()
        self.translations = nn.ModuleList(translations)
        self.dropout = nn.Dropout(p)
        self.head = head

    def forward(self, src, trgs):
        out, joint_embbed = self.translations[0](src)
        outs = [out]
        reouts = []
        joint_embbeds = []
        # if self.training:
        reout, joint_embbed = self.translations[0](out)
        reouts = [reout]
        joint_embbeds = [joint_embbed]
        if len(trgs) > 1:
            for i in range(1, len(trgs)):
                print(i, trgs.shape, len(self.translations))
                out, joint_embbed = self.translations[i](joint_embbed)
                reout, joint_embbed = self.translations[i](out)
                outs.append(out)
                reouts.append(reout)
                joint_embbeds.append(joint_embbed)
            out, joint_embbed = self.translations[-1](joint_embbed)
            outs.append(out)
            joint_embbeds.append(joint_embbed)
        # else:
        #     for i in range(1, len(trgs)):
        #         joint_embbed = self.translations[i-1].encoder(out)
        #         out, joint_embbed = self.translations[i](joint_embbed)
        #         outs.append(out)
        #         joint_embbeds.append(joint_embbed)
        #     joint_embbed = self.translations[-1].encoder(out)
        #     joint_embbeds.append(joint_embbed)
        head_out = self.head(joint_embbed)
        head_out = self.dropout(head_out)
        return outs, reouts, head_out
