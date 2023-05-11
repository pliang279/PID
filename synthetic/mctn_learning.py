import torch
from torch import nn
import torch.optim as optim
import numpy as np
from mctn_model import Translation, MCTN
from sklearn.metrics import accuracy_score
from torchinfo import summary


softmax = nn.Softmax()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(traindata, validdata, encoders, decoders, head, model=None, epoch=100, level=1, criterion_t=nn.MSELoss(), criterion_c=nn.MSELoss(), criterion_r=nn.CrossEntropyLoss(), mu_t=0.01, mu_c=0.01, dropout_p=0.1, early_stop=False, patience_num=15, lr=1e-4, weight_decay=0.01, op_type=torch.optim.AdamW, save='best_mctn.pt'):
    if not model:
        translations = list(enumerate(zip(encoders, decoders)))
        translations = [Translation(encoder, decoder, i).to(device) for i, (encoder, decoder) in translations]
        model = MCTN(translations, head, p=dropout_p).to(device)
    op = op_type(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler = optim.lr_scheduler.ReduceLROnPlateau(op, factor=0.5, patience=7, verbose=True)
    criterion_t = [criterion_t] * level
    criterion_c = [criterion_c] * level

    patience = 0
    bestacc = 0

    for ep in range(epoch):
        model.train()
        print('start training ---------->>')

        sum_total_loss = 0
        sum_loss = 0
        total_batch = 0
        pred = []
        true = []
        for inputs in traindata:
            src, trgs, labels = _process_input(inputs)
            translation_losses = []
            cyclic_losses = []
            total_loss = 0

            op.zero_grad()

            outs, reouts, head_out = model(src, trgs)

            for i in range(len(outs)):
                out = outs[i]
                translation_loss = 0
                for j, o in enumerate(out):
                    translation_loss += criterion_t[i](o, trgs[i][j])
                translation_loss /= out.size(0)
                translation_losses.append(translation_loss)

            for i in range(len(reouts)):
                reout = reouts[i]
                cyclic_loss = 0
                if i == 0:
                    for j, o in enumerate(reout):
                        cyclic_loss += criterion_c[i](o, src[j])
                else:
                    for j, o in enumerate(reout):
                        cyclic_loss += criterion_c[i](o, trgs[i-1][j])
                cyclic_loss /= reout.size(0)
                cyclic_losses.append(cyclic_loss)
            pred.append(torch.argmax(head_out, 1))
            true.append(labels)
            loss = criterion_r(head_out, labels)

            total_loss = mu_t * sum(translation_losses) + mu_c * sum(cyclic_losses) + loss

            sum_total_loss += total_loss
            sum_loss += loss
            total_batch += 1
            total_loss.backward()
            op.step()

        sum_total_loss /= total_batch
        sum_loss /= total_batch

        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        acc = accuracy_score(true.detach().cpu().numpy(), pred.detach().cpu().numpy())

        print('Train Epoch {}, total loss: {}, regression loss: {}, embedding loss: {}, acc: {}'.format(ep, sum_total_loss, sum_loss, sum_total_loss - sum_loss, acc))

        model.eval()
        print('Start Evaluating ---------->>')
        pred = []
        true = []
        val_loss = []
        with torch.no_grad():
            for inputs in validdata:
                src, trgs, labels = _process_input(inputs)
                _, _, head_out = model(src, trgs)
                loss = criterion_r(head_out, labels)
                val_loss.append(loss.item())
                pred.append(torch.argmax(head_out, 1))
                true.append(labels)
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            acc = accuracy_score(true.detach().cpu().numpy(), pred.detach().cpu().numpy())
            print('Eval Epoch: {}, val loss: {}, acc: {}'.format(ep, np.mean(val_loss), acc))

            # scheduler.step(np.mean(val_loss))

            if acc > bestacc:
                patience = 0
                bestacc = acc
                print('<------------ Saving Best Model')
                print()
                if save:
                    torch.save(model, save)
            else:
                patience += 1
            if early_stop and patience > patience_num:
                break
    print("Testing:", bestacc)


def test(model, testdata):
    model.eval()
    print('Start Testing ---------->>')
    pred = []
    true = []
    with torch.no_grad():
        for inputs in testdata:
            src, trgs, labels = _process_input(inputs)
            _, _, head_out = model(src, trgs)
            pred.append(torch.argmax(head_out, 1))
            true.append(labels)
        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        acc = accuracy_score(true.detach().cpu().numpy(), pred.detach().cpu().numpy())
        print('Test Acc: {}'.format(acc))


def _process_input(inputs):
    labels = inputs[-1]
    labels = labels.squeeze(len(labels.size())-1).long().to(device)
    src = inputs[0].float().to(device)
    trgs = [inputs[1].float().to(device)]
    return src, trgs, labels
