import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import loaddatas as lds
import GAT

wait_total = 100
num_epoch = 500
times = 50

def train(train_mask):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[train_mask], data.y[train_mask]).backward()
    optimizer.step()


def test(train_mask, val_mask, test_mask):
    model.eval()
    logits= model(data)
    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        # print(pred)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    accs.append(F.nll_loss(logits[val_mask], data.y[val_mask]))
    print(accs)
    return accs



dataset = lds.loaddatas("Planetoid", "Cora")

for time in range(times):
    data = dataset[0]
    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()
    test_mask = data.test_mask.bool()
    model, data = locals()["GAT"].call(data, dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    best_val_acc = test_acc = 0.0
    for epoch in range(num_epoch):
        train(train_mask)
        train_acc, val_acc, tmp_test_acc, val_loss = test(train_mask, val_mask, test_mask)
        if val_acc >= best_val_acc:
            test_acc = tmp_test_acc
            best_val_acc = val_acc
            best_val_loss = val_loss
            wait_step = 0
        else:
            wait_step += 1
            if wait_step == wait_total:
                print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc)
                break
    del data
    log = f'Epoch: {epoch}, dataset name: ' + "Cora" + ', Method: ' + "GAT" + ' Test: {:.4f} \n'
    print((log.format(test_acc)))

