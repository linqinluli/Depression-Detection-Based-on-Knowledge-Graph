#%%
from pickle import TRUE
import numpy as np
from util import read_data, feature_extract, calAUC
#%%
# train = read_data('train', 1000)
# val = read_data('val', 1000)
train=np.load('half.npy', allow_pickle=True)
val = np.load('val_half.npy', allow_pickle=True)
train=train.tolist()
val = val.tolist()
#%%
train_x, train_y = feature_extract(train)
val_x, val_y = feature_extract(val)
np.save('train_x', train_x)
np.save('train_y', train_y)
np.save('val_x', val_x) 
np.save('val_y', val_y)

#%%
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DEPredictor
from RSDDDataset import RSDDDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
lr = 0.0001
BatchSize = 16
EPOCH = 30

from model import DEPredictor

train_data = RSDDDataset(train_x, train_y)
val_data = RSDDDataset(val_x, val_y)

train_dataloader = DataLoader(train_data, batch_size=BatchSize, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=10000, shuffle=True)

property_num = train_x.shape[1]

loss_fn = nn.BCELoss()
predictor = DEPredictor(property_num).cuda()
print(predictor)

predictor = predictor.double()
optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
#optimizer = torch.optim.SGD(predictor.parameters(), lr=lr)

loss_list = []
best_auc = 0

for epoch in range(80):
    epoch_loss = []
    for batch, (feature, label) in enumerate(train_dataloader):
        x, y = feature.cuda(), label.cuda()
        pred = predictor(x).squeeze(-1)
        loss = loss_fn(pred, y.double())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        loss_list.append(loss.data.cpu().numpy())
        epoch_loss.append(loss.data.cpu().numpy())
    if True:
        print(epoch, 'loss:', np.mean(np.array(epoch_loss)))
        for batch, (feature, label) in enumerate(val_dataloader):
            valx, valy = feature.cuda(), label.cuda()
            valpred = predictor(valx)

        auc = 1
        
        valy = valy.cpu().detach().numpy()
        valpred = valpred.cpu().detach().numpy()

        valpred = np.around(valpred)
        auc = calAUC(valpred, valy)
        acc = accuracy_score(valy, valpred)
        f1 = f1_score(valy, valpred, average='macro')
        recall = recall_score(valy, valpred, average='macro')
        precision = precision_score(valy, valpred, average='macro')
        target_names = ['Depression', 'Control']
        report = classification_report(valy, valpred, target_names=target_names)
        if (auc > best_auc):
            best_auc = auc
            best_score = report
            fn = (("lr_" + str(lr)) + ".pt")
            torch.save(predictor.state_dict(), fn)

        print(report)
        print("valid AUC:", round(auc, 4), '\tbest AUC:', round(best_auc, 4))

# %%
print(best_score)
# %%
np.save('train_x', train_x)
np.save('train_y', train_y)
# %%
from collections import Counter
print(Counter(valy.flatten()))
# %%
loss_list
# %%
