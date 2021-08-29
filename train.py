#%%
from pickle import TRUE
import numpy as np
from util import read_data, feature_extract, calAUC
#%%
train = read_data('train', 3000)
val = read_data('test', 2000)
# train=np.load('half.npy', allow_pickle=True)
# val = np.load('val_half.npy', allow_pickle=True)
# train=train.tolist()
# val = val.tolist()
#%%
train_x, train_y = feature_extract(train)
val_x, val_y = feature_extract(val)
np.save('bert_train_x', train_x)
np.save('bert_train_y', train_y)
np.save('bert_val_x', val_x)
np.save('bert_val_y', val_y)
#%%
from pickle import TRUE
import numpy as np
train_x = np.load('train_x.npy', allow_pickle=True)
train_y = np.load('train_y.npy', allow_pickle=True)
val_x = np.load('val_x.npy', allow_pickle=True)
val_y = np.load('val_y.npy', allow_pickle=True)
#%%
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DEPredictor
from RSDDDataset import RSDDDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from util import calAUC
lr = 0.0001
BatchSize = 32
EPOCH = 30

from model import DEPredictor

train_data = RSDDDataset(train_x, train_y)
val_data = RSDDDataset(val_x, val_y)

train_dataloader = DataLoader(train_data, batch_size=BatchSize, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=10000, shuffle=True)

property_num = train_x.shape[1]
# weight = torch.tensor([0.1]).cuda()
loss_fn = nn.BCELoss()
predictor = DEPredictor(property_num).cuda()
print(predictor)

predictor = predictor.double()
optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
#optimizer = torch.optim.SGD(predictor.parameters(), lr=lr)

loss_list = []
auc_list = []
acc_list = []
f1_list = []
best_auc = 0

for epoch in range(100):
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

        valy = valy.cpu().detach().numpy()
        valpred = valpred.cpu().detach().numpy()

        valpred = np.around(valpred)
        auc = calAUC(valpred, valy)
        acc = accuracy_score(valy, valpred)
        f1 = f1_score(valy, valpred, average='macro')
        recall = recall_score(valy, valpred, average='macro')
        precision = precision_score(valy, valpred, average='macro')
        target_names = ['Depression', 'Control']
        report = classification_report(valy,
                                       valpred,
                                       target_names=target_names)
        auc_list.append(auc)
        f1_list.append(f1)
        acc_list.append(acc)
        if (auc > best_auc):
            best_auc = auc
            best_score = report
            fn = (("lr_" + str(lr)) + ".pt")
            torch.save(predictor.state_dict(), fn)

        # print(report)
        print("valid AUC:", round(auc, 4), '\tbest AUC:', round(best_auc, 4))

# %%
print(len(auc_list))
# %%
from matplotlib import pyplot 
import matplotlib.pyplot as plt
plt.plot(auc_list)
plt.ylabel('some numbers')
plt.show()
# %%

# %%
