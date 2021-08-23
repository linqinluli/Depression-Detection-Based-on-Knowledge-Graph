#%%
import tqdm
import json

path = 'datasets/RSDD/validation'
data = []
num = 20000
with open(path) as f:
    for i in tqdm.tqdm(range(num)):
        lines = f.readline()
        line_data = json.loads(lines)
        if line_data[0]['label'] == 'control':
            line_data[0]['label'] = 0
        elif line_data[0]['label'] == 'depression':
            line_data[0]['label'] = 1
        else:
            continue
        data.append(line_data[0])
#%%
dep_data = []
con_data = []

for i in tqdm.tqdm(range(len(data))):
    if data[i]['label'] == 0:
        con_data.append(data[i])
    elif data[i]['label'] == 1:
        dep_data.append(data[i])
#%%
len(con_data)
# %%
len(dep_data)
# %%
final_data = []
for i in range(777):
    final_data.append(con_data[i+777])
    final_data.append(dep_data[i+33])

# %%
import numpy as np
np.save('val_half.npy', final_data)
a=np.load('val_half.npy', allow_pickle=True)
a=a.tolist()
# %%
len(a)
# %%
