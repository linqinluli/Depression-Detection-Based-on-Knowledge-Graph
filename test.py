#%%
from os import read
import nltk
from util import read_data, word_count

data = read_data('train', 100)
#%%
for post in data[0]['posts']:
    print([post[0]])
#%%
sum_list = []
con = []
dep = []
for person in data:
    sum = 0
    num = len(person['posts'])
    for post in person['posts']:
        sum = sum + word_count(post[1], 'suicide')
    if person['label'] == 'control':
        con.append(sum/num)
        con.append(sum)
    else:
        dep.append(sum/num)
        dep.append(sum)
    sum_list.append(sum/num)
    sum_list.append(sum)

# %%
import numpy as np
print(np.average(sum_list))
print(np.average(dep))
print(np.average(con))
# %%
