#%%
from os import read
import nltk
from nltk.util import pr
from numpy.lib.function_base import percentile
from util import read_data, feature_extract, read_thesaurus

words_dic = [
    'unenergetic', 'slow-moving', 'restless', 'sleepy', 'tired', 'suicide',
    'stressful', 'distracted', 'interested', 'negative', 'unconfident'
]
# %%
tar_words = []
for word_name in words_dic:
    tar_words.append(read_thesaurus(word_name))
    print(len(read_thesaurus(word_name)))
# %%
train = read_data('train', 3333)
res, label, score_res = feature_extract(train)

# %%
import numpy as np
dep_sum = [0] * 11
con_sum = [0] * 11
dep_score = [0] * 11
con_score = [0] * 11
con_data = []
dep_data = []
for i in range(len(label)):
    if label[i] == 1:
        dep_data.append(score_res[i])
        for j in range(len(dep_sum)):
            dep_sum[j] = dep_sum[j] + res[i][j]
            dep_score[j] = dep_score[j] + score_res[i][j]
    else:
        con_data.append(score_res[i])
        for j in range(len(con_sum)):
            con_sum[j] = con_sum[j] + res[i][j]
            con_score[j] = con_score[j] + score_res[i][j]

# %%
con_data = np.array(con_data)
dep_data = np.array(dep_data)
for j in range(len(dep_sum)):
    # print((dep_sum[j])/label.sum(), (con_sum[j])/(len(label)-label.sum()))

    # print((dep_score[j])/label.sum(), (con_score[j])/(len(label)-label.sum()))
    print('[', np.percentile(dep_data[:, j], 25), ',',
          np.percentile(dep_data[:, j], 50), ',',
          np.percentile(dep_data[:, j], 75), '],')
# %%
for i in range(11):
    print(np.average(score_res[:, i]))
# %%
percent = [
    [2.25, 5.875, 14.125],
    [5.3125, 12.75, 29.4375],
    [1.5, 3.5, 9.0],
    [2.75, 7.375, 16.75],
    [6.25, 14.125, 36.375],
    [0.0, 2.0, 4.0],
    [6.75, 14.875, 34.75],
    [0.0, 0.75, 2.0],
    [13.75, 31.75, 83.875],
    [0.0, 1.0, 5.0],
    [1.75, 4.5, 11.0]
]

# %%
percent[1][2]
# %%
