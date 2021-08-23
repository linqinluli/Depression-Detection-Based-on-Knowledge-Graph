#%%
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os
path = os.getcwd()

glove_file = datapath(path+'/glove.6B/glove.6B.300d.txt')
# 输出文件
tmp_file = get_tmpfile("word2vec_300d.txt")


print(glove2word2vec(glove_file, tmp_file))

model = KeyedVectors.load_word2vec_format(tmp_file)

model.most_similar('funny')
#%%
model.most_similar('suicide')
# %%
