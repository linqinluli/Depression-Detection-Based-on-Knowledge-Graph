#%%
from os import read
import nltk
from util import read_data, feature_extract

x = read_data('train', 10)
x,y = feature_extract(x)
# %%
x[3]
# %%
x = [1, 5, 8, 7, 9]
# %%
print(x[:3])
# %%
