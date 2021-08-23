#%%
import re
from util import read_data, feature_extract
from RSDDDataset import RSDDDataset
train = read_data('train', 100)
train_data = RSDDDataset(train)
#%%
feature = feature_extract(train)
#%%
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split

x = feature[:, 0:7]
y = feature[:, 8]
train_data,test_data,train_label,test_label =train_test_split(x,y, random_state=1, train_size=0.6,test_size=0.4) #sklearn.model_selection.
#print(train_data.shape)
 
#3.训练svm分类器
classifier=svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovo') # ovr:一对多策略
classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优先
 
 
#也可直接调用accuracy_score方法计算准确率

# %%
c_list = {0.001, 0.01, 0.1, 1, 2, 5, 10}
k_list = {'poly', 'linear', 'sigmoid', 'rbf'}
for c in c_list:
    for k in k_list:
        classifier=svm.SVC(C=c,kernel='poly') # ovr:一对多策略
        classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优先
    
        from sklearn.metrics import accuracy_score, f1_score, recall_score, auc
        print('c=',c)
        tra_label=classifier.predict(train_data) #训练集的预测标签
        tes_label=classifier.predict(test_data) #测试集的预测标签
        print("训练集ACC：", accuracy_score(train_label,tra_label) )
        print("测试集ACC：", accuracy_score(test_label,tes_label) )
        print("训练集F1：", f1_score(train_label,tra_label, average='macro') )
        print("测试集F1：", f1_score(test_label,tes_label, average='macro') )
        print("训练集recall：", recall_score(train_label,tra_label) )
        print("测试集recall：", recall_score(test_label,tes_label) )

# %%
print(np.sum(tes_label))
print(len(train))
# %%
