
#%%
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split

train_data = np.load('train_x.npy', allow_pickle=True)
train_label = np.load('train_y.npy', allow_pickle=True)
test_data = np.load('train_x.npy', allow_pickle=True)
test_label = np.load('train_y.npy', allow_pickle=True)


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
