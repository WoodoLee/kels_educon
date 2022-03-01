import os
import pandas as pd
import csv

from sklearn.model_selection import StratifiedShuffleSplit
from models.DT import Decision_Tree #Decision tree
from models.SVM import SVM #Support Vector Machine
from models.ExtraTrees import Extra_Trees #Extra trees classifier
from models.GradientBoosting import Grad_Boost
from models.KNN import KNN #k-nearest neighbors
from models.RandomForest import RandomForest # Random Forest

# Nan data is filled by avg of the row data
# path_data ='./preprocessed/prepared/fill/'
path_data ='./preprocessed/prepared/fill/'

file_names = os.listdir(path_data)
flag = False

for file_name in file_names:
    df_ = pd.read_pickle(os.path.join(path_data, file_name))
    if not file_name.startswith('label'):    
        if not flag:
            flag = True
            input = df_
        elif flag:
            input = pd.concat([input, df_], axis=1)
    
    else:
        label = df_

sss = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
X, y = input, label["L2Y6_K_CS"]

for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

y_test = y_test.to_numpy()
print("train size:", X_train.shape)
print("test size:", X_test.shape)

l_size = [X_train.shape, X_test.shape]
l_models = ["DT", "SVM" , "ET" , "GB", "KNN", "RF"]
l_acc = []
l_acc_rough = []

acc, acc_r  = Decision_Tree(X_train, X_test, y_train, y_test)
l_acc.append(acc)
l_acc_rough.append(acc_r)

acc, acc_r  = SVM(X_train, X_test, y_train, y_test, kernel='rbf')
l_acc.append(acc)
l_acc_rough.append(acc_r)

acc, acc_r  = Extra_Trees(X_train, X_test, y_train, y_test)
l_acc.append(acc)
l_acc_rough.append(acc_r)

acc, acc_r  = Grad_Boost(X_train, X_test, y_train, y_test)
l_acc.append(acc)
l_acc_rough.append(acc_r)

acc, acc_r  = KNN(X_train, X_test, y_train, y_test)
l_acc.append(acc)
l_acc_rough.append(acc_r)

acc, acc_r  = RandomForest(X_train, X_test, y_train, y_test)
l_acc.append(acc)
l_acc_rough.append(acc_r)

# print(l_models)
# print(l_acc)
# print(l_acc_rough)

with open('./results/ml_results.csv', 'w',newline='') as f:
    write = csv.writer(f)
    write.writerow(["train_size" , "test_size"])
    write.writerow(l_size) 
    write.writerow(l_models)
    write.writerow(l_acc)
    write.writerow(l_acc_rough)

