import torch
import pandas as pd
import sklearn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from utils.nn_utils import *
from models.ViT import ViT_LRP_nan_excluded


X_datapaths = ['./preprocessed/prepared/nan/L2Y1.pkl','./preprocessed/prepared/nan/L2Y2.pkl','./preprocessed/prepared/nan/L2Y3.pkl','./preprocessed/prepared/nan/L2Y4.pkl','./preprocessed/prepared/nan/L2Y5.pkl','./preprocessed/prepared/nan/L2Y6.pkl']
label_datapath = './preprocessed/prepared/nan/label.pkl'
#X_datapaths = ['./preprocessed/prepared/fill/L2Y1.pkl','./preprocessed/prepared/fill/L2Y2.pkl','./preprocessed/prepared/fill/L2Y3.pkl','./preprocessed/prepared/fill/L2Y4.pkl','./preprocessed/prepared/fill/L2Y5.pkl','./preprocessed/prepared/fill/L2Y6.pkl',]
#label_datapath = './preprocessed/prepared/fill/label.pkl'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# read pickle
input_datas = [] # list of each input pandas dataframe
for datapath in X_datapaths:
    temp = pd.read_pickle(datapath)
    temp = temp.reset_index()
    temp = temp.drop(columns=['index'])
    input_datas.append(temp)


label_data = pd.read_pickle(label_datapath)
label_data = label_data.reset_index()
label_data = label_data.drop(columns=['index'])



split_list = make_split_list(input_datas)
input_concated = np.concatenate(input_datas,axis=1) # concated input. (number of instance x number of features) will be splited with kfold
seq_len = len(input_datas)
label_data = label_data - 1



CLS2IDX = {
    0 : '1등급',
    1 : '2등급',
    2 : '3등급',
    3 : '4등급',
    4 : '5등급',
    5 : '6등급',
    6 : '7등급',
    7 : '8등급',
    8 : '9등급'
}
is_regression = False

batch_size = 32
hidden_features = 100
embbed_dim = 72
n_splits = 10
kfold = KFold(n_splits=n_splits)
fold_acc_dict = {}
epoch = 50

for fold,(train_idx,test_idx) in enumerate(kfold.split(input_concated)):
    print('------------fold no---------{}----------------------'.format(fold))
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
    dataset = KELSDataSet(input_concated,label_data)
    train_loader = DataLoader(
                        dataset, 
                        batch_size=batch_size, sampler=train_subsampler)
    test_loader = DataLoader(
                        dataset,
                        batch_size=batch_size, sampler=test_subsampler)
    sample_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    assert batch_size
    (sample,label) = next(iter(sample_loader))
    sample_datas = batch_to_splited_datas(sample,split_list)
    model_E = ViT_LRP_nan_excluded.VisionTransformer(sample_datas,split_list,seq_len=6, num_classes=9, embed_dim=16*3, depth=8,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0.2, attn_drop_rate=0.2)
    model_E = model_E.to(device)
    val_accs , positive_accs = train_net(model_E,train_loader,test_loader,n_iter=epoch,device=device,mode='E',lr=0.0001,optimizer_cls = optim.AdamW)
    temp_dict = {}
    temp_dict['val_accs'] = val_accs
    temp_dict['positive_accs'] = positive_accs
    fold_acc_dict[fold] = temp_dict


    

#embedding_networks : 년차별로 맞는 mlp 리스트. 리스트 내용물에 따라 인풋 채널 개수 다름.
 # not used in traing; only used to initialize embbeding layer

val_acc_mean = np.zeros_like(fold_acc_dict[0]['val_accs'])
pos_acc_mean = np.zeros_list(fold_acc_dict[0]['positive_accs'])
for i in len(range(fold_acc_dict)):
    val_acc_mean += fold_acc_dict[i]['val_accs']
    pos_acc_mean += fold_acc_dict[i]['positive_accs']

val_acc_mean = val_acc_mean / n_splits
pos_acc_mean = pos_acc_mean / n_splits

for i in range(len(epoch)):

    print("-----------------------------------------------------------------------------------------------------")
    print(f"mean          accuracy across {n_splits} fold in {i}th epoch : {val_acc_mean[i]}%")
    print(f"mean positive accuracy across {n_splits} fold in {i}th epoch : {pos_acc_mean[i]}%")
    print("-----------------------------------------------------------------------------------------------------")



