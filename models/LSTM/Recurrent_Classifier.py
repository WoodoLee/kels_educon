
import torch
import torch.nn as nn
from utils.nn_utils import *



class RecurrentClassifier(nn.Module):

	def __init__(self,sample_datas,split_list, embedding_dim=64, hidden_dim=32, output_size=9,act_layer = nn.Sigmoid, model='LSTM', dropout=0.5):
		"""
		feature_size : 인풋 시퀀스 피쳐 사이즈
		embedding_dim : LSTM 인풋 피쳐 사이즈
		hidden_dim : LSTM 히든 레이어 사이즈.
		인풋 : (batch, seq, feature_size)
		"""
		super(RecurrentClassifier, self).__init__()
		self.embbeding_networks = make_embbeding_networks(sample_datas,hidden_features = embedding_dim * 2,out_features=embedding_dim,dropout=0.2)      
        
		self.split_list = split_list
		self.model = model
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.act_layer = act_layer()
		# self.feature_size = feature_size
		if model == 'LSTM':
			self.rec = nn.LSTM(embedding_dim, hidden_dim,bidirectional=True,num_layers=3,batch_first=True,dropout=0.5)
		elif model == 'GRU':
			self.rec = nn.GRU(embedding_dim, hidden_dim, num_layers=1,batch_first=True)
		elif model == 'RNN':
			self.rec = nn.RNN(embedding_dim, hidden_dim, num_layers=1,batch_first=True)
		else:
			assert()

		#self.hidden2out = MLP(hidden_dim, out_features=output_size)
		self.outputhead = nn.Sequential(
			nn.Linear(hidden_dim*2,hidden_dim),
			act_layer(),
			nn.Linear(hidden_dim,output_size)
										

		)
		#self.hidden2out = nn.Linear(hidden_dim,output_size)
		self.softmax = nn.LogSoftmax()
		# self.encoder = MLP(feature_size,out_features=embedding_dim,act_layer=act_layer)
		#self.Mlp = (feature_size,embedding_dim)
		

		self.dropout_layer = nn.Dropout(p=0.5)




	def forward(self, x):
		batch_size = x.shape[0]
		
		B = x.shape[0]
		datas = batch_to_splited_datas(x,self.split_list)
        # 지금  생생각각해해보보니 batch에서 nan이 한번만 나와도 더하는 과정에서 다 nan됨. grad = nan이면 0으로 해야할듯..?
        
		emb_batch_list, attn_mask= batch_to_embbedings(datas,self.embbeding_networks) # can be used for contrastive loss
        # for emb in emb_batch_list:
        #     if emb.requires_grad == True:
        #         emb.register_hook(self.nan_watch)

        #self.save_emb_batch_list(emb_batch_list)
        # emb_batch_list : 임베딩 벡터들의 리스트. 얘를 이제 batch x seq x feature 행렬로 쌓음
        
		emb_batched_seq = torch.stack(emb_batch_list).transpose(0,1)
		
		#attn_mask = make_attn_mask(emb_batched_seq)
		x = emb_batched_seq 		
		#self.hidden = self.init_hidden(x.size(-1))
		x[torch.isnan(x)] = 0
		embeds = x
		if self.model == 'LSTM':
		
			outputs, (ht, ct) = self.rec(embeds)
			ht = outputs[:,-1,:]
		else:
			outputs, ht = self.rec(embeds)

		# ht is the last hidden state of the sequences
		# ht = (1 x batch_size x hidden_dim)
		# ht[-1] = (batch_size x hidden_dim)

		#output = self.dropout_layer(ht[-1])
		output = self.dropout_layer(ht)
		output = self.outputhead(output)
		#output = self.softmax(output) #criterion에 softmax를 썼기 때문에 붙이면 안됨.

		return output   #output에 5차원으로 해서 라벨을 정수화한것과 비교...? 라벨이 5차원이되어야함