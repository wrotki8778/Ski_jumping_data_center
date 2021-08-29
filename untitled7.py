# -*- coding: utf-8 -*-
"""
Created on Sat May  1 08:57:38 2021

@author: kubaf
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib.pyplot import plot, scatter
import statistics as st
import random
os.chdir('C:/Users/kubaf/Documents/GitHub/Ski_jumping_data_center')

def read_ratings(df, column_names = ['delty','training']):
    counter = range(1, 11)
    for col_name in column_names:
        for i in counter:
            string_name = col_name + '_' + str(i)
            df[string_name] = df[col_name].shift(i, fill_value=0)
    return df

def is_t(string):
    if 'T' in string:
        return 1
    return 0

def uniforming_tensors(tensor):
    pad_ext = max(0,10-tensor.shape[0])
    result = F.pad(input=tensor, pad=(0,0,pad_ext,0), mode='constant', value=0)
    result[-1,-1] = 0
    return result[-10:]

actual_results = pd.read_csv(os.getcwd()+'\\all_results.csv')
actual_comps = pd.read_csv(os.getcwd()+'\\all_comps.csv')
actual_comps = actual_comps[actual_comps['team'] == 0]
actual_comps = actual_comps[actual_comps['training'] == 0]
actual_ratings = pd.read_csv(os.getcwd()+'\\all_ratings.csv')
actual_ratings['training'] = actual_ratings['id'].apply(is_t)

u = [x for _, x in actual_ratings.groupby(['codex'])]
big_u = [read_ratings(x) for x in u]
new_actual_ratings = pd.DataFrame().append(big_u)

actual_results_lstm = pd.merge(actual_results, actual_comps,
                               on='id', how='inner')[['dist', 'codex_x',
                                                      'wind', 'gate_points',
                                                     'hill_size_x', 'date',
                                                     'id', 'round', 'gender',
                                                     'bib', 'team', 'season']]
actual_results_lstm = actual_results_lstm[actual_results_lstm['team']==0]
actual_results_lstm = pd.merge(actual_results_lstm, new_actual_ratings,
                               left_on=['id', 'round', 'codex_x'],
                               right_on=['id', 'round', 'codex'],
                               how='left')
# [['dist', 'codex_x', 'wind', 'speed', 'hill_size_x', 'date', 'training', 'round', 'gender', 'cumm_rating','short_rating', 'bib','season', 'id']]
actual_results_lstm = actual_results_lstm.sort_values(['date', 'round'],
                                                      ascending=[True, True])
actual_results_lstm["round"] = actual_results_lstm["round"].astype('category')
actual_results_lstm["round"] = actual_results_lstm["round"].cat.codes
actual_results_lstm["gender"] = actual_results_lstm["gender"].astype('category')
actual_results_lstm["gender"] = actual_results_lstm["gender"].cat.codes
actual_results_lstm["norm_dist"] = actual_results_lstm["dist"]/actual_results_lstm["hill_size_x"]
actual_results_lstm["bib"] = pd.to_numeric(actual_results_lstm["bib"],errors='coerce')
actual_results_lstm = actual_results_lstm[actual_results_lstm['season'] > 2010]
actual_results_lstm.loc[actual_results_lstm['codex_x']>10000,'codex_x'] = 0
#actual_results_lstm = actual_results_lstm[actual_results_lstm['speed'] > 0]

actual_results_lstm = actual_results_lstm.dropna()
sc = StandardScaler()
sc2 = StandardScaler()

counter = range(1, 11)
col_names = [x + '_' + str(y) for x in ['delty', 'training'] for y in counter]

var_names = ['hill_size_x', 'cumm_rating',
             'wind', 'gate_points', 'gender', 'codex_x', 'training']+col_names
all_var_names = var_names + ['norm_dist']
input_len = len(all_var_names)
actual_results_lstm[var_names]\
    = sc.fit_transform(actual_results_lstm[var_names])
actual_results_lstm[['norm_dist']]\
    = sc2.fit_transform(actual_results_lstm[['norm_dist']])
round_dict = actual_results_lstm[['id', 'round']].drop_duplicates().reset_index()

splitted_df = [x.sort_values('bib') for _, x in actual_results_lstm.groupby(['id','round'])]
splitted_df_indexes = [_ for _, x in actual_results_lstm.groupby(['id','round'])]

x = [x[all_var_names].values[:,:].astype(float)
     for x in splitted_df]
y = [x[['norm_dist']].values[:,-1].astype(float)
     for x in splitted_df]
x = [Variable(torch.Tensor(u)) for u in x]
indices = [(i,n) for i,v in enumerate(y) for n in range(len(v))]
epochs_total = 50000
batch_size = 20
train_indices, test_indices = train_test_split(indices,train_size=0.5, test_size=0.01)
selected_indices, selected_test_indices = [[uniforming_tensors(x[ind[0]][:ind[1]+1])
                                           for ind in train_indices],
                                           [uniforming_tensors(x[ind[0]][:ind[1]+1])
                                           for ind in test_indices]]
selected_targets, selected_test_targets = [[y[ind[0]][ind[1]]
                                           for ind in train_indices],
                                           [y[ind[0]][ind[1]]
                                           for ind in test_indices]]
joined_indices = [torch.stack(krotka)
                  for krotka in zip(*[selected_indices[i::batch_size]
                                      for i in range(batch_size)])]
joined_targets = [krotka
                  for krotka in zip(*[selected_targets[i::batch_size]
                                     for i in range(batch_size)])]
joined_test_indices = [torch.stack(krotka)
                       for krotka in zip(*[selected_test_indices[i::batch_size]
                                           for i in range(batch_size)])]
joined_test_targets = [krotka
                       for krotka in zip(*[selected_test_targets[i::batch_size]
                                           for i in range(batch_size)])]

"""
def find_ten_last_comps(comp, dict_indices, ratings):
    new_dict_indices = dict_indices.copy()
    codexes = ratings.loc[comp['index']:comp['next_index']]['codex']
    for index in codexes.index:
        actual_value = new_dict_indices[codexes.loc[index]]
        if len(actual_value) < 11:
            new_dict_indices.update({codexes.loc[index]:
                                     actual_value + [index]})
        else:
            new_dict_indices.update({codexes.loc[index]:
                                     actual_value[1:] + [index]})
    return new_dict_indices

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)][:, 1:]
        _y = data[i:(i+seq_length)][:, :1]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)
dict_groups = actual_ratings.groupby(['id', 'round']).groups
dict_groups = actual_ratings.groupby(['codex']).groups
dict_indices = actual_ratings.groupby('codex').indices
round_dict = actual_ratings[['id', 'round']].drop_duplicates().reset_index()
round_dict['next_index'] = round_dict['index'].shift(-1,
                                                     fill_value=
                                                     len(actual_ratings))
round_dict_2 = {(x[1],x[2]): (x[0],x[3]) for x in round_dict.values}
results_dict = actual_results[['id', 'round']].drop_duplicates().reset_index()
results_dict['next_index'] = results_dict['index'].shift(-1,
                                                     fill_value=
                                                     len(actual_results))
results_dict_2 = {(x[1],x[2]): (x[0],x[3]) for x in results_dict.values}

ratings_dict = actual_ratings.to_dict()

M = {}
new_dict_indices = dict.fromkeys(dict_indices.keys(), [])

filtered_comps = find_ten_last_comps(round_dict.loc[3540], new_dict_indices,
                                     actual_ratings)

for comp in round_dict.iterrows():
    new_dict_indices = find_ten_last_comps(comp[1], new_dict_indices,
                                           actual_ratings)
    M[(comp[1]['id'], comp[1]['round'])] = new_dict_indices

%%timeit 
np_lines = [read_ratings(data[1], ratings_dict, M)
            for data in actual_ratings.loc[1000:2000].iterrows()]
%%timeit
new_data = actual_ratings.loc[1000:20000].apply(read_ratings, axis = 1,
                                ratings_dict = ratings_dict,
                                last_ratings = M)

#sc = MinMaxScaler()
#training_data = sc.fit_transform(actual_results_lstm)

x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
"""

class LSTMRegressor(nn.Module):
    def __init__(self, batch_size,
                 hidden_size = 100, variable_size = 28):
        super(LSTMRegressor, self).__init__()
        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        
        """
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.variable_size = variable_size
        
        self.lstm = nn.LSTM(variable_size, hidden_size)
        self.drop = nn.Dropout(p=0.3)
        self.label = nn.Linear(hidden_size, 1)
        
    def forward(self, input_sentence, batch_size=None):
        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        
        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)
        
        """
        
        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        input = input_sentence # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)) # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)) # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_hidden_state = self.drop(final_hidden_state)
        final_output = self.label(final_hidden_state) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        return final_output

model = LSTMRegressor(batch_size = batch_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
seed = 0
torch.manual_seed(seed)
losses=[]
# Train the model
for epoch in range(len(joined_indices)):
    optimizer.zero_grad()
    x_pred = joined_indices[epoch].detach().clone()
    y_pred = model(x_pred)
    y_true = joined_targets[epoch]
    if(epoch == epochs_total):
        print(x_pred)
        print(epoch)
        print(y_pred)
        print(y_true)
        print(loss_function(y_pred, y_true))
    single_loss = loss_function(y_pred, torch.FloatTensor([y_true]).unsqueeze(2))
    single_loss.backward()
    losses = losses + [single_loss.item()]
    optimizer.step()
    if epoch%50 == 2:
        print(f'epoch: {epoch:3} loss: {st.mean(losses):10.8f}')

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

plot(moving_average(np.log10(np.array(losses)),100))


# Test the model
test_epoch=0
outputs_collect=[]
hill_sizes = []
comps = []
# Test the model
model.eval()
for n in range(batch_size):
    index = test_indices[n]
    comp = splitted_df[index[0]].iloc[index[1]]
    print(comp)
    hill_sizes = hill_sizes+[actual_comps[actual_comps['id'] == comp['id']]['hill_size_x'].item()]
    comps = comps + [comp]

index = test_indices[test_epoch]
with torch.no_grad():
    x_pred = joined_test_indices[test_epoch].detach().clone()
    y_pred = model(x_pred)
    y_pred = sc2.inverse_transform(y_pred)[0,:,0]
    y_true = sc2.inverse_transform(joined_test_targets[test_epoch])
    outputs = np.append([np.array(y_pred)*np.array(hill_sizes)],
                        [np.array(y_true)*np.array(hill_sizes)],axis=0)




"""
print(df)
    scatter(df[:,0],df[:,1])
    X = np.linspace(np.amin(df),np.amax(df))
    plot(X,X)
    plt.show()
    plt.close()
    r_sq = r2_score(df[:,1],df[:,0])
    print(code)
    print('The R squared is equal to: {}'.format(r_sq))
"""