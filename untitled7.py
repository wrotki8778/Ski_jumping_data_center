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
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import plot, scatter
import statistics as st
import random
os.chdir('C:/Users/kubaf/Documents/GitHub/Ski_jumping_data_center')

actual_results = pd.read_csv(os.getcwd()+'\\all_results.csv')
actual_results['cutted_id'] = actual_results.id.str.slice(0, 10)
actual_comps = pd.read_csv(os.getcwd()+'\\all_comps_r.csv')
actual_comps = actual_comps[actual_comps['team'] == 0]
actual_comps = actual_comps[actual_comps['training'] == 0]
actual_comps['cutted_id'] = actual_comps.id.str.slice(0, 10)
actual_ratings = pd.read_csv(os.getcwd()+'\\all_ratings.csv')
actual_ratings['cutted_id'] = actual_ratings.id.str.slice(0, 10)

actual_results_lstm = pd.merge(actual_results, actual_comps,
                               on='id', how='inner')[['dist', 'codex_x','wind','speed',
                                                     'hill_size_x', 'date',
                                                     'id', 'cutted_id_x',
                                                     'training', 'round',
                                                     'gender','bib','team',
                                                     'season']]
actual_results_lstm = actual_results_lstm[actual_results_lstm['team']==0]
actual_results_lstm = pd.merge(actual_results_lstm, actual_ratings,
                               left_on=['id', 'round', 'codex_x'],
                               right_on=['id', 'round', 'codex'],
                               how='left')[['dist', 'codex_x','wind','speed',
                                            'hill_size_x', 'date',
                                            'training', 'round',
                                            'gender', 'cumm_rating',
                                            'short_rating', 'bib',
                                            'season','id']]
actual_results_lstm = actual_results_lstm.sort_values(['date', 'round'],
                                                      ascending=[True, True])
actual_results_lstm["round"] = actual_results_lstm["round"].astype('category')
actual_results_lstm["round"] = actual_results_lstm["round"].cat.codes
actual_results_lstm["gender"] = actual_results_lstm["gender"].astype('category')
actual_results_lstm["gender"] = actual_results_lstm["gender"].cat.codes
actual_results_lstm["norm_dist"] = actual_results_lstm["dist"]/actual_results_lstm["hill_size_x"]
actual_results_lstm = actual_results_lstm[actual_results_lstm['season'] > 2010]
actual_results_lstm.loc[actual_results_lstm['codex_x']>10000,'codex_x'] = 0
actual_results_lstm = actual_results_lstm[actual_results_lstm['speed'] > 0]

actual_results_lstm = actual_results_lstm.dropna()
sc = StandardScaler()
sc2 = StandardScaler()
var_names = ['hill_size_x', 'cumm_rating', 'short_rating',
             'wind', 'speed', 'gender']
all_var_names = var_names + ['norm_dist']
input_len = len(all_var_names)
actual_results_lstm[var_names]\
    = sc.fit_transform(actual_results_lstm[var_names])
actual_results_lstm[['norm_dist']]\
    = sc2.fit_transform(actual_results_lstm[['norm_dist']])
round_dict = actual_results_lstm[['id', 'round']].drop_duplicates().reset_index()

splitted_df = [actual_results_lstm[(actual_results_lstm['id'] == x['id']) &
                                   (actual_results_lstm['round'] == x['round'])].sort_values('bib')
               for i, x in round_dict.iterrows()]
x = [x[all_var_names].values[:,:].astype(float)
     for x in splitted_df]
y = [x[['norm_dist']].values[:,-1].astype(float)
     for x in splitted_df]
x = [Variable(torch.Tensor(u)) for u in x]
y = [Variable(torch.Tensor(u)) for u in y]
"""
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)][:, 1:]
        _y = data[i:(i+seq_length)][:, :1]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

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

class LSTM(nn.Module):
    def __init__(self, input_size=input_len, hidden_layer_size=100, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)

        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.dropout = nn.Dropout(p=0.2)

        self.hidden_cell = (torch.zeros(num_layers,1,self.hidden_layer_size),
                            torch.zeros(num_layers,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.dropout(lstm_out.view(len(input_seq), -1))
        predictions = self.linear(predictions)
        return predictions[-1]

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
seed = 0
torch.manual_seed(seed)

epochs_total = 3000
max_index = 1000
losses = []
# Train the model
for epoch in range(epochs_total):
    u = random.randint(0, max_index-2)
    if len(x[u]) < 5:
        continue
    i = random.randint(2, len(x[u])-1)
    optimizer.zero_grad()
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                         torch.zeros(1, 1, model.hidden_layer_size))
    x_pred = x[u][max(i-10,0):i+1].detach().clone()
    x_pred[-1, -1] = 0
    y_pred = model(x_pred)
    if(epoch == epochs_total):
        print(x_pred)
        print(epoch)
        print(y_pred)
        print(y[u][i].data.numpy())
        print(loss_function(y_pred, y[u][i]))
    single_loss = loss_function(y_pred, y[u][i])
    single_loss.backward()
    losses = losses + [single_loss.item()]
    optimizer.step()
    if epoch%5 == 2:
        print(f'epoch: {epoch:3} loss: {st.mean(losses):10.8f}')

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

plot(moving_average(np.log10(np.array(losses)),100))


# Test the model
def r_squared(data_1,data_2):
    u = st.mean(data_2)
    total_ssq = sum((data_2-u)**2)
    model_ssq = sum((data_1-u)**2)
    return(model_ssq/total_ssq)

def predict_lstm(model, code):
    outputs_collect = []
    comp = actual_comps.loc[actual_comps['id'] == code]
    filtered_dict = round_dict.loc[round_dict['id'] == code]
    filtered_dict = [u[0] for u in filtered_dict.iterrows()]
    ha_es = comp['hill_size_x'].item()
    for instance in filtered_dict:
        for i in range(1,x[instance].shape[0]):
            with torch.no_grad():
                model.eval()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                     torch.zeros(1, 1, model.hidden_layer_size))
                x_pred = x[instance][:i+1].detach().clone()
                x_pred[-1, -1] = 0
                y_pred = model(x_pred)
                y_pred = sc2.inverse_transform([y_pred.item()])
                y_true = sc2.inverse_transform([y[instance][i].item()])
                outputs_collect = outputs_collect \
                    + [np.c_[ha_es*y_pred, ha_es*y_true]]
    df = np.array(outputs_collect).squeeze()
    f1 = plt.figure()
    scatter(df[:,0],df[:,1])
    plt.show()
    plt.close()
    r_sq = r_squared(df[:,0],df[:,1])
    print(code)
    print('The R squared is equal to: {}'.format(r_sq))
    return outputs_collect
id_test = ['2017JP3817RL','2019JP3192RL','2018JP3068RL']
model.eval()
for code in id_test:
    results = predict_lstm(model, code)


