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
from matplotlib.pyplot import plot
os.chdir('C:/Users/kubaf/Documents/GitHub/Ski_jumping_data_center')

actual_results = pd.read_csv(os.getcwd()+'\\all_results.csv')
actual_results['cutted_id'] = actual_results.id.str.slice(0, 10)
actual_comps = pd.read_csv(os.getcwd()+'\\all_comps_r.csv')
actual_comps['cutted_id'] = actual_comps.id.str.slice(0, 10)
actual_ratings = pd.read_csv(os.getcwd()+'\\all_ratings_r.csv')
actual_ratings['cutted_id'] = actual_ratings.id.str.slice(0, 10)

actual_results_lstm = pd.merge(actual_results, actual_comps,
                               on='id', how='left')[['dist', 'codex_x',
                                                     'hill_size_x', 'date',
                                                     'id', 'cutted_id_x',
                                                     'training', 'round',
                                                     'gender']]
actual_results_lstm = pd.merge(actual_results_lstm, actual_ratings,
                               left_on=['id', 'round', 'codex_x'],
                               right_on=['id', 'round', 'codex'],
                               how='left')[['dist', 'codex_x',
                                            'hill_size_x', 'date',
                                            'training', 'round',
                                            'gender', 'cumm_rating']]
actual_results_lstm = actual_results_lstm.sort_values(['date', 'round'],
                                                      ascending=[True, True])
actual_results_lstm["round"] = actual_results_lstm["round"].astype('category')
actual_results_lstm["round"] = actual_results_lstm["round"].cat.codes
actual_results_lstm["gender"] = actual_results_lstm["gender"].astype('category')
actual_results_lstm["gender"] = actual_results_lstm["gender"].cat.codes
actual_results_lstm["norm_dist"] = actual_results_lstm["dist"]/actual_results_lstm["hill_size_x"]
actual_results_lstm = actual_results_lstm.dropna()
sc = StandardScaler()
actual_results_lstm[['norm_dist','codex_x','hill_size_x','cumm_rating','round']]\
    = sc.fit_transform(actual_results_lstm[['norm_dist','codex_x','hill_size_x','cumm_rating','round']])

round_dict = actual_results_lstm[['date', 'round']].drop_duplicates()

splitted_df = [actual_results_lstm[(actual_results_lstm['date'] == x['date']) &
                                   (actual_results_lstm['round'] == x['round'])].sample(frac=1)
               for i, x in round_dict.iloc[1:1000].iterrows()]
x = [x.values[:,[1,2,5,6]].astype(float)
     for x in splitted_df]
y = [x[['norm_dist','codex_x']].values[:,:-1].astype(float)
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

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=False,
                            dropout=0.2)

        self.fc = nn.Linear(hidden_size,1)
    def forward(self, x, h_out):
        c_0 = Variable(torch.zeros(
            self.num_layers, 1, self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_out, c_0))
        #print(h_out)
        out = self.fc(ula)
        #print(out)
        return out, h_out
    def init_h_out(self):
        h_out = Variable(torch.zeros(
            self.num_layers, 1, self.hidden_size))
        #print(h_out.shape)
        return h_out


num_epochs = 200
learning_rate = 1e-4

input_size = 4
hidden_size = 10
num_layers = 2


lstm = LSTM(input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
# optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate, momentum=0.99)
losses = []
# Train the model
for epoch in range(num_epochs):
    h_out = lstm.init_h_out()
    for i in range(x[epoch].size(0)):
        outputs, h_out = lstm(x[epoch][i:i+1].unsqueeze(0).float(), h_out)
        #print(h_out.data.numpy())
        optimizer.zero_grad()
    # obtain the loss function
        print(epoch)
        print(outputs.data.numpy())
        print(y[epoch][i:i+1].data.numpy())
        loss = criterion(outputs.squeeze(0), y[epoch][i:i+1].float())
        loss.backward(retain_graph=True)
        losses = losses + [loss.item()]
        optimizer.step()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

plot(moving_average(np.log10(np.array(losses)),100))

# Train the model
outputs_collect = []
h_out = lstm.init_h_out()
epoch = num_epochs
for i in range(x[epoch].size(0)):
    print(h_out.data.numpy())
    outputs, h_out = lstm(x[epoch][i:i+1].unsqueeze(0).float(), h_out)
    # obtain the loss function
    print(epoch)
    print(outputs.squeeze(0).data.numpy())
    print(y[epoch][i:i+1].data.numpy())
    outputs_collect = outputs_collect \
        + [np.c_[outputs.squeeze(0).data.numpy(),
                 y[epoch][i:i+1].data.numpy()]]

train_predict = lstm(x[70][:1].unsqueeze(0).float())

data_predict = train_predict.data.numpy()
data_actual = y[70][:1].data.numpy()

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()