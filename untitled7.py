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
from sklearn.preprocessing import MinMaxScaler

os.chdir('C:/Users/kubaf/Documents/Skoki')

actual_results = pd.read_csv(os.getcwd()+'\\all_results.csv')
actual_comps = pd.read_csv(os.getcwd()+'\\all_comps.csv')

actual_results_lstm = pd.merge(actual_results, actual_comps,
                               on='id', how='left')[['dist', 'codex_x',
                                                     'hill_size_x', 'date',
                                                     'round']]
actual_results_lstm = actual_results_lstm.sort_values(['date', 'round'],
                                                      ascending=[True, True])
actual_results_lstm["round"] = actual_results_lstm["round"].astype('category')
actual_results_lstm["round"] = actual_results_lstm["round"].cat.codes
actual_results_lstm["norm_dist"] = actual_results_lstm["dist"]/actual_results_lstm["hill_size_x"] - 0.85
# actual_results_lstm = actual_results_lstm.drop(['date'],axis=1)
actual_results_lstm = actual_results_lstm.dropna()
# actual_results_lstm["codex_x"] = actual_results_lstm["codex_x"].astype('category')
round_dict = actual_results_lstm[['date', 'round']].drop_duplicates()

splitted_df = [actual_results_lstm[(actual_results_lstm['date'] == x['date']) &
                                   (actual_results_lstm['round'] == x['round'])]
               for i, x in round_dict.iloc[1:100].iterrows()]
x = [x[['norm_dist','codex_x']].values[:,1:]
     for x in splitted_df]
y = [x[['norm_dist','codex_x']].values[:,:1]
     for x in splitted_df]
x = [Variable(torch.tensor(u)) for u in x]
y = [Variable(torch.tensor(u)) for u in y]
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
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size,1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out


num_epochs = 1
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1


lstm = LSTM(input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(x[epoch].unsqueeze(0).float())
    optimizer.zero_grad()

    # obtain the loss function
    print(epoch)
    print(outputs.data.numpy())
    print(y[epoch].data.numpy())
    loss = criterion(outputs, y[epoch].float())

    loss.backward()

    optimizer.step()
    if epoch % 10 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

lstm.eval()
train_predict = lstm(x[70].unsqueeze(0).float())

data_predict = train_predict.data.numpy()


plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()