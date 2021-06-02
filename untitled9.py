# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:35:05 2021

@author: kubaf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(1)
window = 12
epochs = 150

scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit(Alcohol_Salescsv[:150,1].reshape(-1,1))
data = scaler.transform(Alcohol_Salescsv[:,1].reshape(-1,1))

lstm = nn.LSTM(1, 100)  # Input dim is 3, output dim is 3
inputs = [torch.Tensor(data[l:l+12].astype(np.float))
          for l in range(len(data)-13)]  # make a sequence of length 5
outputs = [torch.Tensor(data[l+12].astype(np.float))
          for l in range(len(data)-13)]  # make a sequence of length 5
"""
# initialize the hidden state.
hidden = (torch.randn(1, 1, 100),
          torch.randn(1, 1, 100))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 1), torch.randn(1, 1, 1))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)
"""
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=10, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

processed_data = [[inputs[i],outputs[i]] for i in range(len(inputs))]
for i in range(epochs):
    for seq, labels in processed_data:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%10 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

test_data = processed_data[epochs:]
test_epochs = 50
for i in range(test_epochs):
    for seq, labels in test_data:
        # optimizer.nograd()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        print(y_pred, labels)
        single_loss = loss_function(y_pred, labels)
        # single_loss.backward()
        # optimizer.step()
    if i % 10 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')



