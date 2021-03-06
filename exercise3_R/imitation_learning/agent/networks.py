import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=3):
        super(CNN, self).__init__()
        self.lstm_hiden = 15
        # TODO : define layers of a convolutional neural network
        self.cnn1 = torch.nn.Conv2d(in_channels=history_length, out_channels=history_length, kernel_size=3)
        self.sbs1 = torch.nn.MaxPool2d(kernel_size=2)
        self.lstm = torch.nn.LSTM(input_size=47 * 47, hidden_size=self.lstm_hiden ** 2, batch_first=True)
        self.sbs2 = torch.nn.AvgPool2d(kernel_size=2)
        self.linear = torch.nn.Linear(in_features=7 * 7, out_features=n_classes)


    def forward(self, x):
        # TODO: compute forward pass
        x = self.cnn1(x)
        x = self.sbs1(x)
        x, _ = self.lstm(x.view(x.shape[0], x.shape[1], -1))
        # x = self.sbs2(x.view(x.shape[0], x.shape[1], 10, 10))
        x = self.sbs2(x.view(x.shape[0], x.shape[1], self.lstm_hiden,self.lstm_hiden))
        x = self.linear(x.view(x.shape[0], x.shape[1], -1))
        return x

