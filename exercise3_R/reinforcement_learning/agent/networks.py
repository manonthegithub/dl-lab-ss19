import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""

class MLP(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=400):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)



class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=3):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        # self.cnn1 = torch.nn.Conv2d(in_channels=history_length, out_channels=history_length, kernel_size=4)
        # self.sbs1 = torch.nn.AvgPool2d(kernel_size=4)
        # self.lstm = torch.nn.LSTM(input_size=23*23, hidden_size=10*10, batch_first=True)
        # self.sbs2 = torch.nn.AvgPool2d(kernel_size=2)
        # self.linear = torch.nn.Linear(in_features=5*5, out_features=n_classes)
        self.cnn1 = torch.nn.Conv2d(in_channels=history_length, out_channels=history_length, kernel_size=3)
        self.sbs1 = torch.nn.MaxPool2d(kernel_size=2)
        self.lstm = torch.nn.LSTM(input_size=47*47, hidden_size=15*15, batch_first=True)
        self.sbs2 = torch.nn.AvgPool2d(kernel_size=2)
        self.linear = torch.nn.Linear(in_features=7*7, out_features=n_classes)


    def forward(self, x):
        # TODO: compute forward pass
        x = self.cnn1(x)
        x = self.sbs1(x)
        x, _ = self.lstm(x.view(x.shape[0], x.shape[1], -1))
        # x = self.sbs2(x.view(x.shape[0], x.shape[1], 10, 10))
        x = self.sbs2(x.view(x.shape[0], x.shape[1], 15, 15))
        x = self.linear(x.view(x.shape[0], x.shape[1], -1))
        return x



