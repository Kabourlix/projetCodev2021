import pytorch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self,state_dim,action_dim):
    super(LinearRegression,self).__init__()
    self.linear = nn.Linear(state_dim,action_dim)

    def forward(self,x):
    return self.linear(x)