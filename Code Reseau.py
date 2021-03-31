import pytorch
from torch.autograd import Variable
import torch.nn as nn
class LinearRegression(nn.Module):
    def __init__(self,state_dim,action_dim):
    super(LinearRegression,self).__init__()
    self.linear = nn.Linear(state_dim,action_dim)

    def forward(self,x):
    return self.linear(x)

    model = LinearRegression(sate_dim, action_dim)
    criterion = nn.MSELoss()
    learning_parameter = 0.01
    n_iter = 1001