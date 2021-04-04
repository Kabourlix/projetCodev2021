import pytorch
import torch.nn as nn
from torch.autograd import Variable
from Code Reseau import LinearRegression

def train():
    model = LinearRegression(sate_dim, action_dim)
    criterion = nn.MSELoss()
    learning_parameter = 0.01
    n_iter = 1001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)