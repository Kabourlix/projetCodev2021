import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Code_Reseau import BehavioralCloning
from dataExtractor import TrajDataSet

#Init. dataset and dataloader
data_set = TrajDataSet("trips_SV_2008_2015.csv")

loader = torch.utils.data.DataLoader(data_set,batch_size=16,shuffle=True)

state,action = next(iter(loader)) #Here we got our tensors.
state_dim = state.__len__()
action_dim = action.__len__()

# Initialisation des variables
learning_parameter = 0.01
epochs = 150
model = BehavioralCloning(state_dim, action_dim)
criterion = nn.MSELoss()
optimizer = optimizer = torch.optim.SGD(model.parameters(), learning_parameter)

# Boucle d'entraînement
history = []
for epoch in range(epochs):
    inputs = Variable(state)
    labels = Variable(action)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    history.append(loss.item())
    loss.backward()
    optimizer.step()

[w,b] = model.parameters()
print(w)
print(b)

# Test de la régression
with torch.no_grad():
    predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

plt.clf()
plt.plot(x_train, y_train, 'go', label = 'True data', alpha =0.5)
plt.plot(x_train, predicted, '--', label = 'Predictions', alpha = 0.5)
plt.legend(loc='best')
plt.show()

#def evaluate(model, val_loader):
#    outputs = [model.validation_step(batch) for batch in val_loader]
#    return model.validation_epoch_end(outputs)

#def fit(epochs, learning_parameter, model, train_loader, val_loader, opt_function = torch.optim.SGD)
#    history = []
#    optimizer = opt_function(model.parameters(), learning_parameter)
#    for epoch in range(epochs):
#        # Phase d'entraînement
#        for batch in train_loader:
#            loss = model.training_step(batch)
#            loss.backward()
#            optimizer.step()
#            optimizer.zero_grad()
#        # Phase de validation
#        result = evaluate(model,val_loader)
#        model.epoch_end(epoch, result, epochs)
#        history.append(result)
#    return history

