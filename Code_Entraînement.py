import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Code_Reseau import BehavioralCloning
from dataExtractor import TrajDataSet

#Importation des données
set = TrajDataSet("trips_SV_2008_2015.csv")
loader = torch.utils.data.DataLoader(set,batch_size=16,shuffle=True)
a,b,c,d = next(iter(loader))
states = (a,b)
exp_actions = (c,d)

# Initialisation des variables
learning_parameter = 0.01
epochs = 150
model = BehavioralCloning(set.__len__(),set.__len__())
criterion = nn.MSELoss()
optimizer = optimizer = torch.optim.SGD(model.parameters(), learning_parameter)

# Boucle d'entraînement
history = []
for epoch in range(epochs):
    inputs = Variable(torch.from_numpy(states))
    labels = Variable(torch.from_numpy(exp_actions))
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
    predicted = model(Variable(torch.from_numpy(states))).data.numpy()
    print(predicted)

plt.clf()
plt.plot(states, exp_actions, 'go', label = 'True data', alpha =0.5)
plt.plot(states, predicted, '--', label = 'Predictions', alpha = 0.5)
plt.legend(loc='best')
plt.show()



#En dessous : autre solution pour la régression mais moins compréhensible

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

