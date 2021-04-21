import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Code_Reseau import BehavioralCloning
from dataExtractor import TrajDataSet, DataAdjust


#Init. datasets and dataloaders
frame = DataAdjust("trips_SV_2008_2015.csv")
train_d, test_d = frame.subset_data(45)
train_set = TrajDataSet(train_d)
test_set = TrajDataSet(test_d)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=16,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=16,shuffle=True)

def perso_export():
    return test_d
colony_tensor = torch.from_numpy(np.array([77.264,-11.773])) #Colony coordinate to normalize data

state_dim = 2600
action_dim = 2600


# Initialisation des variables
learning_parameter = 0.0001
epochs = 100
model = BehavioralCloning(state_dim, action_dim)
criterion = nn.MSELoss()
optimizer = optimizer = torch.optim.SGD(model.parameters(), learning_parameter)

# Boucle d'entraînement
history = []
for epoch in range(epochs):
    print(f'We are at epoch {epoch}')
    if epoch%10 == 0: # On regarde le comportement du réseau sur les données de test toutes les 10 epochs
        model.eval()
        with torch.no_grad():
            for batch,(state,action) in enumerate(test_loader):
                inputs = Variable(state.float())
                labels = Variable(action.float())
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                history.append(loss.item())
                # pas d'optimisation ici
    else :
        model.train()
        for batch,(state,action) in enumerate(train_loader):
            inputs = Variable(state.float())
            labels = Variable(action.float())
            optimizer.zero_grad()
            outputs = model(inputs) #! The issue is here. 
            loss = criterion(outputs, labels)
            history.append(loss.item())
            loss.backward()
            optimizer.step()
        #print(f"inputs = {inputs} ; label = {labels} ; outputs = {outputs} ; loss = {loss}")
    #Add a condition to test the test-data-set (not used for training, only evaluation)
    # condition d'arrêt si overfitting ?

[w,b] = model.parameters()
print(w)
print(b)

# Test de la régression
# with torch.no_grad():
#     predicted = model(Variable(torch.from_numpy(state))).data.numpy()
#     print(predicted)

#plt.clf()
#plt.plot(state, action, 'go', label = 'True data', alpha =0.5)
#plt.plot(state, predicted, '--', label = 'Predictions', alpha = 0.5)
#plt.legend(loc='best')
#plt.show()

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

