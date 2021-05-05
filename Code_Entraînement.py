###############################   IMPORTATIONS   ################################

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Code_Reseau import BehavioralCloning
from dataExtractor import TrajDataSet, DataAdjust

##########   INITIALISATION OF THE DATASETS AND THE DATALOADERS   ############### 

frame = DataAdjust("trips_SV_2008_2015.csv") 
train_d, test_d = frame.subset_data(45)
std1 = [train_d.lon.std(),train_d.lat.std()]
std2 = [test_d.lon.std(),test_d.lat.std()]
col_coord = [-77.264,-11.773]
#! Export in csv with pd.datafile.to_csv
train_set = TrajDataSet(train_d,torchvision.transforms.Normalize(col_coord,std1)) # Creation of the train set
test_set = TrajDataSet(test_d,torchvision.transforms.Normalize(col_coord,std2)) # Creation of the test 

train_loader = torch.utils.data.DataLoader(train_set,batch_size=16,shuffle=True) # Creation of the train loader
test_loader = torch.utils.data.DataLoader(test_set,batch_size=16,shuffle=True) # Creation of the test loader

####################   INITIALISATION OF THE VARIABLES   ########################## 

def perso_export():
    return test_d

state_dim = 2600
action_dim = 2600
learning_parameter = 0.00001 # We want to keep it small to prevent gradient explosions
epochs = 3 # Number of episodes
model = BehavioralCloning(state_dim, action_dim) # Importation of the network
criterion = nn.MSELoss() # Here we choose a Mean Squared Error to compute our loss
optimizer = torch.optim.SGD(model.parameters(), learning_parameter) # We use the Stochastic Gradient Descent from PyTorch to optimize our network

#####################   TRAINING OF OUR NEURAL NETWORK   #########################

history = [] # This list will store all of the losses
train_losses = [] # This one will store the losses of each training epoch
test_losses = [] # This one will store the losses of the testing epochs, every ten training epoch
for epoch in range(epochs):
    print(f'We are at epoch {epoch}')
    if epoch%5 == 0: # Every ten training epoch, we look the behavior of our network on the testing loader
        model.eval()
        with torch.no_grad():
            for batch,(state,action) in enumerate(test_loader):
                inputs = Variable(state.float())
                labels = Variable(action.float())
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # We could add a condition here to prevent from overfitting
                history.append(loss.item())
                # There is no optimisation here, we only look the behavior
            test_losses.append(loss.item())
    else :
        model.train()
        for batch,(state,action) in enumerate(train_loader):
            print(f'The current state is {state}.')
            inputs = Variable(state.float())
            labels = Variable(action.float())
            optimizer.zero_grad()
            outputs = model(inputs)
            # Caution : gradients may be exploding because of two potential things : a too high l.p. or unnormalized inputs
            loss = criterion(outputs, labels)
            history.append(loss.item())
            loss.backward()
            optimizer.step()
        train_losses.append(loss.item())

################################   RESULTS   ######################################

# Printing the model parameters
print(model.parameters())
print(test_losses)
#print(w)
#print(b)
# Printing the evolution of the loss
plt.clf()
x_train = [i for i in range(len(train_losses))]
y_train = [loss for loss in train_losses]
x_test = [i for i in range(len(test_losses))]
y_test = [loss for loss in test_losses]
plt.subplot(211)
plt.plot(x_train,y_train)
plt.ylabel("Evolution of the train loss")
plt.subplot(212)
plt.plot(x_test,y_test)
plt.ylabel("Evolution of the test loss")
plt.xlabel("epoch")
plt.show()
plt.savefig("Loss_Evolution.png")

#################   OTHER SOLUTION FOR THIS LINEAR REGRESSION   #################

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

