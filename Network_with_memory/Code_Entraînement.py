###############################   IMPORTATIONS   ################################
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Code_Reseau import BehavioralCloning
from dataExtractor import TrajDataSet, DataAdjust

##########   INITIALISATION OF THE DATASETS AND THE DATALOADERS   ############### 

#frame = DataAdjust("trips_SV_2008_2015.csv") 
#train_d, test_d = frame.subset_data(45)
#Importing data
train_d = DataAdjust("data/train_data_memory.csv",drop=False)
test_d = DataAdjust("data/test_data_memory.csv",drop=False)

memory = 10
batch = 16

#Transform them in Data Set
train_set = TrajDataSet(train_d.get_data_Frame(),mem_nb=memory) # Creation of the train set
test_set = TrajDataSet(test_d.get_data_Frame(),mem_nb=memory) # Creation of the test 

#Make them dataLoader
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch,shuffle=True) # Creation of the train loader
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch,shuffle=True) # Creation of the test loader

state,action = next(iter(train_loader))

print(f'The first state is {state} of size {state.size()}.')
b = state.resize(1,2*batch*memory)
print(f'THe resized is {b} of size {b.size()}.')
####################   INITIALISATION OF THE VARIABLES   ########################## 

learning_parameter = 0.00001 # We want to keep it small to prevent gradient explosions
epochs = 20 # Number of episodes
model = BehavioralCloning(memory,batch) # Importation of the network
criterion = nn.MSELoss() # Here we choose a Mean Squared Error to compute our loss
optimizer = torch.optim.SGD(model.parameters(), learning_parameter) # We use the Stochastic Gradient Descent from PyTorch to optimize our network
nb_evaluation = 1 # This parameter choses the number of evaluations we want for n epochs

#####################   TRAINING OF OUR NEURAL NETWORK   #########################

history = [] # This list will store all of the losses
train_losses = [] # This one will store the losses of each training epoch
test_losses = [] # This one will store the losses of the testing epochs, every ten training epoch
for epoch in range(epochs):
    print(f'We are at epoch {epoch}')
    # Here we train our network 
    model.train() # instruction to train
    for r_batch,(state,action) in enumerate(train_loader):
        if r_batch == batch:
            inputs = Variable(state.float())
            #print(f'Voici nos inputs {inputs}')
            labels = Variable(action.float())
            optimizer.zero_grad()
            outputs = model(inputs )
            # Caution : gradients may be exploding because of two potential things : a too high l.p. or unnormalized inputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # Here we test our network with the train loader and the test loader, apart from the training part
    if epoch%nb_evaluation == 0: # Every nb_evaluation epoch, we look the behavior of our network on the two loaders
        model.eval()
        with torch.no_grad():
            # Testing on the train loader
            for r_batch,(state,action) in enumerate(train_loader):
                if r_batch == batch:
                    inputs = Variable(state.float())
                    labels = Variable(action.float())
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    history.append(loss.item())
            train_losses.append(loss.item())
            # Testing on the test loader
            for r_batch,(state,action) in enumerate(test_loader):
                if r_batch == batch:
                    inputs = Variable(state.float())
                    labels = Variable(action.float())
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    history.append(loss.item())
            test_losses.append(loss.item())
    if (train_losses[-1] < 100):
        break


########### Saving the model ##################
torch.save(model,'models/linear_Memory.pt') 
###############################################


################################   RESULTS   ######################################

# Printing the model parameters
print(model.parameters())
print(test_losses)
# Printing the evolution of the loss
plt.clf()
x = [i for i in range(len(train_losses))]
y_train = [loss for loss in train_losses]
y_test = [loss for loss in test_losses]
plt.plot(x,y_train, color="red",label="Train loss")
plt.plot(x,y_test, color="blue",label="Test loss")
plt.legend()
plt.xlabel("Epoch")
plt.title("Evolution of our losses")
plt.savefig("img/Loss_Evolution_linear_memory.png")
plt.show()

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

