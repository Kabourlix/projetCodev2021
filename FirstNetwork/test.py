################################   IMPORTATIONS   #################################

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import dataExtractor as extr
import Code_Reseau as network
personnal_data = extr.DataAdjust('trips_SV_2008_2015.csv')

"""
The aim is to test our network by outputing a trajectory.
We provide an initial coordonate and then we get he next move for x iterations 
where x is the lenght of the expert trajectory associated
"""
#Importing one random trajectory to plot it
col_coord = [-77.264,-11.773]

traj_data = personnal_data.select_random_traj() #TODO : Modify this line later when it would be edited in dataExtractor.py 
std = [traj_data.lon.std(),traj_data.lat.std()]
traj_dataset = extr.TrajDataSet(traj_data,torchvision.transforms.Normalize(col_coord,std))
traj_data_loader = torch.utils.data.DataLoader(traj_dataset,batch_size=1,shuffle=False)


def normalize_action(action_state):
    """
    action_state = [lon,lat]
    """
    return (action_state - col_coord)/std

######################   INITIALISATION OF THE VARIABLES   #################################

state,action = next(iter(traj_data_loader)) #Here we got our tensors. inutile puisque seulement dans la boucle non ?
state_dim = len(state) #! Here state and action are (2,1) it is juste coordinates : must be checked. 
action_dim = len(action)
model = network.BehavioralCloning(state_dim,action_dim)
time_step = traj_dataset.traj.shape[0]
trajectory = [state.numpy()[0]] #It got the initial state

####################################   LOOP   ######################################

for iteration in range(time_step-1):
    next_action = model.forward(state.float()) # Get the next action
    ac_np = next_action.detach().numpy() # [r,theta] Transform to numpy for manipulation
    next_state = state.numpy() #Prepare the next state in numpy for manipulation
    #For lisibility, write the exponential form ouf our action's velocity vector
    r = ac_np[0][0]
    theta = ac_np[0][1]
    ######################
    mult = r*np.array([np.cos(theta),np.sin(theta)]) # [r cos(theta),r sin(theta)], real and imaginary part ouf our velocity (R + i I)
    next_state += mult #Calculate next state.
    #next_state = normalize_action(next_state)
    trajectory.append(next_state[0].copy()) #Add it to the trajectory list
    state = torch.from_numpy(next_state) # Modify state value for next loop

###########################   PRINTING TRAJECTORIES   ######################################

trajectory = np.array(trajectory)
expert_traj = traj_dataset.traj[:,0:2]
plt.subplot(211)
plt.plot(trajectory[:,0],trajectory[:,1])
plt.scatter(col_coord[0],col_coord[1],color = 'red')

plt.ylabel("y_trained")
plt.subplot(212)
plt.plot(expert_traj[:,0],expert_traj[:,1],color='red')
plt.scatter(col_coord[0],col_coord[1],color = 'green')
plt.xlabel("x")
plt.ylabel("y_expert")
plt.savefig("../img/firstLinearNetwork/trained_traj_1.png") #L'indice 1 se réfère au premier réseau. 

#############################################################################