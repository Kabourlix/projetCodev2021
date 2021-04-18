import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import dataExtractor as extr
import Code_Reseau as network

personnal_data = extr.DataAdjust('trips_SV_2008_2015.csv')

"""
The aim is to test our network by outputing a trajectory.
We provide an initial coordonate and then we get he next move for x iterations 
where x is the lenght of the expert trajectory associated
"""
#Importing one random trajectory to plot it
traj_data = personnal_data.select_random_traj()
traj_dataset = extr.TrajDataSet(traj_data)
traj_data_loader = torch.utils.data.DataLoader(traj_dataset,batch_size=1,shuffle=False)

#Parameters of the simulation

state,action = next(iter(traj_data_loader)) #Here we got our tensors. inutile puisque seulement dans la boucle non ?
state_dim = len(state)
action_dim = len(action)
model = network.BehavioralCloning(state_dim,action_dim)
time_step = 400 #à modifier (longueur de la trajectoire, nb de pas)

trajectory = [state.numpy()[0]] #It got the initial state

for iteration in range(time_step-1):
    next_action = model.forward(state.float()) # Get the next action
    ac_np = next_action.detach().numpy() # [r,theta] Transform to numpy for manipulation
    next_state = state.numpy() #Prepare the next state in numpy for manipulation
    #For lisibility
    r = ac_np[0][0]
    theta = ac_np[0][1]
    ######################
    mult = r*np.array([np.cos(theta),np.sin(theta)]) # [r cos(theta),r sin(theta)]
    next_state += mult #Calculate next state. 
    trajectory.append(next_state[0].copy()) #Add it to the trajectory list
    state = torch.from_numpy(next_state) # Modify state value for next loop

trajectory = np.array(trajectory)
plt.plot(trajectory[:,0],trajectory[:,1])



##############################

