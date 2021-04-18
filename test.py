import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import dataExtractor as extr
import Code_Reseau as network


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

trajectory = [state.numpy()] #It got the initial state

for iteration in range(time_step-1):
    next_action = model.forward(state) # Get the next action
    ac_np = next_action.numpy() # [r,theta] Transform to numpy for manipulation
    next_state = state.numpy() #Prepare the next state in numpy for manipulation
    mult = ac_np[0]*[np.cos(ac_np[1]),np.sin(ac_np[1])] # [r cos(theta),r sin(theta)]
    next_state += mult #Calculate next state. 
    trajectory.append(next_state.copy()) #Add it to the trajectory list
    state = next_state.from_numpy() # Modify state value for next loop

time = [i for i in range(time_step)]

plt.plot(time,trajectory)



##############################

