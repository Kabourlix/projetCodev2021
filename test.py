import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import dataExtractor as extr
import Code_Reseau as network

def recenter(array):
	"""
	This function recenters our coordinate around (0,0) in order to compare data.
	We calculate the barycenter of our data b and translate all our point with it. 

	Parameters
	----------
	array : np.array
		a list of coordinate. We must have array.shape = (.,2)

	Returns
	-------
	adjusted_traj : np.array
		return the (.,2) array centered on (0,0) (via the barycenter meaning).

	"""
	b = np.array([np.mean(array[:,0]),np.mean(array[:,1])])
	return (array-b).copy()

data = extr.DataAdjust('trips_SV_2008_2015.csv')

"""
The aim is to test our network by outputing a trajectory.
We provide an initial coordonate and then we get he next move for x iterations 
where x is the lenght of the expert trajectory associated
"""
def get_forseen_traj(personnal_data=data):
	
	#Importing one random trajectory to plot it
	traj_data = personnal_data.select_random_traj() #TODO : Modify this line later when it would be edited in dataExtractor.py 
	traj_dataset = extr.TrajDataSet(traj_data)
	traj_data_loader = torch.utils.data.DataLoader(traj_dataset,batch_size=1,shuffle=False)
	
	#Parameters of the simulation
	
	state,action = next(iter(traj_data_loader)) #Here we got our tensors. inutile puisque seulement dans la boucle non ?
	state_dim = len(state) #! Here state and action are (2,1) it is juste coordinates : must be checked. 
	action_dim = len(action)
	model = network.BehavioralCloning(state_dim,action_dim)
	time_step = traj_dataset.traj.shape[0]
	
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
	return np.array(trajectory),traj_dataset.traj[:,0:2]



################ Printing trajectories ######################################
def plot_traj():
	traj, expert_traj = get_forseen_traj(data)
	
	adjusted_trained_traj = recenter(traj)
	adjusted_expert_traj = recenter(expert_traj)
	
	plt.subplot(211)
	plt.plot(adjusted_trained_traj[:,0],adjusted_trained_traj[:,1])
	
	plt.ylabel("y_trained")
	plt.subplot(212)
	plt.plot(adjusted_expert_traj[:,0],adjusted_expert_traj[:,1],color='red') #! It prints strange stuff : it doesn't seem to work properly
	plt.xlabel("x")
	plt.ylabel("y_expert")
	plt.savefig("trained_traj.png")
	plt.show()
#############################################################################