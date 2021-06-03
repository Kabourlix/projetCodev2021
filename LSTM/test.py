################################   IMPORTATIONS   #################################

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import dataExtractor as extr
import Code_Reseau as network

def unormalized(traj,data_obj,label):
    """
        Traj is a (n,2) np.ndarray. 
        we output a denormalized traj
    """
    traj[:,0] = (traj[:,0]+col_coord[0])*data_ob.td_df.loc[label,'lon']
    traj[:,1] = (traj[:,1]+col_coord[1])*data_ob.td_df.loc[label,'lat']


def action_to_state(ac_tensor,current_state):
    """
        This function is aimed at transforming an action in a state to put inside the model. 
        Parameters :
            - ac_tensor : torch.tensor of shape [nb_mem,2], this is an action provided by the model 
            - current_state : torch.tensor of shape [nb_mem,2], this the state related to the previous action. Where the action is made.
        Returns :
            - next_state : np.array of shape (nb_mem,2). This is the state one step forward. It need treatment before becoming a new model input. 
    """
    ac_np = ac_tensor.detach().numpy() #We get [r,theta]
    next_state = current_state.numpy() #Prepare the next step for manipulation
    #For lisibility, we write the exponential form ouf our action's velocity vector
    #print(f' action : {ac_np}')
    r = ac_np[:,0]
    r = r.reshape((len(r),1))
    theta = ac_np[:,1]
    theta = theta.reshape((len(theta),1))
    ######################
    #print(f'On a r : \n {r} \n et également theta : \n {theta} \n En outre, shape de theta {theta.shape}.')
    #print(np.concatenate((np.cos(theta),np.sin(theta)),axis=1))
    mult = r*np.concatenate((np.cos(theta),np.sin(theta)),axis=1) # [r cos(theta),r sin(theta)], real and imaginary part ouf our velocity (R + i I)
    #! Problem with the shape given
    next_state += mult #Calculate next state.
    return next_state

#We import the test data Frame. 
personnal_data = extr.DataAdjust('data/test_data.csv',drop=False)

"""
The aim is to test our network by outputing a trajectory.
We provide an initial coordonate and then we get he next move for x iterations 
where x is the lenght of the expert trajectory associated
"""
#Importing one random trajectory to plot it
col_coord = [-77.264,-11.773]

traj_data,rd_data = personnal_data.select_random_traj()  #Random traj
#traj_data,rd_label = personnal_data.select_random_traj()  #Random traj
#expert_traj = personnal_data.unormalized(rd_label) #We get the random traj without normalization for plotting down below.
# ! I must check this. There is still some errors.  
traj_dataset = extr.TrajDataSet(traj_data,mem_nb = 10)
traj_data_loader = torch.utils.data.DataLoader(traj_dataset,batch_size=1,shuffle=False)



######################   INITIALISATION OF THE VARIABLES   #################################

state,action = next(iter(traj_data_loader)) #Here we got our tensors. inutile puisque seulement dans la boucle non ?

model = torch.load('FirstNetwork/models/linear_Memory.pt')
model.eval()

time_step = traj_dataset.traj.shape[0]
#print(f'Nombre d éléments dans la traj : {time_step}')
trajectory = state.numpy()[0] #It got the initial state (The three first one)
#print(f'Trajectory : {trajectory}')
####################################   LOOP   ######################################

for iteration in range(time_step-1):
    next_action = model.forward_for_plot(state.float()) # Get the next action : here it is torch.tensor([a_i,a_{i+1},a_{i+2}]) where a_j = [.,.]
    next_state = action_to_state(next_action,state)
    #print(f'Next_state[0][-1] vaut {next_state[0][-1]}')
    trajectory = np.append(trajectory,next_state[0][-1].copy()) #Add it to the trajectory list, only the last one since the other are already in it.
    trajectory = trajectory.reshape((len(trajectory)//2,2)) 
    #print(f'Next_state[0][-1] vaut {next_state[0][-1]}')
    state = torch.from_numpy(next_state) # Modify state value for next loop



###########################   PRINTING TRAJECTORIES   ######################################
print(f'Après l_algo, voilà la traj {trajectory}')
trajectory = np.array(trajectory)
#unormalized(trajectory,personnal_data,rd_data)
expert_traj = traj_dataset.traj[:,0:2]
print(f'Expert traj : {expert_traj}')
plt.subplot(211)
plt.plot(trajectory[:,0],trajectory[:,1])
plt.scatter(col_coord[0],col_coord[1],color = 'red')

plt.ylabel("y_trained")
plt.subplot(212)
plt.plot(expert_traj[:,0],expert_traj[:,1],color='red')
plt.scatter(col_coord[0],col_coord[1],color = 'green')
plt.xlabel("x")
plt.ylabel("y_expert")
plt.savefig("FirstNetwork/img/trained_traj_1.png") #L'indice 1 se réfère au premier réseau. 
plt.show()

#Rajouter un plt.scatter

plt.figure()
plt.plot(trajectory[:,0],trajectory[:,1])
plt.scatter(col_coord[0],col_coord[1],color = 'red')
plt.show()

#############################################################################