import torch
import pandas as pd
import numpy as np 

class TrajDataSet(torch.utils.data.Dataset):
	"""
	This class create our dataSet for our network. We will then create a DataLoader via this dataset so as to 
	train our network.
	"""
	def __init__(self,file_name,transform=None):
		"""
		It is the constructor of our class. 
		
		traj is a set of the following form [(s0,a0),(s1,a1),...,(sn,an)] with si a position and ai a speed vector.

		"""
		data = pd.read_csv(file_name)
		self.traj = np.array([data.lon,data.lat,data.step_speed,data.step_direction]).transpose()

	def __len__(self):
		return len(self.traj)

	def __getitem__(self,idx):
		"""
		This function return the idx-th pairs state/action of the array as a tensor. 
		"""
		return (torch.from_numpy(self.traj[idx][:2]),torch.from_numpy(self.traj[idx][2:])) #We output a tuple of tensor (state,action)
		#I've got an unexpected error where torch.from_numpy is not recognized by my python interpreter (in VisualCode), don't know why. 

######## Séparation du jeu de données #########
data = TrajDataSet('trips_SV_2008_2015') #Import dataset

#Create trip labels
trip_label = data['trip']
trip_label.drop_duplicates(keep='first',inplace=True)
nb_label = trip_label.shape()[0]

def select_random_traj(dataSet, )
 



#Création du dataset
#data = pd.read_csv("trips_SV_2008_2015.csv")
#traj_data = []
#traj_idx = data.trip[0]
#i=0
#while i<len(data) and data.trip[i]== traj_idx:
#    traj_data.append([data.lon[i],data.lat[i],data.step_speed[i],data.step_direction[i]])
#    i+=1
#traj_data_np = np.array(traj_data)


