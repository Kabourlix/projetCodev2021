import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np 

class TrajDataSet(torch.utils.data.DataSet):
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
		self.traj = np.array([[data.lon,data.lat,data.step_speed,data.step_direction])
		#Je ne suis pas très sûr des données à mettre dedans, on va tester avec ça.  


	def __len__():
		return len(self.traj)

	def __getitem__(idx):
		"""
		This function return the idx-th pairs state/action of the array as a tensor. 
		"""
		return torch.from_numpy(self.traj[idx])


#Création du dataset
#data = pd.read_csv("trips_SV_2008_2015.csv")
#traj_data = []
#traj_idx = data.trip[0]
#i=0
#while i<len(data) and data.trip[i]== traj_idx:
#    traj_data.append([data.lon[i],data.lat[i],data.step_speed[i],data.step_direction[i]])
#    i+=1
#traj_data_np = np.array(traj_data)


