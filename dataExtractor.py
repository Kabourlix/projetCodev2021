import torch
import pandas as pd
import numpy as np 

class TrajDataSet(torch.utils.data.Dataset):
	"""
	This class create our dataSet for our network. We will then create a DataLoader via this dataset so as to 
	train our network.
	"""
	def __init__(self,data,transform=None):
		"""
		It is the constructor of our class. 
		Inputs : 
			- data : DataFrame from pandas containing lon,lat,step_speed and step_direction. 
		traj is a set of the following form [(s0,a0),(s1,a1),...,(sn,an)] with si a position and ai a speed vector.

		"""
		self.traj = np.array([data.lon,data.lat,data.step_speed,data.step_direction]).transpose()

	def __len__(self):
		return len(self.traj)

	def __getitem__(self,idx):
		"""
		This function return the idx-th pairs state/action of the array as a tensor. 
		"""
		return (torch.from_numpy(self.traj[idx][:2]),torch.from_numpy(self.traj[idx][2:])) #We output a tuple of tensor (state,action)
		#I've got an unexpected error where torch.from_numpy is not recognized by my python interpreter (in VisualCode), don't know why. 

class DataAdjust():
	"""
	This class enables to deal with our data. 
	"""
	def __init__(self,file_name,label_name = 'trip',drop_label = ['dive','prediction'],method=pd.read_csv):
		#Import the data in dataFrame.
		self.data = method(file_name) #Init our data frame.
		self.data.drop(drop_label,axis=1,inplace=True) #We delete the useless columns for our work. 
		
		#Label series (in our context it is the trips' names.)
		self.label = self.data.loc[:,label_name].copy() #Init our label serie
		self.label.drop_duplicates(keep='first',inplace=True) #Eliminate copies. 
		self.nb_label = self.label.shape[0] #Ammount of labels
		
		#We make the label name an attribute of our class to make it more readable. 
		self.label_name = label_name
		
	def temp(self):
		return self.data

	def select_random_traj(self): #! Problem : we want to select a random trajectory from the untrained data
		"""
		Returns
		-------
		pandas.dataFrame
			This returns a dataFrame form of a unique trajectory

		"""
		return self.data[self.data[self.label_name] == self.label.sample().iloc[0]]
		#return selected_data[selected_data[self.label_name] == self.label.sample().iloc[0]]
		#TODO : Adjust this so as to make the selected traj in the test one. 
	
	def subset_data(self,first_ammount):
		"""
		Parameters
		-------
		first_ammount : int. This is the ammount of trajectories you want int he first dataframe to extract. 

		Returns 
		------
		data_1 : pandas.dataFrame. This is a random substract of the our dataframe containing first_ammount of trajectories.
		data_2 : pandas.dataFrame. Another one with total-first_ammount trajectories. 
		"""
		extract_labels = self.label.sample(first_ammount)
		mask = self.data[self.label_name].isin(extract_labels)
		data_1 = self.data[mask].copy()
		data_2 = self.data[~mask].copy()
		return (data_1,data_2)

personnal_data = DataAdjust('trips_SV_2008_2015.csv') #Global variable to extract once our data in a dataFrame. 

# ######## Séparation du jeu de données #########
# data = TrajDataSet('trips_SV_2008_2015') #Import dataset

# #Create trip labels
# trip_label = data['trip']
# trip_label.drop_duplicates(keep='first',inplace=True)
# nb_label = trip_label.shape()[0]

# def select_random_traj(dataSet, )

#Création du dataset
#data = pd.read_csv("trips_SV_2008_2015.csv")
#traj_data = []
#traj_idx = data.trip[0]
#i=0
#while i<len(data) and data.trip[i]== traj_idx:
#    traj_data.append([data.lon[i],data.lat[i],data.step_speed[i],data.step_direction[i]])
#    i+=1
#traj_data_np = np.array(traj_data)


