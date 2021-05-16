################################   IMPORTATIONS   #################################
import torch
import pandas as pd
import numpy as np 


##########################   CREATION OF OUR DATASET   #################################

class TrajDataSet(torch.utils.data.Dataset):
	"""
	This class creates our dataSet for our network. We will then create a DataLoader via this dataset so as to 
	train our network.
	"""
	def __init__(self,data,mem_nb = 3,transform=None):
		"""
		It is the constructor of our class. 
		Inputs : 
			- data : DataFrame from pandas containing lon,lat,step_speed and step_direction. 
			- mem_nb : int. It is the ammount of points (si,ai) to use at each step of the network
		traj is a set of the following form [(s0,a0),(s1,a1),...,(sn,an)] with si a position and ai a speed vector.

		"""
		self.traj = np.array([data.lon,data.lat,data.step_speed,data.step_direction]).transpose()
		self.memory = mem_nb

	def __len__(self):
		return len(self.traj) + 1 - self.memory

	def __getitem__(self,idx):
		"""
		This function return the idx-th pairs state/action of the array as a tensor. 
		"""
		return (torch.from_numpy(self.traj[idx:idx+self.memory][:2]),torch.from_numpy(self.traj[idx:idx+self.memory][2:])) #We output a tuple of tensor (state,action)
		#I've got an unexpected error where torch.from_numpy is not recognized by my python interpreter (in VisualCode), don't know why. 

#######################   MANIPULATION OF OUR DATASET   #################################

class DataAdjust():
	"""
	This class enables to deal with our data. 
	Here is how to use it. 
	First, create an instance with your raw data : my_data = DataAdjust("my_data.csv")
	If you want to normalize your data, do this : my_data.normalize()
	For training a network, subset your data : my_data.subset_data(ammount_in_first_data,"train_data.csv","test_data.csv")
 
	Then, for training, do as follow. 
	trained_data = DataAdjust("train_data.csv")
	"""
	def __init__(self,file_name,drop = True,label_name = 'trip',drop_label = ['datetime','dive','prediction'],method=pd.read_csv):
		#Import the data in dataFrame.
		self.data = method(file_name) #Init our data frame.
		if drop:
			self.data.drop(drop_label,axis=1,inplace=True) #We delete the useless columns for our work. 
  
  
		
		################################################## Labels of the trips ###################################
		#Label series (in our context it is the trips' names)
		self.label = self.data.loc[:,label_name].copy() #Init our label serie
		self.label.drop_duplicates(keep='first',inplace=True) #Eliminate copies. 
		self.nb_label = self.label.shape[0] #Ammount of labels
		print(self.nb_label)
		
		#Access the label in the methods of our class 
		self.label_name = label_name
		###########################################################################################################
  
		self.colony = [-77.264,-11.773] #Coordinates of the colony.  

		################################ STD DATA FRAME ################################

		mask = ['lon','lat','step_speed','step_direction']
		
		#            Init our std_df and make the first series concatenation (we have particules columns label so we do it out of the loop)
		self.std_df = self.data.loc[self.data['trip'] == self.label.iloc[0],mask].std()
		temp = self.data.loc[self.data.trip == self.label.iloc[1],mask].std()
		self.std_df = pd.concat([self.std_df,temp],axis = 1)
		self.std_df.rename(columns={0 : self.label.iloc[0], 1 : self.label.iloc[1]},inplace = True)
  

		#                                 We go through the label to get the std of each trip
		for idx in range(2,self.nb_label):
			temp = self.data.loc[self.data.trip == self.label.iloc[idx],mask].std() # We get a serie with index (lon,lat,...)
			self.std_df = pd.concat([self.std_df,temp],axis=1) #We concatenate with the previous ones found, the label of the series added is 0
			self.std_df.rename(columns={0 : self.label.iloc[idx]},inplace = True) #We change its label to the trip's name associated
		
  
		self.std_df = self.std_df.swapaxes(0,1) #When the loop is over, we swap axes to have the trips in index.
		#self.std_df has the following structure
		# index  lon_std lat_std step_speed_std step_direction_std
		# trip_name ...
		#################################################################################
	def get_data_Frame(self):
		return self.data
	def temp(self):
		return self.std_df
	def select_random_traj(self): #! Problem : we want to select a random trajectory from the untrained data
		"""
		Returns
		-------
		pandas.dataFrame
			This returns a dataFrame form of a unique trajectory

		"""
		return self.data[self.data[self.label_name] == self.label.sample().iloc[0]]
		#return selected_data[selected_data[self.label_name] == self.label.sample().iloc[0]] 
  
	def normalize(self):
		"""
			This function normalizes our inputs. For a trajectory j, any coordinate [x,y] becomes [(x-colony_x)/std_j, (y-colony_y)/std_j] : ([x,y] - colony)/std_j
  		"""
		for trip_label in self.label:
			self.normalize_traj(trip_label) #We normalize each trajectory

	def normalize_traj(self,trip_name):
		"""
			This function normalize one trajectory at a time. It is done inplace.
			Parameters 
			--------
			trip_name : str. This is the label of the trajectory we want to normalize. 
        """ 
		self.data.loc[self.data.trip == trip_name,'lon'].map(lambda s : (s-self.colony[0])/self.std_df.loc[trip_name,'lon'])
		self.data.loc[self.data.trip == trip_name,'lat'].map(lambda s : (s-self.colony[1])/self.std_df.loc[trip_name,'lat'])
		self.data.loc[self.data.trip == trip_name,'step_speed'].map(lambda s : s/self.std_df.loc[trip_name,'step_speed'])
		self.data.loc[self.data.trip == trip_name,'step_direction'].map(lambda s : s/self.std_df.loc[trip_name,'step_direction'])
	
	def subset_data(self,data1_filename,data2_filename,first_ammount = 45):
		"""
		This function subset our data into two data frame. It is aimed at splitting our data in two : one for training and the other for testing.  
		!This function should save the subset dataFrame into a csv to reuse them later. 
		Parameters
		-------
		first_ammount : int. This is the ammount of trajectories you want int he first dataframe to extract. 
		data1_filename : str. It is the filename of the csv that will contain the split. 
		data2_filename : str. Same with the second dataframe. 

		Returns 
		------
		data_1 : pandas.dataFrame. This is a random substract of the our dataframe containing first_ammount of trajectories.
		data_2 : pandas.dataFrame. Another one with total-first_ammount trajectories. 
		"""
		extract_labels = self.label.sample(first_ammount)
		mask = self.data[self.label_name].isin(extract_labels)
		data_1 = self.data[mask].copy()
		data_2 = self.data[~mask].copy()

		# We save our dataFrame on the hardDrive.
		data_1.to_csv(data1_filename,index=False)
		data_2.to_csv(data2_filename,index=False)

		return (data_1,data_2)


if __name__ == '__main__':
	df = DataAdjust('data/trips_SV_2008_2015.csv')
	df.normalize() #We normalize the data
	a,b = df.subset_data("data/train_data_memory.csv", "data/test_data_memory.csv") #This save the two dataFrame in a csv we will reuse in train
	test = TrajDataSet(a,mem_nb=3)
	test_loader = torch.utils.data.DataLoader(test,batch_size=1,shuffle=True)
	state = next(iter(test_loader))
	print(f'Voici notre état {state} ainsi que sa shape.') #! On obtient pas le bon objet voulue, il faudrait regarder sur internet.
	