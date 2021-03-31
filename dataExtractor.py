import torch

class TrajDataSet():
	"""
	This class create our dataSet for our network. We will then create a DataLoader via this dataset so as to 
	train our network.
	"""
	def __init__(self, traj,transform=None):
		"""
		It is the constructor of our class. 
		traj is a numpy array containing all the position of n trajectories. It is a [n x len(trajectoire)] array. 
		"""
		self.array = traj
		#Voir les autres attributs nécessaires. Cf tuto. 


	def __len__():
		return len(array)

	def __getitem__(idx):
		"""
		This function return the idx-th trajectory of the array. As a tensor. 
		"""
		return torch.from_numpy(array[idx])