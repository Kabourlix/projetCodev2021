import torch
import panda as pd

class TrajDataSet():
	"""
	This class create our dataSet for our network. We will then create a DataLoader via this dataset so as to 
	train our network.
	"""
	def __init__(self,trajectory,transform=None):
		"""
		It is the constructor of our class. 
		traj is a set of the following form [(s0,a0),(s1,a1),...,(sn,an)] with si a position and ai a speed vector.
		"""
		self.array = trajectory

		#Je ne suis pas très sûr des données à mettre dedans, on va tester avec ça.  


	def __len__():
		return len(self.array)

	def __getitem__(idx):
		"""
		This function return the idx-th pairs state/action of the array as a tensor. 
		"""
		return torch.from_numpy(self.array[idx])