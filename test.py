import numpy as np 
import pandas as pd
import torch
from dataExtractor import TrajDataSet
import Code_Reseau as net


"""
The aim is to test our network by outputing a trajectory.
We provide an initial coordonate and then we get he next move for x iterations 
where x is the lenght of the expert trajectory associated
"""

d_set = TrajDataSet("trips_SV_2008_2015") #Il faudra sûrement importer cela à part dans un seul fichier au lien de 
#l'extraire dans divers fichiers : moins lourd. 

#Parameters of the simulation

state_dim,action_dim = d_set.__len__(),d_set.__len__()
time_step = 400 #à modifier


##############################

