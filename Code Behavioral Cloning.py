# Première étape : importation du dataset
import torch
from torch.utils.data import Dataset

# Création de notre classe ExpertTrajDataSet :
class ExpertTrajDataSet(Dataset):

    def __init__(self,transform=None):

