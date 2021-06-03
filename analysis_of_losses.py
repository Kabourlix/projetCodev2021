#This is related to analysis of losses for the Newtork with sequence of position as inputs.

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def min_indice(L):
    """
        This function returns the min of a list and its index
        L is np.array
    """
    min_l = min(L)
    return [min_l,np.where(L==min_l)[0][0]]
class Loss_stat :
    """
    This class provides statistics on a a given loss list. 
    A loss list is the given loss for a certain number of epochs repeated x times.
    """
    def __init__(self,filename='train_losses.csv',list_name = 'train',model_name = 'linearNoMem',repetition = 60,nb_epoch = 15):
        loss_list = pd.read_csv(filename).to_numpy() #We read the csv and convert it to numpy.
        #Init of our attributes
        self.min_list = []
        self.rep = repetition
        self.nb_epoch = nb_epoch
        self.list_name = list_name
        self.model_name = model_name
        #########################
        #We create the min_list that will be used for stats.
        for k in range(self.rep):
            self.min_list.append(min_indice(loss_list[(self.nb_epoch+1)*k:self.nb_epoch*(k+1)+k,1]))
        self.min_list = np.array(self.min_list)
        

    def plot_loss(self):
        """
            This method plot loss calculated (x-axis unit is arbitrary)
        """
        #x = [k for k in range(self.rep)]
        loss = self.min_list[:,0]//100 #For clarity
        #plt.plot(x,self.min_list[:,0])
        plt.hist(loss,density=True)
        plt.xlabel(self.list_name + '_loss')
        plt.ylabel('Frequency')
        plt.xticks(range(12),[x*100 for x in range(12)])
        plt.title('Distribution of '+self.list_name+'_loss ('+str(self.rep)+' iterations)')
        plt.savefig('img/stats/'+self.list_name+'_lossFrequency_'+self.model_name+'.png')
        plt.show()

    def print_mean_loss(self):
        """
            This function print mean loss of the given list.
        """
        print(f'Moyenne {self.list_name} : {np.mean(np.array(self.min_list[:,0]))}')

    def epoch_stat(self):
        epochs = self.min_list[:,1] #For clarity 
        #x = [k for k in range(self.rep)]
        plt.hist(epochs,density=True)
        plt.xlabel('epoch')
        plt.ylabel('frequency')
        plt.title('Distribution of epoch where minimal '+ self.list_name + '_loss was reach ('+str(self.rep)+' iterations)')
        #plt.title('Epoch where the min '+ self.list_name +'_loss was reach.')
        plt.savefig('img/stats/'+self.list_name+'_min_epoch_'+self.model_name+'.png')
        plt.show()
        print(f"L'époque moyenne d'atteinte du minimal {self.list_name}_loss est de {np.mean(epochs)}.")


if __name__ == "__main__":
    stat_train = Loss_stat() #The default constructor deals with train
    stat_train.epoch_stat()
    stat_train.plot_loss()