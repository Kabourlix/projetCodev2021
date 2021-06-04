import analysis_of_losses as stat
import pandas as pd

if __name__ == "__main__":
    stat_train = stat.Loss_stat(filename='data/train_losses_Linear_NoMemory.csv',model_name='LinearNoMem',nb_epoch=5) #The default constructor deals with train
    stat_train.epoch_stat()
    stat_train.plot_loss()