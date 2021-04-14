import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Création de la classe BehavioralCloning pour le modèle de régression linéaire
class BehavioralCloning(nn.Module):

    def __init__(self,state_dim,action_dim):
        # Notre réseau n'a qu'une seule couche linéaire
        super(BehavioralCloning,self).__init__()
        self.linear = nn.Linear(2,2) #Modif au pif : à vérifier. 

    def forward(self,x):
        # On renvoie simplement la transformation linéaire du tenseur état en entrée
        return self.linear(x)

#    def training_step(self,batch):
#        # Création d'une action et calcul de la perte associée
#        states, target_actions = batch
#        out_actions = self(states)
#        loss = nn.MSELoss(out_actions,target_actions)
#        return loss

#    def validation_step(self,batch):
#        states, target_actions = batch
#        out_actions = self(states)
#        loss = nn.MSELoss(out_actions,target_actions)
#        return {'val_loss': loss.detach()}

#    def validation_epoch_end(self,outputs):
#        batch_losses = [x['val_loss'] for x in outputs]
#        epoch_loss = torch.stack(batch_losses).mean()
#        return {'val_loss': epoch_loss.item()}

#    def epoch_end(self,epoch,result,num_epochs):
#        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
#            print('Epoch [{}], val_loss: {:.4f}'.format(epoch+1, result['val_loss']))