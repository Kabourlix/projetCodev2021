################################   IMPORTATIONS   #################################

import torch.nn as nn

#######################   CREATION OF THE NEURAL NETWORK   ######################

class BehavioralCloning(nn.Module):

    # Definition of the network
    def __init__(self, state_dim, action_dim):
        super(BehavioralCloning,self).__init__() # Here we only use a single linear layer
        self.linear = nn.Sequential(nn.Linear(2,100),nn.Linear(100,100),nn.Linear(100,2)) # The tensor dimensions have to match


    # Choice of the operations within our network
    def forward(self,x):
        return self.linear(x) # We only return the linear transformation of the input tensor


#################   OTHER SOLUTION FOR THIS LINEAR REGRESSION   #################

#    def training_step(self,batch):
#        # Creation of an action, computing of the associated loss
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