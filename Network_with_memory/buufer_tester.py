import dataExtractor as de
import torch

a = de.DataAdjust('data/test_data.csv',drop=False)

test = de.TrajDataSet(a.get_data_Frame(),mem_nb=3)
test_loader = torch.utils.data.DataLoader(test,batch_size=1,shuffle=False)
#state = next(iter(test_loader))
#exemple = test[0]
#print(f'\n---------------\n Voici notre état \n {state} \n ---------------')
#print(f'Voici ce que nous renvoie le 10 élément de test \n {exemple} \n ---------------')
#print(f'Le contenu de test avant le passage à torch : \n {test.get_traj()[0:3]} \n ---------------')
#print(f'La shape de notre Tenseur renvoyé par le test_loader : state {state[0].shape} et action {state[1].shape}')

############# Test du Réseau de neurones 
print('Avant les test.')
couche = torch.nn.Linear(2, 2)

(state,action) = next(iter(test_loader))
action = couche.forward(state.float())

output = couche(state.float())
print(state.numpy()[0][0])
print('Test réussi')
