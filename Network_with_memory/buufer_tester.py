import dataExtractor as de
import torch

a = de.DataAdjust('data/test_data.csv',drop=False)

test = de.TrajDataSet(a.get_data_Frame(),mem_nb=3)
test_loader = torch.utils.data.DataLoader(test,batch_size=1,shuffle=True)
state = next(iter(test_loader))
exemple = test[10]
print(f'\n---------------\n Voici notre état \n {state} \n ---------------')
print(f'Voici ce que nous renvoie le 10 élément de test \n {exemple} \n ---------------')
print(f'Le contenu de test avant le passage à torch : \n {test.get_traj()[10:13]} \n ---------------')
