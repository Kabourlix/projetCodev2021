import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

train = pd.read_csv('train_losses.csv').to_numpy()
test = pd.read_csv('test_losses.csv').to_numpy()
min_train = []
min_test = []

for k in range(20):
    min_train.append(min(train[16*k:15*(k+1)+k,1]))
    min_test.append(min(test[16*k:15*(k+1)+k,1]))

x = [k for k in range(20)]
plt.plot(x,min_train)
plt.show()

print(f'Moyenne train : {np.mean(np.array(min_train))}')
print(f'Moyenne test : {np.mean(np.array(min_test))}')
