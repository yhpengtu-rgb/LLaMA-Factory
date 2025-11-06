import numpy as np

data = np.load('./results/100Proteins_10ps/0-th_protein_trajectory.npy', allow_pickle=True)
for a in data:
    print(a)
    
a = 1