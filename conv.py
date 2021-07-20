import os
import json
num = 0
folder = 'snrinf/'#'inf_10/subtomogram_mrc'
mol = "3hhb"
folder = mol
for f in os.listdir(folder):
    #if(f[4:8] != "targ"):
    os.rename(f'{folder}/{f}', f'{folder}/{mol}{f}')
#for mol in os.listdir(folder):
#    if(not os.path.isdir(folder + mol)): continue
#    for f in os.listdir(folder + mol):
#        if(f[0:4] != mol): os.rename(f'{folder}{mol}/{f}', f'{folder}{mol}/{mol}{f}')
