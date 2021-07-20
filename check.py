import mrcfile
import os
orig = 'data3_SNRinfinity'
mine = 'myzData10'
d = f'/home/myz/{mine}/subtomogram_mrc'
for i in range(len(os.listdir(d))):
    mrc = mrcfile.open(os.path.join(d,f'tomotarget{i}.mrc'), mode='r+', permissive=True)
    a = mrc.data
    print(f'{a.shape}, {i}')
