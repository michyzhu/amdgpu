import pickle
import numpy as np
import mrcfile
def pickley(filename):
    with open(filename, 'rb') as pickle_file:
        return pickle.load(pickle_file, encoding='latin1')

def pickled(o, path,protocol = -1):
    with open(path, 'wb') as f:
        pickle.dump(o, f, protocol=protocol)

def map2mrc(map, file):
    with mrcfile.new(file, overwrite=True) as mrc:
        mrc.set_data(map.astype(np.float32))

if __name__=='__main__':
    classes = ['1I6V', '1QO1']#, '3DY4', '4V4A', '5LQW']
    x_train = pickley('10_2000_30_01.pickle')
    x_train = np.expand_dims(np.array([x_train[_]['v'] for _ in range(0, len(x_train), 1) if x_train[_]['id'] in classes]),-1)
    print(f'x_train.shape:{x_train.shape}')
    map2mrc(x_train[0],'small.mrc')

