import numpy as np
def pickley(filename):
    with open(filename, 'rb') as pickle_file:
        return pickle.load(pickle_file, encoding='latin1')
def pickled(o, path,protocol = -1):
    with open(path, 'wb') as f:
        pickle.dump(o, f, protocol=protocol)

def cluster(featurepath, templatepath):
    features = pickley(featurepath)
    

    dist = np.linalg.norm(point1 - point2)


filepath = f'/shared/home/c_myz/training/features/B=8_1000per5classes_snr-0.9.pickle'
features = pickley(filepath)
    

