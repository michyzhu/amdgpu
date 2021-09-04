import numpy as np
from sklearn.semi_supervised import LabelPropagation

# how many labeled points should we have in training?
numlabels = 3

def pickley(filename):
    with open(filename, 'rb') as pickle_file:
        return pickle.load(pickle_file, encoding='latin1')

#d = pickley('PICA_py3/fts.pickle')
#x = d['x']
#y = d['y']

x_train = np.load('val_outputs.npy')
y_train = np.load('val_targets.npy')
x_val = np.load('val_outputs.npy')
y_val = np.load('val_targets.npy')

# define label propagation and train
label_prop_model = LabelPropagation()

# unlabel most indices for training
labels = np.copy(y_train)
random_unlabeled_points = np.random.choice(x_train.shape[0],x_train.shape[0] - numlabels) 
labels[random_unlabeled_points] = -1

label_prop_model.fit(x_train, labels)

# validation
print(label_prop_model.score(x_val,y_val))
