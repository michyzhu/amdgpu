import sys

try:
    import pickle5 as pickle
except ImportError:
    import pickle

def pickley(filename):
    with open(filename, 'rb') as pickle_file:
        return pickle.load(pickle_file, encoding='latin1')

if __name__ == "__main__":
    numArgs = len(sys.argv)
    if(numArgs != 2):
        print(f'error: run with "python infoSimu.py datafile.pickle"')
        pass
    simu = pickley(sys.argv[1])
    try:
        print(f'This is a features file, post-extraction through Moco.\ninp: {simu["inp"]}\nop: {simu["op"]}\n')
    except KeyError:
        try:
            print(f'This is a data file, just simulated.\ninp: {simu["inp"]}')
        except KeyError:
            print('Sorry, wrong file format! It needs an "inp" field, generated using simu.py')
