import torch
import sys
if __name__ == "__main__":
    path = sys.argv[1]
    la = torch.load(path + '/checkpoint/latest.ckpt')
    print(f'latest: epoch {la["epoch"]}, acc: {la["acc"]}')
    be =  torch.load(path + '/checkpoint/best.ckpt') 
    print(f'best: epoch {be["epoch"]}, acc: {be["acc"]}')

