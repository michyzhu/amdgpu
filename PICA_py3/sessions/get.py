import torch
import sys
if __name__ == "__main__":
    path = sys.argv[1]
    ckpt = 'latest.ckpt'
    if(len(sys.argv) > 2): ckpt = 'best.ckpt'
    c = torch.load(path + '/checkpoint/' + ckpt)
    print(f'epoch: {c["epoch"]}, acc: {c["acc"]}')
